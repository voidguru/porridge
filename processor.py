"""Telemetry processing utilities: smoothing, metrics, and Kalman filter."""
import numpy as np
import pandas as pd
from math import radians, degrees, atan2, sin, cos, sqrt
from filterpy.kalman import KalmanFilter
from pyproj import Geod

geod = Geod(ellps='WGS84')

def haversine_meters(lat1, lon1, lat2, lon2):
    # approximate haversine (meters)
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def bearing_deg(lat1, lon1, lat2, lon2):
    # returns degrees to North 0-360
    lon1r, lat1r, lon2r, lat2r = map(radians, [lon1, lat1, lon2, lat2])
    y = sin(lon2r - lon1r) * cos(lat2r)
    x = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(lon2r - lon1r)
    brng = (degrees(atan2(y, x)) + 360.0) % 360.0
    return brng

def compute_derived(df: pd.DataFrame, time_col='timestamp'):
    df = df.copy()
    # ensure timestamps
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)
    df = df.reset_index(drop=True)
    # compute dt (seconds)
    df['dt'] = df[time_col].diff().dt.total_seconds().fillna(0.0)
    # compute distance between consecutive points (meters)
    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()
    dist = np.zeros(len(df), dtype=float)
    bearing = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        dist[i] = haversine_meters(lat[i-1], lon[i-1], lat[i], lon[i])
        bearing[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    df['dist_m'] = dist
    df['bearing'] = bearing
    # speed m/s (computed from dist/dt) if not provided
    df['speed'] = df['speed'].fillna(df['dist_m'] / df['dt'].replace(0, np.nan)).fillna(0.0)
    # compute acceleration (dv/dt)
    df['dv'] = df['speed'].diff().fillna(0.0)
    df['accel'] = df['dv'] / df['dt'].replace(0, np.nan)
    df['accel'] = df['accel'].fillna(0.0)
    # heading rate (deg/s) and convert to rad/s
    df['dheading'] = np.deg2rad(np.unwrap(np.deg2rad(df['bearing'].fillna(method='ffill').to_numpy())))
    # approximate heading_rate in rad/s
    dheading = np.zeros(len(df))
    for i in range(1, len(df)):
        dheading[i] = (df['bearing'].iloc[i] - df['bearing'].iloc[i-1]) * np.pi/180.0 / max(df['dt'].iloc[i], 1e-6)
    df['heading_rate'] = dheading
    # lateral acceleration approx: v * heading_rate (v in m/s, heading_rate in rad/s) -> a_lat = v * omega
    df['a_lat'] = df['speed'] * df['heading_rate']
    # total accel magnitude (combine measured accel columns if present, else combine longitudinal and lateral)
    if {'ax','ay','az'}.issubset(df.columns):
        # if real accelerometer columns present, use their magnitude
        df['gforce'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2) / 9.80665
    else:
        df['a_total'] = np.sqrt(df['accel']**2 + df['a_lat']**2)
        df['gforce'] = df['a_total'] / 9.80665
    return df

def kalman_smooth_positions(df: pd.DataFrame, pos_cols=('lat','lon'), time_col='timestamp'):
    """Apply a simple 2D constant-velocity Kalman filter to lat/lon using filterpy.
    Returns a new DataFrame with 'lat_k' and 'lon_k' smoothed.
    """
    df = df.copy().reset_index(drop=True)
    # Convert lat/lon to meters using local projection via Geod to get displacements
    # We'll build a local ENU by projecting relative to first point
    lat0 = df.loc[0, pos_cols[0]]
    lon0 = df.loc[0, pos_cols[1]]
    # approximate conversion: use small-distance flat-earth conversion (meters per deg)
    # use geod to get accurate forward calculations for each point relative to origin
    xs = np.zeros(len(df))
    ys = np.zeros(len(df))
    for i in range(len(df)):
        # compute forward distance and azimuth from origin to point, then decompose
        fwd_az, back_az, dist = geod.inv(lon0, lat0, df.loc[i,pos_cols[1]], df.loc[i,pos_cols[0]])
        # decompose into x east, y north
        theta = np.deg2rad(fwd_az)
        xs[i] = dist * np.sin(theta)
        ys[i] = dist * np.cos(theta)
    # Kalman filter on x,y with constant-velocity model (4D state: x, vx, y, vy)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt_median = np.nanmedian(df[time_col].diff().dt.total_seconds().fillna(0.0).replace(0,np.nan))
    if np.isnan(dt_median) or dt_median<=0:
        dt_median = 1.0
    # State transition
    kf.F = np.array([[1, dt_median, 0, 0],
                     [0, 1,         0, 0],
                     [0, 0,         1, dt_median],
                     [0, 0,         0, 1]])
    kf.H = np.array([[1,0,0,0],
                     [0,0,1,0]])
    # Covariances
    kf.R *= 5.0  # measurement noise
    kf.P *= 10.0
    kf.Q = np.eye(4) * 0.01
    # initial state
    kf.x = np.array([xs[0], 0.0, ys[0], 0.0])
    xs_k = np.zeros_like(xs)
    ys_k = np.zeros_like(ys)
    for i in range(len(xs)):
        z = np.array([xs[i], ys[i]])
        kf.predict()
        kf.update(z)
        xs_k[i] = kf.x[0]
        ys_k[i] = kf.x[2]
    # convert back to lat/lon from local meters by inverse geod
    lat_k = np.zeros(len(xs_k))
    lon_k = np.zeros(len(xs_k))
    for i in range(len(xs_k)):
        # compute az and dist from origin
        dist = np.hypot(xs_k[i], ys_k[i])
        if dist < 1e-6:
            lat_k[i] = lat0
            lon_k[i] = lon0
            continue
        theta = atan2(xs_k[i], ys_k[i])  # theta measured from north clockwise when using sin for x
        az = degrees(theta)
        # geod.fwd expects lon, lat, azimuth
        lon_k[i], lat_k[i], _ = geod.fwd(lon0, lat0, az, dist)
    df['lat_k'] = lat_k
    df['lon_k'] = lon_k
    return df
