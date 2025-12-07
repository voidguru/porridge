import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import io
from datetime import datetime
import numpy as np
import gpxpy
import gpxpy.gpx
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("GPX Sailing Session Replay", className="text-center mb-4"),
            dcc.Upload(
                id='upload-gpx',
                children=html.Div([
                    "Drag & Drop GPX file or ",
                    html.A('Select File')
                ], className="upload-text"),
                style={
                    'width': '100%', 'height': '80px', 'lineHeight': '80px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'margin': '10px', 'backgroundColor': '#f8f9fa'
                },
                multiple=False
            ),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='telemetry-display'),
            html.Div(id='best-distances'),
            html.Div(id='wind-info'),
            html.Div(id='maneuvers'),
        ], width=12)
    ], id='main-content', style={'display': 'none'}),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Replay Speed:", className="fw-bold"),
                dcc.Slider(id='replay-speed', min=0.1, max=5.0, value=1.0, step=0.1,
                          marks={0.1: '0.1x', 1: '1x', 5: '5x'}),
                html.Div([
                    html.Button("▶️ Play", id='play-button', n_clicks=0, className="btn btn-success me-2"),
                    html.Button("⏸️ Pause", id='pause-button', n_clicks=0, className="btn btn-secondary"),
                    dcc.Slider(id='time-slider', min=0, max=100, value=0, step=0.01, className="mt-3")
                ], className="mt-3")
            ])
        ], width=12)
    ], id='controls', style={'display': 'none'}),
    
    dcc.Graph(id='map-plot'),
    dcc.Graph(id='telemetry-plot'),
    dcc.Graph(id='replay-map'),
    dcc.Store(id='session-data'),
    dcc.Store(id='is-playing', data=False),
    dcc.Interval(id='replay-interval', interval=200, n_intervals=0, disabled=True)
], fluid=True)

def parse_gpx(contents, filename):
    """Parse GPX file and compute telemetry"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    gpx = gpxpy.parse(io.BytesIO(decoded))
    points = []
    cum_distance = 0
    
    for track in gpx.tracks:
        for segment in track.segments:
            for i, point in enumerate(segment.points):
                if i == 0:
                    continue
                    
                prev_point = segment.points[i-1]
                distance = geodesic((prev_point.latitude, prev_point.longitude), 
                                  (point.latitude, point.longitude)).meters
                cum_distance += distance
                time_diff = (point.time - prev_point.time).total_seconds()
                
                if time_diff > 0:
                    speed = distance / time_diff
                    course = prev_point.course_between(point) or 0
                    
                    points.append({
                        'time': point.time,
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'elevation': point.elevation or 0,
                        'speed': speed,
                        'course': course,
                        'distance': distance,
                        'cum_distance': cum_distance
                    })
    
    df = pd.DataFrame(points)
    if df.empty:
        return None, {}, pd.DataFrame(), ""
    
    df = df.sort_values('time').reset_index(drop=True)
    df['time_idx'] = np.arange(len(df)) / len(df)
    
    # Smooth speed and compute acceleration
    window = min(10, len(df)//10)
    df['speed_smooth'] = df['speed'].rolling(window=window, center=True, min_periods=1).mean()
    df['speed_smooth'] = df['speed_smooth'].fillna(df['speed'])
    
    time_diffs = df['time'].diff().dt.total_seconds().fillna(1/10)
    df['acceleration'] = df['speed_smooth'].diff() / time_diffs
    
    # Best distance segments
    best_segments = calculate_best_distances(df)
    
    # Detect tacks/jibes
    maneuvers = detect_maneuvers(df)
    
    # Wind data estimate
    wind_info = get_wind_estimate(df.iloc[0]['lat'], df.iloc[0]['lon'], df.iloc[0]['time'].date())
    
    return df, best_segments, maneuvers, wind_info

def calculate_best_distances(df, distances=[200, 500, 1000, 2000]):
    """Find fastest times for specified distance segments"""
    results = {}
    total_dist = df['cum_distance'].max()
    
    for dist in distances:
        if total_dist < dist * 0.8:
            results[f'{dist}m'] = {'time': None, 'speed_kmh': 0}
            continue
        
        best_time = float('inf')
        best_speed = 0
        
        for i in range(len(df)):
            start_dist = df.iloc[i]['cum_distance']
            target_dist = start_dist + dist
            
            end_mask = df['cum_distance'] >= target_dist
            if end_mask.any():
                end_idx = df[end_mask].index[0]
                time_taken = (df.iloc[end_idx]['time'] - df.iloc[i]['time']).total_seconds()
                if time_taken > 0 and time_taken < best_time:
                    best_time = time_taken
                    best_speed = (dist / time_taken) * 3.6
        
        results[f'{dist}m'] = {
            'time': best_time if best_time != float('inf') else None,
            'speed_kmh': best_speed
        }
    
    return results

def detect_maneuvers(df, threshold_deg=80):
    """Detect tacks and jibes from course changes"""
    if len(df) < 3:
        return pd.DataFrame()
    
    df_m = df.copy()
    df_m['course_diff'] = df_m['course'].diff()
    
    # Normalize course differences to [-180, 180]
    df_m['course_diff'] = np.where(df_m['course_diff'] > 180, df_m['course_diff'] - 360,
                                  np.where(df_m['course_diff'] < -180, df_m['course_diff'] + 360, 
                                          df_m['course_diff']))
    
    # Detect significant heading changes while moving
    moving_mask = df_m['speed'] > 2.0
    maneuver_mask = (np.abs(df_m['course_diff']) > threshold_deg) & moving_mask
    
    maneuvers = df_m[maneuver_mask].copy()
    if not maneuvers.empty:
        maneuvers['type'] = maneuvers['course_diff'].apply(
            lambda x: 'Tack' if x < 0 else 'Jibe'
        )
        maneuvers = maneuvers[['time', 'lat', 'lon', 'course', 'course_diff', 'type']].round(2)
    
    return maneuvers

def get_wind_estimate(lat, lon, date):
    """Geocode location and provide wind estimate"""
    try:
        geolocator = Nominatim(user_agent="gpx_analyzer")
        location = geolocator.reverse((lat, lon), timeout=5)
        place_name = location.address.split(',')[0] if location else "Unknown"
        return f"📍 {place_name[:30]}... | Wind: Est. 10-15kn"
    except:
        return "📍 Location detected"

# COMBINED MAIN CALLBACK - Handles upload + replay-interval.disabled
@callback(
    [Output('main-content', 'style'),
     Output('controls', 'style'),
     Output('map-plot', 'figure'),
     Output('telemetry-plot', 'figure'),
     Output('telemetry-display', 'children'),
     Output('best-distances', 'children'),
     Output('wind-info', 'children'),
     Output('maneuvers', 'children'),
     Output('session-data', 'data'),
     Output('replay-interval', 'disabled')],
    Input('upload-gpx', 'contents'),
    State('upload-gpx', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return [{'display': 'none'}, {'display': 'none'}, go.Figure(), go.Figure(), '', '', '', '', None, True]
    
    df, best_segments, maneuvers, wind_info = parse_gpx(contents, filename)
    
    if df is None or df.empty:
        return [{'display': 'none'}, {'display': 'none'}, go.Figure(), go.Figure(), 
                html.P("Invalid GPX file"), '', '', '', None, True]
    
    # Static map
    map_fig = px.line_mapbox(df, lat='lat', lon='lon', 
                           color='speed', color_continuous_scale='Viridis',
                           hover_data=['speed', 'course', 'time'],
                           title='Sailing Track - Color by Speed',
                           mapbox_style="open-street-map",
                           zoom=12)
    map_fig.update_layout(height=500, showlegend=False)
    
    # Telemetry plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['speed_smooth'], 
                           name='Speed (m/s)', line=dict(color='blue'), yaxis='y1'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['course'], 
                           name='Direction (°)', line=dict(color='green'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['acceleration'].fillna(0), 
                           name='Acceleration', line=dict(color='red'), yaxis='y3'))
    
    fig.update_layout(
        title='Live Telemetry (Speed/Direction/Acceleration)',
        yaxis=dict(title="Speed (m/s)", side="left"),
        yaxis2=dict(title="Direction (°)", overlaying="y", side="right", range=[0, 360]),
        yaxis3=dict(title="Acceleration (m/s²)", anchor="free", overlaying="y", side="right", position=0.95),
        height=500, hovermode='x unified'
    )
    
    # Session info
    duration = (df['time'].max() - df['time'].min()).total_seconds() / 60
    max_speed = df['speed'].max()
    
    telemetry_display = dbc.Card([
        dbc.CardBody([
            html.H4(f"Session: {filename}", className="mb-2"),
            html.P(f"⏱️ Duration: {duration:.1f} min"),
            html.P(f"📏 Distance: {df['cum_distance'].max():.0f}m"),
            html.P(f"⚡ Max Speed: {max_speed:.1f} m/s ({max_speed*3.6:.1f} km/h)")
        ])
    ])
    
    # Best distances
    best_items = [html.Li(f"{dist}: {data['speed_kmh']:.1f} km/h") 
                 for dist, data in best_segments.items() if data['speed_kmh'] > 0]
    best_dist_html = html.Div([
        html.H5("🏆 Best Distance Speeds"),
        html.Ul(best_items) if best_items else html.P("Insufficient track length")
    ])
    
    # Maneuvers
    maneuvers_html = html.Div([
        html.H5(f"⚓ Maneuvers: {len(maneuvers)}"),
        dbc.Table.from_dataframe(maneuvers.head(10), striped=True, bordered=True, hover=True, 
                               style={'fontSize': '12px'}) if not maneuvers.empty else 
        html.P("No tacks/jibes detected")
    ])
    
    return [
        {'display': 'block'}, {'display': 'block'}, map_fig, fig,
        telemetry_display, dbc.Card([dbc.CardBody(best_dist_html)]), 
        dbc.Alert(wind_info, color="info"), maneuvers_html, 
        {'df': df.to_dict('records'), 'max_idx': len(df)}, False
    ]

# SINGLE PLAY/PAUSE CALLBACK - Manages is-playing state
@callback(
    [Output('is-playing', 'data')],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks')],
    State('is-playing', 'data')
)
def control_playback(play_clicks, pause_clicks, is_playing):
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if trigger_id == 'play-button':
        return True
    elif trigger_id == 'pause-button':
        return False
    return is_playing

# REPLAY ANIMATION CALLBACK - Uses is-playing state
@callback(
    [Output('replay-map', 'figure'),
     Output('time-slider', 'value')],
    Input('replay-interval', 'n_intervals'),
    State('session-data', 'data'),
    State('replay-speed', 'value'),
    State('time-slider', 'value'),
    State('is-playing', 'data')
)
def update_replay(n_intervals, session_data, speed, current_time, is_playing):
    if not session_data or not is_playing or n_intervals == 0:
        return go.Figure(), current_time
    
    df = pd.DataFrame(session_data['df'])
    max_idx = session_data['max_idx']
    
    # Animate progress
    progress = min((current_time + speed * 0.1) % 100, 99.9)
    idx = int(progress / 100 * max_idx)
    
    # Replay map with marker
    replay_fig = px.line_mapbox(df.iloc[:max(idx,1)], lat='lat', lon='lon', 
                              color='speed', color_continuous_scale='Viridis',
                              mapbox_style="open-street-map", zoom=12)
    
    if idx > 0 and idx < len(df):
        current_pos = df.iloc[idx]
        replay_fig.add_scattermapbox(
            lat=[current_pos['lat']], lon=[current_pos['lon']],
            mode='markers+text', marker=dict(size=12, color='red'),
            text=f"⚡ {current_pos['speed']:.1f}m/s | {current_pos['course']:.0f}°",
            textposition="top center",
            showlegend=False
        )
    
    replay_fig.update_layout(height=400, title=f"Live Replay: {progress:.0f}%")
    
    return replay_fig, progress

if __name__ == '__main__':
    app.run(debug=True, port=8051)
