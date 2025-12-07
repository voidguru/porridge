import argparse
from storage import ParquetStore
from processor import compute_derived, kalman_smooth_positions
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Output, Input, State, callback_context
import dash_leaflet as dl
import plotly.graph_objs as go
import json, math, time

def downsample_df(df, max_points=20000):
    # naive downsample by selecting roughly equally spaced indices
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n-1, max_points, dtype=int)
    return df.iloc[idx].reset_index(drop=True)

def create_app(data_dir):
    store = ParquetStore(data_dir)
    sessions = store.list_sessions()
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2('Telemetry Viewer (Kalman-smoothed)'),
        dcc.Dropdown(id='session-dropdown', options=[{'label': s.split('/')[-1], 'value': s} for s in sessions], placeholder='Select a session'),
        html.Div(id='session-meta'),
        dl.Map([dl.TileLayer()], id='map', style={'width':'100%','height':'60vh'}),
        dcc.Graph(id='speed-graph'),
        dcc.Graph(id='accel-graph'),
        dcc.Graph(id='heading-graph'),
        html.Div([
            html.Button('Play', id='play-btn'),
            html.Button('Pause', id='pause-btn'),
            dcc.Slider(id='time-slider', min=0, max=1, step=1, value=0),
            dcc.Interval(id='interval', interval=200, n_intervals=0, disabled=True)
        ])
    ], style={'width':'95%', 'margin':'auto'})
    # callbacks
    @app.callback(Output('time-slider','max'), Input('session-dropdown','value'))
    def set_slider_max(path):
        if not path:
            return 1
        df = store.read_session(path)
        return max(1, len(df)-1)

    @app.callback(Output('map','children'),
                  Output('session-meta','children'),
                  Output('speed-graph','figure'),
                  Output('accel-graph','figure'),
                  Output('heading-graph','figure'),
                  Input('session-dropdown','value'))
    def load_session(path):
        if not path:
            raise dash.exceptions.PreventUpdate
        df = store.read_session(path)
        df = compute_derived(df)
        df = kalman_smooth_positions(df)
        # downsample for plotting
        ddf = downsample_df(df, max_points=10000)
        coords = ddf[['lat_k','lon_k']].values.tolist()
        poly = dl.Polyline(positions=coords, color='blue')
        # set marker at start
        start = coords[0] if len(coords)>0 else [0,0]
        marker = dl.Marker(position=start, id='vehicle-marker')
        # graphs
        fig_speed = go.Figure(go.Scatter(x=ddf['timestamp'], y=ddf['speed'], mode='lines'))
        fig_accel = go.Figure(go.Scatter(x=ddf['timestamp'], y=ddf['accel'], mode='lines'))
        fig_head = go.Figure(go.Scatter(x=ddf['timestamp'], y=ddf['bearing'], mode='lines'))
        meta = f"Points: {len(df)} | Duration: {df['dt'].sum():.1f}s"
        return [dl.TileLayer(), poly, marker], meta, fig_speed, fig_accel, fig_head

    # slider -> marker update
    @app.callback(Output('map','children'),
                  Input('time-slider','value'),
                  State('session-dropdown','value'))
    def update_marker(idx, path):
        if not path:
            raise dash.exceptions.PreventUpdate
        df = store.read_session(path)
        # read single row (avoid full copies)
        row = df.iloc[int(idx)]
        # use smoothed coords if present
        lat = row.get('lat_k', row['lat'])
        lon = row.get('lon_k', row['lon'])
        poly = dl.Polyline()  # keep poly empty; main poly is reloaded when session changes
        marker = dl.Marker(position=[lat, lon], id='vehicle-marker')
        return [dl.TileLayer(), poly, marker]

    @app.callback(Output('interval','disabled'),
                  Input('play-btn','n_clicks'),
                  Input('pause-btn','n_clicks'),
                  State('interval','disabled'))
    def play_pause(play_n, pause_n, disabled):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        button = ctx.triggered[0]['prop_id'].split('.')[0]
        if button == 'play-btn':
            return False
        else:
            return True

    @app.callback(Output('time-slider','value'),
                  Input('interval','n_intervals'),
                  State('time-slider','value'),
                  State('session-dropdown','value'),
                  State('interval','interval'))
    def tick(n, current, path, interval):
        if not path:
            raise dash.exceptions.PreventUpdate
        df = store.read_session(path)
        # advance index based on interval and approximate sample rate
        step = max(1, int((interval/1000.0) * 1))  # simple: 1 index per tick
        new = min(len(df)-1, int(current) + step)
        return new

    return app

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data', help='directory with parquet sessions')
    args = p.parse_args()
    app = create_app(args.data_dir)
    app.run_server(host='0.0.0.0', port=8050)
