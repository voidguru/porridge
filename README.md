# Dash Telemetry Viewer (Full)

Features:
- Stream/parse large GPX files (iterparse) and convert to Parquet (chunked) to handle >500 MB GPX files.
- Kalman filter smoothing of GPS trajectory (2D constant velocity model via `filterpy`).
- Compute speed (m/s), acceleration (m/s^2), heading (degrees to North 0-360), and approximate g-force.
- Dash app (dash + dash-leaflet + plotly) with trajectory overlay, moving marker replay, and synchronized plots.
- Downsampling for visualization to avoid sending millions of points to the browser.

Run:
1. Create environment and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Convert a large GPX to parquet (chunked):
   ```bash
   python import_gpx.py path/to/large.gpx data/session_YYYYMMDDHHMM.parquet
   ```
   This will stream the GPX and emit a Parquet file suitable for the dashboard.
3. Start the app:
   ```bash
   python app.py --data-dir data
   ```

Notes:
- The GPX importer uses an iterative XML parser (lxml.iterparse) to avoid loading entire GPX into memory.
- For very large sessions, the UI will downsample the trajectory for plotting but the playback uses the full parquet file on the server side.
