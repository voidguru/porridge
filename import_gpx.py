#!/usr/bin/env python3
"""Stream-parse GPX and write Parquet (chunked)."""
import sys
from lxml import etree
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

NS = '{http://www.topografix.com/GPX/1/1}'

def parse_gpx_to_parquet(gpx_path, out_parquet, chunk_size=100000):
    """Parse GPX trkpt elements streaming and write to Parquet in chunks."""
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = etree.iterparse(str(gpx_path), events=('end',), tag=NS + 'trkpt')
    cols = ['timestamp','lat','lon','ele','speed','bearing','ax','ay','az']
    buf = {c: [] for c in cols}
    n = 0
    for _, elem in reader:
        lat = float(elem.get('lat'))
        lon = float(elem.get('lon'))
        ele = None
        time = None
        speed = None
        bearing = None
        # child elements
        for ch in elem:
            tag = etree.QName(ch.tag).localname
            text = ch.text
            if tag == 'ele' and text:
                ele = float(text)
            elif tag == 'time' and text:
                try:
                    time = pd.to_datetime(text)
                except Exception:
                    time = None
            elif tag.lower() in ('speed',):  # some GPX variants
                try:
                    speed = float(text)
                except Exception:
                    speed = None
        # append
        buf['timestamp'].append(time if time is not None else pd.NaT)
        buf['lat'].append(lat)
        buf['lon'].append(lon)
        buf['ele'].append(ele)
        buf['speed'].append(speed)
        buf['bearing'].append(bearing)
        buf['ax'].append(np.nan)
        buf['ay'].append(np.nan)
        buf['az'].append(np.nan)
        n += 1
        # clear element to save memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if n % chunk_size == 0:
            df = pd.DataFrame(buf)
            # write to parquet append-mode via pyarrow
            table = pa.Table.from_pandas(df)
            if not out_path.exists():
                pq.write_table(table, out_path)
            else:
                pq.write_table(table, out_path, append=True)
            # reset buffer
            buf = {c: [] for c in cols}
            print(f"Wrote {n} points...", flush=True)
    # final flush
    if len(buf['lat']) > 0:
        df = pd.DataFrame(buf)
        table = pa.Table.from_pandas(df)
        if not out_path.exists():
            pq.write_table(table, out_path)
        else:
            pq.write_table(table, out_path, append=True)
    print(f"Finished. Total points: {n}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('gpx', help='input GPX file path')
    p.add_argument('out', help='output parquet path (e.g. data/session.parquet)')
    p.add_argument('--chunk', type=int, default=100000, help='chunk size for writes')
    args = p.parse_args()
    parse_gpx_to_parquet(args.gpx, args.out, chunk_size=args.chunk)
