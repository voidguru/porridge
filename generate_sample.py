import pandas as pd, numpy as np
from datetime import datetime, timedelta
import sys
def gen(path='data/sample.parquet', n=5000):
    t0 = datetime.utcnow()
    times = [t0 + timedelta(seconds=i) for i in range(n)]
    # simple circular motion
    lat0, lon0 = 37.0, -122.0
    r = 0.01  # degrees ~1km
    lats = [lat0 + r * np.cos(2*np.pi*i/n) for i in range(n)]
    lons = [lon0 + r * np.sin(2*np.pi*i/n) for i in range(n)]
    speeds = np.random.uniform(5,15,size=n)
    df = pd.DataFrame({'timestamp':times,'lat':lats,'lon':lons,'ele':[0]*n,'speed':speeds})
    df.to_parquet(path)
    print('wrote',path)
if __name__=='__main__':
    gen()
