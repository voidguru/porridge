import pandas as pd
from pathlib import Path

class ParquetStore:
    def __init__(self, base_dir='data'):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def read_session(self, path):
        return pd.read_parquet(path)

    def list_sessions(self):
        return [str(p) for p in sorted(self.base.glob('*.parquet'))]
