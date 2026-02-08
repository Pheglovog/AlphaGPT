import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")

        # 验证 limit_tokens 参数
        try:
            limit_tokens = int(limit_tokens)
            limit_tokens = max(1, min(limit_tokens, 10000))  # 限制在 1-10000 之间
        except (ValueError, TypeError):
            limit_tokens = 500

        top_query = f"""
        SELECT address FROM tokens
        LIMIT {limit_tokens}
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")

        # 验证并转义地址字符串
        valid_addrs = [addr.replace("'", "") for addr in addrs if addr]
        if not valid_addrs:
            raise ValueError("No valid token addresses found.")
        addr_str = "'" + "','".join(valid_addrs) + "'"

        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.fillna(method='ffill').fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")