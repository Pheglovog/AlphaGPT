#!/usr/bin/env python3
"""
AlphaQuant 真实数据训练脚本
集成 Tushare Pro API 加载真实市场数据
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm
import pickle
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphaquant.model.alpha_quant import AlphaQuant, ModelConfig
from alphaquant.factors.china_factors import ChinaFactorEngine
from alphaquant.data_providers.tushare import TushareProProvider as TushareDataProvider


class RealFactorDataset(Dataset):
    """真实因子数据集"""

    def __init__(
        self,
        features: torch.Tensor,
        market_sentiment: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        stock_ids: List[int],
        dates: List[str],
        sequence_length: int = 60
    ):
        self.features = features
        self.market_sentiment = market_sentiment
        self.targets = targets
        self.stock_ids = stock_ids
        self.dates = dates
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.features.shape[0] - self.sequence_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取序列
        feat_seq = self.features[idx:idx + self.sequence_length]
        sent = self.market_sentiment[idx + self.sequence_length - 1]

        # 目标
        target_return = self.targets['return'][idx + self.sequence_length]
        target_sharpe = self.targets['sharpe'][idx + self.sequence_length]
        target_drawdown = self.targets['drawdown'][idx + self.sequence_length]

        return {
            'features': feat_seq,
            'market_sentiment': sent,
            'target_return': target_return,
            'target_sharpe': target_sharpe,
            'target_drawdown': target_drawdown,
            'stock_id': self.stock_ids[idx + self.sequence_length],
            'date': self.dates[idx + self.sequence_length]
        }


class RealDataLoader:
    """真实数据加载器

    特性:
    - 从 Tushare Pro API 加载真实市场数据
    - 计算技术因子
    - 数据验证和清洗
    - 缓存机制避免重复请求
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = './cache',
        sequence_length: int = 60
    ):
        self.token = token or os.getenv('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("TUSHARE_TOKEN 环境变量未设置")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = sequence_length
        self.factor_engine = ChinaFactorEngine()

        logger.info(f"RealDataLoader initialized (cache: {self.cache_dir})")

    def _get_cache_path(self, stock_id: int, start_date: str, end_date: str) -> Path:
        """获取缓存文件路径"""
        filename = f"{stock_id}_{start_date}_{end_date}.pkl"
        return self.cache_dir / filename

    def _load_stock_data(
        self,
        stock_id: int,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """加载单只股票数据（带缓存）"""
        cache_path = self._get_cache_path(stock_id, start_date, end_date)

        # 尝试从缓存加载
        if use_cache and cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # 从 Tushare 加载
        try:
            provider = TushareDataProvider(token=self.token)

            # 转换股票代码格式
            ts_code = f"{stock_id:06d}.SZ" if stock_id < 600000 else f"{stock_id:06d}.SH"

            # 异步加载数据
            async def load_data():
                async with provider:
                    df = await provider.get_daily_quotes(
                        ts_code=ts_code,
                        start_date=start_date.replace('-', ''),
                        end_date=end_date.replace('-', '')
                    )
                    return df

            # 运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(load_data())
            loop.close()

            if df is None or len(df) < self.sequence_length + 10:
                logger.warning(f"Insufficient data for stock {stock_id}")
                return None

            # 数据清洗
            df = self._clean_data(df)

            # 缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)

            return df

        except Exception as e:
            logger.error(f"Failed to load data for stock {stock_id}: {e}")
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 去除重复
        df = df.drop_duplicates(subset=['trade_date'])

        # 排序
        df = df.sort_values('trade_date')

        # 填充缺失值 (Pandas 3.0 兼容)
        df = df.ffill().bfill()

        # 验证数据
        if len(df) < self.sequence_length + 10:
            raise ValueError(f"Insufficient data after cleaning: {len(df)}")

        return df

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        try:
            # 添加基础价格因子
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_ratio'] = df['vol'] / df['vol'].rolling(20).mean()

            # 添加简单技术指标
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])

            return df

        except Exception as e:
            logger.error(f"Failed to calculate factors: {e}")
            return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI 指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_targets(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[float, float, float]:
        """计算目标值（未来收益、夏普、回撤）"""

        # 未来 5 日收益
        future_returns = df['returns'].iloc[idx+1:idx+6].sum()

        # 未来 20 日夏普比率
        window_returns = df['returns'].iloc[idx+1:idx+21]
        if len(window_returns) < 20:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (window_returns.mean() / (window_returns.std() + 1e-6))

        # 未来 20 日最大回撤
        future_prices = df['close'].iloc[idx+1:idx+21]
        if len(future_prices) < 5:
            max_drawdown = 0.0
        else:
            peak = future_prices.iloc[0]
            max_drawdown = -min([(price - peak) / peak for price in future_prices])

        return future_returns, sharpe_ratio, max_drawdown

    def load_dataset(
        self,
        stock_ids: List[int],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[RealFactorDataset]:
        """加载完整数据集

        Args:
            stock_ids: 股票代码列表（如 [1, 2, 3, 600000]）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_cache: 是否使用缓存

        Returns:
            RealFactorDataset
        """
        logger.info(f"Loading real data for {len(stock_ids)} stocks...")

        all_features = []
        all_sentiments = []
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_stock_ids = []
        all_dates = []

        for stock_id in stock_ids:
            logger.info(f"Loading stock {stock_id}...")

            # 加载数据
            df = self._load_stock_data(stock_id, start_date, end_date, use_cache)
            if df is None:
                continue

            # 计算因子
            df = self._calculate_factors(df)

            # 确保有足够的因子列
            factor_cols = [col for col in df.columns if col.startswith('factor_')]
            if not factor_cols:
                logger.warning(f"No factors found for stock {stock_id}")
                continue

            # 构建样本
            num_samples = len(df) - self.sequence_length - 20
            for i in range(num_samples):
                # 特征
                feature_seq = df[factor_cols].iloc[i:i+self.sequence_length].values
                feature_seq = (feature_seq - feature_seq.mean(axis=0)) / (feature_seq.std(axis=0) + 1e-6)

                # 市场情绪（使用量价因子）
                sentiment = df[['volume_ratio', 'volatility']].iloc[i+self.sequence_length-1].values
                sentiment = np.concatenate([sentiment, np.zeros(13)])  # 填充到 15 维

                # 目标
                target_return, target_sharpe, target_drawdown = self._calculate_targets(df, i + self.sequence_length - 1)

                # 添加到列表
                all_features.append(feature_seq)
                all_sentiments.append(sentiment)
                all_returns.append(target_return)
                all_sharpes.append(target_sharpe)
                all_drawdowns.append(target_drawdown)
                all_stock_ids.append(stock_id)
                all_dates.append(df['trade_date'].iloc[i + self.sequence_length - 1])

        if not all_features:
            logger.error("No valid samples loaded")
            return None

        # 转换为张量
        features = torch.tensor(np.array(all_features), dtype=torch.float32)
        market_sentiment = torch.tensor(np.array(all_sentiments), dtype=torch.float32)

        targets = {
            'return': torch.tensor(np.array(all_returns), dtype=torch.float32),
            'sharpe': torch.tensor(np.array(all_sharpes), dtype=torch.float32),
            'drawdown': torch.tensor(np.array(all_drawdowns), dtype=torch.float32)
        }

        logger.info(f"Loaded {len(features)} samples")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Sentiment shape: {market_sentiment.shape}")

        return RealFactorDataset(
            features,
            market_sentiment,
            targets,
            all_stock_ids,
            all_dates,
            sequence_length=self.sequence_length
        )


class Trainer:
    """训练器（支持真实数据）"""

    def __init__(
        self,
        config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        patience: int = 10,
        min_delta: float = 1e-6
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 早停机制
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        logger.info(f"Using device: {self.device}")

        # 创建模型
        self.model = AlphaQuant(config).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 损失函数
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0
        total_mse_loss = 0
        total_l1_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()

            # 移动到设备
            features = batch['features'].to(self.device)
            sentiment = batch['market_sentiment'].to(self.device)
            target_return = batch['target_return'].to(self.device)
            target_sharpe = batch['target_sharpe'].to(self.device)
            target_drawdown = batch['target_drawdown'].to(self.device)

            # 前向传播
            output = self.model(features, sentiment)

            # 计算损失（使用真实目标）
            mse_loss = self.criterion_mse(output['value'], target_return)
            l1_loss = self.criterion_l1(output['value'], target_return)

            # 风险调整损失
            # 惩罚预测为正但实际为负的情况
            risk_penalty = torch.where(
                (output['value'] > 0) & (target_return < 0),
                torch.abs(target_return) * 2,
                torch.tensor(0.0, device=self.device)
            )
            risk_loss = risk_penalty.mean()

            loss = mse_loss + 0.5 * l1_loss + 0.1 * risk_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_l1_loss += l1_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return {
            'loss': total_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader),
            'l1_loss': total_l1_loss / len(dataloader)
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0
        total_mse_loss = 0
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            features = batch['features'].to(self.device)
            sentiment = batch['market_sentiment'].to(self.device)
            target_return = batch['target_return'].to(self.device)

            output = self.model(features, sentiment)

            mse_loss = self.criterion_mse(output['value'], target_return)
            loss = mse_loss

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()

            all_predictions.extend(output['value'].cpu().numpy())
            all_targets.extend(target_return.cpu().numpy())

        # 计算相关性
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

        return {
            'loss': total_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader),
            'correlation': correlation
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = './checkpoints'
    ):
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Starting training (epochs: {num_epochs})")

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Training Epoch {epoch}/{num_epochs}")

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # 打印
            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Correlation: {val_metrics['correlation']:.4f}"
            )

            # 早停检查
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.counter = 0
                self.save_checkpoint(save_dir, epoch, val_metrics)
                logger.info(f"✅ Saved best model (val_loss: {self.best_val_loss:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"⏰ Early stopping at epoch {epoch}")
                    self.early_stop = True
                    break

            if self.early_stop:
                break

        logger.info("Training completed!")

    def save_checkpoint(self, save_dir: str, epoch: int, val_metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_correlation': val_metrics['correlation'],
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }

        path = os.path.join(save_dir, f'best_model_realdata_epoch{epoch}.pt')
        torch.save(checkpoint, path)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train AlphaQuant with Real Data')
    parser.add_argument('--token', type=str, help='Tushare API token')
    parser.add_argument('--stocks', type=str, default='1,2,3,600000,600519', help='Stock IDs (comma-separated)')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_real', help='Save directory')
    parser.add_argument('--cache-dir', type=str, default='./cache_real', help='Cache directory')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')

    args = parser.parse_args()

    # 解析股票代码
    stock_ids = [int(s) for s in args.stocks.split(',')]

    # 模型配置
    config = ModelConfig(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_basic_factors=6,
        num_advanced_factors=18,
        max_formula_len=64
    )

    # 加载真实数据
    logger.info("Loading real data from Tushare...")
    data_loader = RealDataLoader(
        token=args.token,
        cache_dir=args.cache_dir,
        sequence_length=60
    )

    dataset = data_loader.load_dataset(
        stock_ids=stock_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache
    )

    if dataset is None:
        logger.error("Failed to load dataset")
        sys.exit(1)

    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # 训练
    trainer = Trainer(
        config,
        learning_rate=args.lr,
        device=args.device,
        patience=10
    )

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
