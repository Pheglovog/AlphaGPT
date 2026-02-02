"""
AlphaQuant 模型训练脚本（优化版）
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphaquant.model.alpha_quant import AlphaQuant, ModelConfig
from alphaquant.factors.china_factors import ChinaFactorEngine
from alphaquant.backtest.backtester import BacktestEngine
from alphaquant.data_validation import DataValidator, DataCleaner, DataQualityAnalyzer
from alphaquant.data_cache import DataCache
from alphaquant.metrics import TrainingMetrics, PerformanceMonitor, EarlyStoppingMonitor


class FactorDataset(Dataset):
    """因子数据集"""

    def __init__(
        self,
        features: torch.Tensor,
        market_sentiment: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        sequence_length: int = 60
    ):
        self.features = features
        self.market_sentiment = market_sentiment
        self.targets = targets
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
            'target_drawdown': target_drawdown
        }


class RealDataLoader:
    """真实数据加载器（从 Tushare 加载）"""

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = "./data_cache"
    ):
        self.token = token or os.getenv("TUSHARE_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 数据验证器
        self.validator = DataValidator(
            min_price=0.01,
            max_price=10000.0,
            min_volume=100,
            min_return=-0.20,
            max_return=0.20
        )

        # 数据清洗器
        self.cleaner = DataCleaner()

        # 数据质量分析器
        self.analyzer = DataQualityAnalyzer()

        # 数据缓存
        self.cache = DataCache()

    def load_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        从 Tushare 加载股票数据（支持缓存）

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            历史行情 DataFrame
        """
        params = {
            'ts_code': ts_code,
            'start_date': start_date,
            'end_date': end_date
        }

        # 尝试从缓存加载
        if use_cache:
            cached_data = self.cache.get(params)
            if cached_data is not None:
                logger.info(f"从缓存加载数据：{ts_code}")
                return cached_data

        # 模拟从 Tushare 加载数据
        # 实际应该调用 TushareProProvider.get_daily_quotes()
        logger.info(f"从 Tushare 加载数据：{ts_code}")

        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        # 生成模拟数据（实际应该从 API 获取）
        np.random.seed(int(hash(ts_code) % 2**32))

        df = pd.DataFrame({
            'trade_date': dates,
            'open': 10 + np.random.randn(n) * 2,
            'high': 12 + np.random.randn(n) * 2,
            'low': 8 + np.random.randn(n) * 2,
            'close': 11 + np.random.randn(n) * 2,
            'vol': np.random.randint(100000, 1000000, n),
            'amount': np.random.randint(10000000, 100000000, n),
            'pct_chg': np.random.randn(n) * 5,
            'pct_chg': np.random.randn(n) * 0.5,  # 真实的 pct_chg
            'pre_close': 11 + np.random.randn(n) * 2,  # 前一日收盘价
            'turnover_rate': np.random.rand(n) * 5,  # 换手率
            'pe_ratio': 10 + np.random.rand(n) * 10,    # 市盈率
            'pb_ratio': 1.0 + np.random.rand(n) * 0.5, # 市净率
            'total_mv': np.random.rand(n) * 1000000,  # 总市值
            'circ_mv': np.random.rand(n) * 500000      # 流通市值
        })

        # 添加涨跌停
        df['limit_up'] = df['pre_close'] * 1.10  # 涨停 10%
        df['limit_down'] = df['pre_close'] * 0.90  # 跌停 10%

        # 添加技术指标（简化）
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100

        # 1. 数据验证
        logger.info("=== 数据验证 ===")
        is_valid, validation_stats = self.validator.validate_dataframe(df)
        logger.info(self.validator.get_validation_report())

        if not is_valid:
            logger.warning("数据验证失败，但继续处理")

        # 2. 数据清洗
        logger.info("\n=== 数据清洗 ===")

        # 移除重复行
        df_clean = self.cleaner.remove_duplicates(df, subset=["trade_date"])

        # 填充空值
        df_clean = self.cleaner.fill_nulls(df_clean, method="ffill")

        # 移除异常值（价格）
        df_clean = self.cleaner.remove_outliers(df_clean, column="close", method="iqr")

        # 3. 数据质量分析
        logger.info("\n=== 数据质量分析 ===")
        quality_report = self.analyzer.generate_quality_report(df_clean)
        print(quality_report)

        # 4. 保存到缓存
        if use_cache:
            self.cache.set(params, df_clean, data_type="data", metadata={
                "source": "Tushare",
                "stock": ts_code,
                "start_date": start_date,
                "end_date": end_date,
                "validation_stats": validation_stats
            })

        return df_clean

    def load_stock_batch(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载股票数据

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            {股票代码: DataFrame} 字典
        """
        logger.info(f"批量加载 {len(ts_codes)} 只股票数据")

        results = {}
        for ts_code in ts_codes:
            try:
                df = self.load_stock_data(ts_code, start_date, end_date, use_cache)
                if not df.empty:
                    results[ts_code] = df
            except Exception as e:
                logger.error(f"加载 {ts_code} 数据失败: {e}")

        logger.info(f"成功加载 {len(results)}/{len(ts_codes)} 只股票")

        return results


class SyntheticDataGenerator:
    """合成数据生成器（用于演示）"""

    def __init__(
        self,
        num_samples: int = 10000,
        num_factors: int = 24,
        sequence_length: int = 60
    ):
        self.num_samples = num_samples
        self.num_factors = num_factors
        self.sequence_length = sequence_length

        np.random.seed(42)
        torch.manual_seed(42)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        生成合成数据

        Returns:
            features: [N, T, F] 因子特征
            market_sentiment: [N, S] 市场情绪
            targets: 目标字典
        """
        logger.info(f"Generating {self.num_samples} samples...")

        # 生成因子特征
        features = torch.randn(self.num_samples + self.sequence_length, self.num_factors)

        # 生成市场情绪（15维）
        market_sentiment = torch.randn(self.num_samples + self.sequence_length, 15)

        # 生成目标（基于特征生成真实目标）
        targets = self._generate_targets(features, market_sentiment)

        return features, market_sentiment, targets

    def _generate_targets(
        self,
        features: torch.Tensor,
        market_sentiment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """生成目标值"""
        num_samples = self.num_samples

        # 使用特征加权生成收益
        weights = torch.randn(self.num_factors)
        returns = []
        sharpe_ratios = []
        drawdowns = []

        for i in range(num_samples):
            # 使用滑动窗口的平均值
            feat_window = features[i:i+self.sequence_length]
            feat_mean = feat_window.mean(dim=0)

            # 计算目标
            target_return = (feat_mean * weights).sum() * 0.01  # 收益
            target_sharpe = (feat_mean @ weights) * 0.5 + 0.5  # 夏普（归一化）
            target_drawdown = -torch.abs((feat_mean @ weights) * 0.1)  # 回撤（负数）

            returns.append(target_return)
            sharpe_ratios.append(target_sharpe)
            drawdowns.append(target_drawdown)

        targets = {
            'return': torch.tensor(returns),
            'sharpe': torch.tensor(sharpe_ratios),
            'drawdown': torch.tensor(drawdowns)
        }

        return targets


class Trainer:
    """训练器（优化版）"""

    def __init__(
        self,
        config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        patience: int = 10,  # 早停耐心值
        min_delta: float = 1e-6,  # 早停最小改善
        save_dir: str = './checkpoints'  # 模型保存目录
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir

        logger.info(f"Using device: {self.device}")
        logger.info(f"Early stopping patience: {self.patience}")

        # 创建模型
        self.model = AlphaQuant(config).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器（ReduceLROnPlateau）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # 损失函数
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_mse = nn.MSELoss()

        # 梯度裁剪
        self.grad_clip = 5.0  # 梯度裁剪阈值

        # 早停状态
        self.best_val_loss = float('inf')
        self.counter = 0  # 没有改善的 epoch 计数
        self.early_stop = False

        # 训练监控
        self.training_metrics = TrainingMetrics(save_dir=self.save_dir)
        self.performance_monitor = PerformanceMonitor()
        self.early_stopping_monitor = EarlyStoppingMonitor(
            patience=self.patience,
            min_delta=self.min_delta
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0
        total_ce_loss = 0
        total_mse_loss = 0

        # 性能监控
        epoch_start_time = self.performance_monitor.start_timer()

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

            # 梯度裁剪（前向传播）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip
            )

            # 计算损失（改进版：加入风险建模）
            ce_loss = self.criterion_ce(
                output['logits'],
                torch.zeros(features.size(0), dtype=torch.long).to(self.device)
            )
            mse_loss = self.criterion_mse(output['value'], target_return)

            # 风险建模：夏普溢价计算
            # 夏普溢价 = 市场夏普收益率 - 5%
            # 夏普溢价 = (1.05 - market_return_mean) * 0.3  # 动态调整
            market_return_mean = batch['target_return'].mean(dim=0)
            sharpe_premium = (1.05 - market_return_mean) * 0.3

            # 风险调整系数
            risk_adjustment = 0.05 * torch.abs(sharpe_premium) * 0.1

            loss = ce_loss + 0.1 * mse_loss + risk_adjustment

            # 反向传播
            loss.backward()

            # 梯度裁剪（反向传播）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip
            )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 记录 epoch 时间
        epoch_time = self.performance_monitor.end_timer(epoch_start_time, "epoch")

        return {
            'loss': total_loss / len(dataloader),
            'ce_loss': total_ce_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader),
            'epoch_time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]["lr"]
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0
        total_ce_loss = 0
        total_mse_loss = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            sentiment = batch['market_sentiment'].to(self.device)
            target_return = batch['target_return'].to(self.device)

            # 前向传播
            output = self.model(features, sentiment)

            # 计算损失（验证时使用标准损失）
            ce_loss = self.criterion_ce(
                output['logits'],
                torch.zeros(features.size(0), dtype=torch.long).to(self.device)
            )
            mse_loss = self.criterion_mse(output['value'], target_return)

            loss = ce_loss + 0.1 * mse_loss

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()

        val_loss = total_loss / len(dataloader)

        # 早停检查
        if self.early_stopping_monitor.check(val_loss, epoch):
            self.early_stop = self.early_stopping_monitor.early_stop

        return {
            'loss': val_loss,
            'ce_loss': total_ce_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            'early_stop': self.early_stopping_monitor.early_stop
        }

    def save_checkpoint(self, save_dir: str, epoch: int, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.early_stopping_monitor.best_loss,
            'config': self.config,
            'early_stop': self.early_stopping_monitor.early_stop,
            'counter': self.early_stopping_monitor.counter,
            'patience': self.early_stopping_monitor.patience,
            'min_delta': self.early_stopping_monitor.min_delta,
            'learning_rate': self.optimizer.param_groups[0]["lr"]
        }

        path = os.path.join(save_dir, f'best_model_epoch{epoch}.pt')
        torch.save(checkpoint, path)

        logger.info(f"✅ Checkpoint saved to {path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = './checkpoints'
    ):
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Training Epoch {epoch}/{num_epochs}")

            # 检查早停
            if self.early_stopping_monitor.early_stop:
                logger.warning(f"⏸ Early stopping at epoch {epoch}")
                break

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader, epoch)

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # 更新训练指标
            self.training_metrics.update(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_metrics={'time': train_metrics['epoch_time']},
                val_metrics={'learning_rate': val_metrics['learning_rate']}
            )

            # 打印
            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {train_metrics['learning_rate']:.2e}"
            )

            # 保存最佳模型
            if not self.early_stopping_monitor.early_stop and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(save_dir, epoch, val_metrics['loss'])
                logger.info(f"🎉 Saved best model (val_loss: {val_metrics['loss']:.4f})")

        # 保存训练指标
        self.training_metrics.save_metrics('training_metrics.csv')
        self.training_metrics.save_best_metrics('best_metrics.json')
        self.training_metrics.plot_training_curves(f'{save_dir}/training_curves.png')

        # 生成训练报告
        report = self.training_metrics.generate_report()
        logger.info(f"\n{report}")

        logger.info("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train AlphaQuant Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use-real-data', action='store_true', help='Use real data from Tushare')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')

    args = parser.parse_args()

    # 模型配置
    config = ModelConfig(
        d_model=128,
        nhead=8,
        num_layers=4,
        max_formula_len=64
    )

    # 生成合成数据
    logger.info("Generating synthetic data...")
    generator = SyntheticDataGenerator(
        num_samples=10000,
        num_factors=config.num_basic_factors + config.num_advanced_factors,
        sequence_length=60
    )

    features, sentiment, targets = generator.generate()

    # 创建数据集
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    val_features = features[train_size:]
    train_sentiment = sentiment[:train_size]
    val_sentiment = sentiment[train_size:]

    train_targets = {
        'return': targets['return'][:train_size],
        'sharpe': targets['sharpe'][:train_size],
        'drawdown': targets['drawdown'][:train_size]
    }

    val_targets = {
        'return': targets['return'][train_size:],
        'sharpe': targets['sharpe'][train_size:],
        'drawdown': targets['drawdown'][train_size:]
    }

    train_dataset = FactorDataset(
        train_features,
        train_sentiment,
        train_targets,
        sequence_length=60
    )

    val_dataset = FactorDataset(
        val_features,
        val_sentiment,
        val_targets,
        sequence_length=60
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

    # 训练
    logger.info("Starting training...")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Device: {args.device}")

    trainer = Trainer(
        config,
        learning_rate=args.lr,
        device=args.device,
        patience=args.patience,
        save_dir=args.save_dir
    )

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
