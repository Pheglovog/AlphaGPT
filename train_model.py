#!/usr/bin/env python3
"""
AlphaQuant 模型训练脚本
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
    """训练器"""

    def __init__(
        self,
        config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda'
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # 损失函数
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_mse = nn.MSELoss()

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

            # 计算损失（简化版）
            # 实际应该根据生成的公式计算回测奖励
            ce_loss = self.criterion_ce(
                output['logits'],
                torch.zeros(features.size(0), dtype=torch.long).to(self.device)
            )
            mse_loss = self.criterion_mse(output['value'], target_return)

            loss = ce_loss + 0.1 * mse_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}'
            })

        return {
            'loss': total_loss / len(dataloader),
            'ce_loss': total_ce_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader)
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0
        total_ce_loss = 0
        total_mse_loss = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            sentiment = batch['market_sentiment'].to(self.device)
            target_return = batch['target_return'].to(self.device)

            output = self.model(features, sentiment)

            ce_loss = self.criterion_ce(
                output['logits'],
                torch.zeros(features.size(0), dtype=torch.long).to(self.device)
            )
            mse_loss = self.criterion_mse(output['value'], target_return)
            loss = ce_loss + 0.1 * mse_loss

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()

        return {
            'loss': total_loss / len(dataloader),
            'ce_loss': total_ce_loss / len(dataloader),
            'mse_loss': total_mse_loss / len(dataloader)
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

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Training Epoch {epoch}/{num_epochs}")

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 打印
            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(save_dir, epoch, val_metrics['loss'])
                logger.info(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")

    def save_checkpoint(self, save_dir: str, epoch: int, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        path = os.path.join(save_dir, f'best_model_epoch{epoch}.pt')
        torch.save(checkpoint, path)

        logger.info(f"Checkpoint saved to {path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train AlphaQuant Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')

    args = parser.parse_args()

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

    trainer = Trainer(
        config,
        learning_rate=args.lr,
        device=args.device
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
