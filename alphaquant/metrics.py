"""
训练监控模块
跟踪训练过程中的关键指标（收益、夏普、波动率等）
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path


class TrainingMetrics:
    """训练指标收集器"""

    def __init__(self, save_dir: str = "./metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 指标历史
        self.metrics_history: List[Dict[str, Any]] = []

        # 最佳指标
        self.best_metrics = {
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "best_train_loss": float("inf"),
            "best_epoch_train": 0,
            "best_sharpe": float("-inf"),
            "best_epoch_sharpe": 0
        }

        # 当前指标
        self.current_metrics: Optional[Dict[str, Any]] = None

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        更新训练指标

        Args:
            epoch: 当前 epoch
            train_loss: 训练损失
            val_loss: 验证损失
            train_metrics: 训练指标（可选）
            val_metrics: 验证指标（可选）
            additional_metrics: 额外的指标（可选）
        """
        timestamp = datetime.now()

        # 基础指标
        metrics = {
            "timestamp": timestamp,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_improvement": self._calculate_improvement(train_loss, self._get_last_metric("train_loss"))
        }

        # 添加训练指标
        if train_metrics:
            for key, value in train_metrics.items():
                metrics[f"train_{key}"] = value

        # 添加验证指标
        if val_metrics:
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

        # 添加额外指标
        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[key] = value

        # 更新最佳指标
        if val_loss < self.best_metrics["best_val_loss"]:
            self.best_metrics["best_val_loss"] = val_loss
            self.best_metrics["best_epoch"] = epoch

        if train_loss < self.best_metrics["best_train_loss"]:
            self.best_metrics["best_train_loss"] = train_loss
            self.best_metrics["best_epoch_train"] = epoch

        if val_metrics and "sharpe" in val_metrics:
            if val_metrics["sharpe"] > self.best_metrics["best_sharpe"]:
                self.best_metrics["best_sharpe"] = val_metrics["sharpe"]
                self.best_metrics["best_epoch_sharpe"] = epoch

        # 添加到历史
        self.metrics_history.append(metrics)

    def _get_last_metric(self, metric_name: str) -> float:
        """获取最后一次的指标值"""
        if not self.metrics_history:
            return float("inf")

        for i in range(len(self.metrics_history) - 1, -1, -1):
            if metric_name in self.metrics_history[i]:
                return self.metrics_history[i][metric_name]

        return float("inf")

    def _calculate_improvement(self, current: float, previous: float) -> float:
        """计算损失改进百分比"""
        if previous == float("inf"):
            return 0.0

        if previous == 0:
            return 0.0

        return (previous - current) / previous

    def get_best_metrics(self) -> Dict[str, Any]:
        """获取最佳指标"""
        return self.best_metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]
        best_metrics = self.get_best_metrics()

        return {
            "current_metrics": latest_metrics,
            "best_metrics": best_metrics,
            "total_epochs": len(self.metrics_history)
        }

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """获取指标历史"""
        return self.metrics_history

    def clear_history(self) -> None:
        """清空指标历史"""
        self.metrics_history.clear()

    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """
        保存指标到文件

        Args:
            filepath: 文件路径（可选）
        """
        if filepath is None:
            filepath = self.save_dir / "training_metrics.csv"

        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)

        logger.info(f"Metrics saved to {filepath}")
