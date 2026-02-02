"""
训练监控模块
跟踪训练过程中的关键指标（收益、夏普、波动率等）
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.current_metrics = metrics

        # 记录日志
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Best Val Loss: {self.best_metrics['best_val_loss']:.4f} (Epoch {self.best_metrics['best_epoch']})"
        )

        return metrics

    def _calculate_improvement(self, current: float, previous: Optional[float]) -> float:
        """计算改进幅度"""
        if previous is None or previous == 0:
            return 0.0
        return (previous - current) / abs(previous) * 100

    def _get_last_metric(self, metric_name: str) -> Optional[float]:
        """获取最后一个指标值"""
        if not self.metrics_history:
            return None

        for metrics in reversed(self.metrics_history):
            if metric_name in metrics:
                return float(metrics[metric_name])

        return None

    def get_dataframe(self) -> pd.DataFrame:
        """
        将指标历史转换为 DataFrame

        Returns:
            指标 DataFrame
        """
        df = pd.DataFrame(self.metrics_history)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('epoch')
        return df

    def get_best_metrics(self) -> Dict[str, Any]:
        """
        获取最佳指标

        Returns:
            最佳指标字典
        """
        df = self.get_dataframe()

        if df.empty:
            return self.best_metrics

        best_metrics = {
            "best_val_loss": df['val_loss'].min(),
            "best_epoch_val_loss": df['val_loss'].idxmin(),
            "best_train_loss": df['train_loss'].min(),
            "best_epoch_train_loss": df['train_loss'].idxmin()
        }

        # 找出最佳夏普
        if 'val_sharpe' in df.columns:
            best_sharpe_idx = df['val_sharpe'].idxmax()
            best_metrics['best_sharpe'] = df.loc[best_sharpe_idx, 'val_sharpe']
            best_metrics['best_epoch_sharpe'] = best_sharpe_idx

        return best_metrics

    def calculate_improvement_metrics(self) -> Dict[str, float]:
        """
        计算改进指标

        Returns:
            改进指标字典
        """
        if len(self.metrics_history) < 2:
            return {}

        df = self.get_dataframe()

        metrics = {}

        # 损失改进
        first_loss = df['val_loss'].iloc[0]
        last_loss = df['val_loss'].iloc[-1]
        metrics['loss_improvement'] = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0

        # 平均损失改进（最近 10 epochs）
        if len(df) >= 10:
            recent_losses = df['val_loss'].tail(10).values
            metrics['recent_loss_trend'] = (recent_losses[-1] - recent_losses[0]) / recent_losses[0] * 100

        # 损失收敛速度
        if len(df) >= 10:
            early_loss = df['val_loss'].head(10).mean()
            late_loss = df['val_loss'].tail(10).mean()
            metrics['convergence_speed'] = (early_loss - late_loss) / early_loss * 100 if early_loss > 0 else 0

        return metrics

    def save_metrics(self, filename: str = "metrics.csv"):
        """
        保存指标到 CSV 文件

        Args:
            filename: 文件名
        """
        df = self.get_dataframe()

        if not df.empty:
            file_path = self.save_dir / filename
            df.to_csv(file_path, index=False)
            logger.info(f"指标已保存到：{file_path}")
        else:
            logger.warning("没有指标数据可保存")

    def save_best_metrics(self, filename: str = "best_metrics.json"):
        """
        保存最佳指标到 JSON 文件

        Args:
            filename: 文件名
        """
        import json

        file_path = self.save_dir / filename

        best_metrics = self.get_best_metrics()

        # 转换 numpy 类型为 Python 原生类型
        def convert_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        best_metrics_converted = convert_types(best_metrics)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(best_metrics_converted, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"最佳指标已保存到：{file_path}")

    def generate_report(self) -> str:
        """
        生成训练报告

        Returns:
            训练报告字符串
        """
        df = self.get_dataframe()

        if df.empty:
            return "没有训练指标数据"

        best_metrics = self.get_best_metrics()
        improvement = self.calculate_improvement_metrics()

        report = f"""
        === 训练报告 ===

        基本信息：
        - 总训练 Epoch 数：{len(df)}
        - 开始时间：{df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}
        - 结束时间：{df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}
        - 训练时长：{(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} 小时

        损失指标：
        - 初始 Val Loss：{df['val_loss'].iloc[0]:.4f}
        - 最终 Val Loss：{df['val_loss'].iloc[-1]:.4f}
        - 最佳 Val Loss：{best_metrics['best_val_loss']:.4f} (Epoch {best_metrics['best_epoch_val_loss']})
        - 最终 Train Loss：{df['train_loss'].iloc[-1]:.4f}
        - 最佳 Train Loss：{best_metrics['best_train_loss']:.4f} (Epoch {best_metrics['best_epoch_train_loss']})
        - 损失改进：{improvement.get('loss_improvement', 0):.2f}%

        夏普比率：
        - 最终 Val 夏普：{best_metrics.get('best_sharpe', 'N/A'):.2f} (Epoch {best_metrics.get('best_epoch_sharpe', 'N/A')})

        收敛情况：
        - 损失收敛速度：{improvement.get('convergence_speed', 0):.2f}%
        - 最近损失趋势：{improvement.get('recent_loss_trend', 0):.2f}%
        """

        return report

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        绘制训练曲线

        Args:
            save_path: 保存路径（可选）
        """
        df = self.get_dataframe()

        if df.empty:
            logger.warning("没有数据可绘制")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练指标', fontsize=16, fontweight='bold')

        # 1. 损失曲线
        axes[0, 0].plot(df.index, df['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(df.index, df['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 损失改进
        if len(df) >= 2:
            loss_change = df['val_loss'].diff()
            axes[0, 1].bar(df.index[1:], loss_change[1:])
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss Change')
            axes[0, 1].set_title('Validation Loss Change')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 夏普比率（如果有）
        if 'val_sharpe' in df.columns:
            axes[1, 0].plot(df.index, df['val_sharpe'], label='Val Sharpe', linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].set_title('Sharpe Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 学习率（如果有）
        if 'learning_rate' in df.columns:
            axes[1, 1].plot(df.index, df['learning_rate'], marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')

        plt.tight_layout()

        # 保存图表
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线已保存到：{save_path}")
        else:
            plt.show()

        plt.close()


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.training_times: List[float] = []
        self.epoch_times: List[float] = []
        self.data_load_times: List[float] = []

    def start_timer(self) -> float:
        """启动计时器"""
        return time.time()

    def end_timer(self, start_time: float, timer_type: str = "epoch") -> float:
        """
        结束计时器并记录

        Args:
            start_time: 开始时间
            timer_type: 计时器类型（epoch, data_load, training）

        Returns:
            耗时（秒）
        """
        elapsed = time.time() - start_time

        if timer_type == "epoch":
            self.epoch_times.append(elapsed)
        elif timer_type == "data_load":
            self.data_load_times.append(elapsed)
        elif timer_type == "training":
            self.training_times.append(elapsed)

        return elapsed

    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计

        Returns:
            性能统计字典
        """
        stats = {}

        if self.epoch_times:
            stats["avg_epoch_time"] = np.mean(self.epoch_times)
            stats["min_epoch_time"] = np.min(self.epoch_times)
            stats["max_epoch_time"] = np.max(self.epoch_times)

        if self.data_load_times:
            stats["avg_data_load_time"] = np.mean(self.data_load_times)

        if self.training_times:
            stats["total_training_time"] = np.sum(self.training_times)
            stats["avg_epoch_time"] = np.mean(self.training_times)

        return stats


class EarlyStoppingMonitor:
    """早停监控器"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        min_improvement_rate: float = 0.01,  # 最小改进率（百分比）
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_improvement_rate = min_improvement_rate
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

        self.best_weights = None

    def check(self, val_loss: float, epoch: int) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 当前验证损失
            epoch: 当前 epoch

        Returns:
            是否应该早停
        """
        improvement = (self.best_loss - val_loss) / abs(self.best_loss) * 100 if self.best_loss != 0 else 0

        # 检查是否有足够改进
        if improvement < self.min_improvement_rate and abs(self.best_loss - val_loss) < self.min_delta:
            self.counter += 1
            logger.debug(f"没有显著改进（{improvement:.2f}% < {self.min_improvement_rate*100:.2f}%），等待计数：{self.counter}/{self.patience}")
        else:
            self.counter = 0
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                logger.info(f"新的最佳模型：Val Loss {val_loss:.4f} (Epoch {epoch})")

        # 检查是否应该早停
        if self.counter >= self.patience:
            self.early_stop = True
            logger.warning(f"早停触发！{self.patience} 个 epochs 没有显著改进")
            logger.info(f"最佳模型在 Epoch {self.best_epoch}，Val Loss {self.best_loss:.4f}")

            return True

        return False


# 使用示例
def example_usage():
    """使用示例"""
    # 1. 创建训练指标收集器
    metrics = TrainingMetrics()

    # 2. 模拟训练过程
    print("=== 模拟训练过程 ===")
    for epoch in range(1, 11):
        train_loss = 0.5 * (0.9 ** epoch) + 0.05 * np.random.randn()
        val_loss = 0.5 * (0.9 ** epoch) + 0.08 * np.random.randn()

        # 添加额外的指标
        train_metrics = {
            "accuracy": 0.7 + 0.02 * epoch,
            "precision": 0.65 + 0.03 * epoch
        }

        val_metrics = {
            "accuracy": 0.68 + 0.02 * epoch,
            "sharpe": 1.0 + 0.1 * epoch
        }

        additional_metrics = {
            "learning_rate": 0.001 * (0.95 ** epoch),
            "model_size_mb": 12.5
        }

        # 更新指标
        metrics.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            additional_metrics=additional_metrics
        )

    # 3. 生成报告
    print("\n=== 训练报告 ===")
    report = metrics.generate_report()
    print(report)

    # 4. 绘制训练曲线
    print("\n=== 绘制训练曲线 ===")
    metrics.plot_training_curves("training_curves.png")

    # 5. 保存指标
    print("\n=== 保存指标 ===")
    metrics.save_metrics("training_metrics.csv")
    metrics.save_best_metrics("best_metrics.json")


if __name__ == "__main__":
    example_usage()
