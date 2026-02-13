"""
训练指标单元测试

测试 TrainingMetrics 类的功能
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquant.metrics import TrainingMetrics


class TestTrainingMetrics:
    """测试训练指标收集器"""

    def test_initialization(self):
        """测试指标收集器初始化"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 检查初始状态
        assert metrics.save_dir.exists()
        assert metrics.metrics_history == []
        assert metrics.best_metrics["best_val_loss"] == float("inf")
        assert metrics.best_metrics["best_epoch"] == 0
        assert metrics.best_metrics["best_train_loss"] == float("inf")
        assert metrics.best_metrics["best_sharpe"] == float("-inf")
        assert metrics.best_metrics["best_epoch_sharpe"] == 0

    def test_update_basic_metrics(self):
        """测试基础指标更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新指标
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_metrics={"accuracy": 0.8},
            val_metrics={"accuracy": 0.75}
        )

        # 检查更新
        assert len(metrics.metrics_history) == 1
        assert metrics.metrics_history[0]["epoch"] == 1
        assert metrics.metrics_history[0]["train_loss"] == 0.5
        assert metrics.metrics_history[0]["val_loss"] == 0.6
        assert metrics.metrics_history[0]["train_accuracy"] == 0.8
        assert metrics.metrics_history[0]["val_accuracy"] == 0.75

    def test_update_without_metrics(self):
        """测试不包含训练/验证指标的更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        assert len(metrics.metrics_history) == 1
        assert "train_accuracy" not in metrics.metrics_history[0]

    def test_best_val_loss_update(self):
        """测试最佳验证损失更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 第一次更新
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        assert metrics.best_metrics["best_val_loss"] == 0.6
        assert metrics.best_metrics["best_epoch"] == 1

        # 第二次更新（更好）
        metrics.update(
            epoch=2,
            train_loss=0.4,
            val_loss=0.5
        )

        assert metrics.best_metrics["best_val_loss"] == 0.5
        assert metrics.best_metrics["best_epoch"] == 2

    def test_best_train_loss_update(self):
        """测试最佳训练损失更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        assert metrics.best_metrics["best_train_loss"] == 0.5
        assert metrics.best_metrics["best_epoch_train"] == 1

        # 更新训练损失但验证损失更差
        metrics.update(
            epoch=2,
            train_loss=0.4,
            val_loss=0.7
        )

        assert metrics.best_metrics["best_train_loss"] == 0.4
        assert metrics.best_metrics["best_epoch_train"] == 2
        # 最佳验证损失应该保持不变
        assert metrics.best_metrics["best_val_loss"] == 0.6

    def test_sharpe_metric_update(self):
        """测试夏普比率更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新包含夏普比率的指标
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            val_metrics={"sharpe": 1.5}
        )

        assert metrics.best_metrics["best_sharpe"] == 1.5
        assert metrics.best_metrics["best_epoch_sharpe"] == 1

    def test_multiple_updates(self):
        """测试多次指标更新"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 多次更新
        for i in range(1, 6):
            metrics.update(
                epoch=i,
                train_loss=1.0 / i,
                val_loss=1.0 / i + 0.1,
                train_metrics={"accuracy": 0.8 + i * 0.02},
                val_metrics={"accuracy": 0.75 + i * 0.02}
            )

        # 检查历史记录
        assert len(metrics.metrics_history) == 5

        # 检查最佳指标
        best_val_loss = metrics.best_metrics["best_val_loss"]
        best_epoch = metrics.best_metrics["best_epoch"]

        # 验证（使用 approx 避免浮点精度问题）
        assert pytest.approx(0.3, abs=1e-6) == best_val_loss
        assert 5 == best_epoch

        # 检查最后更新
        last_metrics = metrics.metrics_history[-1]
        assert pytest.approx(0.2, abs=1e-6) == last_metrics["train_loss"]
        assert pytest.approx(0.3, abs=1e-6) == last_metrics["val_loss"]

    def test_loss_improvement_calculation(self):
        """测试损失改进计算"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 第一次更新
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        # 检查损失改进（应该没有，因为是第一次）
        assert "loss_improvement" in metrics.metrics_history[0]

        # 第二次更新（有改进）
        metrics.update(
            epoch=2,
            train_loss=0.4,
            val_loss=0.5
        )

        # 检查损失改进
        last_metrics = metrics.metrics_history[-1]
        improvement = last_metrics["loss_improvement"]
        expected_improvement = (0.5 - 0.4) / 0.5  # 20% 改进

        assert abs(improvement - expected_improvement) < 0.001

    def test_additional_metrics(self):
        """测试额外指标"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新包含额外指标
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_metrics={"accuracy": 0.8},
            val_metrics={"accuracy": 0.75},
            additional_metrics={
                "learning_rate": 0.001,
                "batch_size": 32,
                "model_size": 1000000
            }
        )

        # 检查额外指标
        last_metrics = metrics.metrics_history[-1]
        assert "learning_rate" in last_metrics
        assert "batch_size" in last_metrics
        assert "model_size" in last_metrics

        assert last_metrics["learning_rate"] == 0.001
        assert last_metrics["batch_size"] == 32
        assert last_metrics["model_size"] == 1000000

    def test_timestamp_recording(self):
        """测试时间戳记录"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        # 检查时间戳
        assert "timestamp" in metrics.metrics_history[0]
        assert isinstance(metrics.metrics_history[0]["timestamp"], datetime)

    def test_clear_metrics_history(self):
        """测试清空指标历史"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 添加一些指标
        metrics.update(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6
        )

        assert len(metrics.metrics_history) == 1

        # 清空历史
        metrics.metrics_history.clear()

        assert len(metrics.metrics_history) == 0
        # 最佳指标应该保持
        assert metrics.best_metrics["best_val_loss"] == 0.6

    def test_save_dir_creation(self):
        """测试保存目录创建"""
        custom_save_dir = "./custom_test_metrics"
        metrics = TrainingMetrics(save_dir=custom_save_dir)

        from pathlib import Path
        save_dir = Path(custom_save_dir)

        assert save_dir.exists()
        assert save_dir.is_dir()

    def test_metrics_with_nan(self):
        """测试包含 NaN 值的指标"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新包含 NaN 的指标
        metrics.update(
            epoch=1,
            train_loss=float('nan'),
            val_loss=0.6,
            train_metrics={"accuracy": float('nan')},
            val_metrics={"accuracy": 0.75}
        )

        # 检查 NaN 值被保留
        last_metrics = metrics.metrics_history[-1]
        assert isinstance(last_metrics["train_loss"], float)
        assert isinstance(last_metrics["train_accuracy"], float)

    def test_worst_case_scenario(self):
        """测试最坏场景（非常大的损失）"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 更新为非常大的损失
        metrics.update(
            epoch=1,
            train_loss=1000.0,
            val_loss=1000.0
        )

        # 检查最佳指标
        assert metrics.best_metrics["best_val_loss"] == 1000.0
        assert metrics.best_metrics["best_train_loss"] == 1000.0

        # 更新为更小的损失
        metrics.update(
            epoch=2,
            train_loss=0.1,
            val_loss=0.2
        )

        # 检查最佳指标更新
        assert pytest.approx(0.2, abs=1e-6) == metrics.best_metrics["best_val_loss"]
        assert 2 == metrics.best_metrics["best_epoch"]

    def test_get_best_metrics(self):
        """测试获取最佳指标"""
        metrics = TrainingMetrics(save_dir="./test_metrics")

        # 多次更新
        for i in range(1, 4):
            metrics.update(
                epoch=i,
                train_loss=1.0 / i,
                val_loss=1.0 / i + 0.1,
                val_metrics={"sharpe": i * 0.5}
            )

        # 获取最佳指标（注意：approx 需要左侧为预期值，右侧为实际值）
        best_val_loss = metrics.best_metrics["best_val_loss"]
        best_epoch = metrics.best_metrics["best_epoch"]
        best_sharpe = metrics.best_metrics["best_sharpe"]
        best_epoch_sharpe = metrics.best_metrics["best_epoch_sharpe"]

        # 验证
        assert pytest.approx(0.43333333333333335, abs=1e-6) == best_val_loss  # 第 3 轮的 val_loss
        assert pytest.approx(3, abs=0.1) == best_epoch
        assert pytest.approx(1.5, abs=1e-6) == best_sharpe
        assert pytest.approx(3, abs=0.1) == best_epoch_sharpe
