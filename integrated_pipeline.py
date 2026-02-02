"""
AlphaGPT 训练集成示例
展示如何使用数据验证、缓存和监控模块
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphaquant.data_validation import DataValidator, DataCleaner, DataQualityAnalyzer
from alphaquant.data_cache import DataCache
from alphaquant.metrics import TrainingMetrics, PerformanceMonitor, EarlyStoppingMonitor
from train_model_optimized import SyntheticDataGenerator, Trainer, ModelConfig
from alphaquant.model.alpha_quant import AlphaQuant
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class AlphaGPTPipeline:
    """AlphaGPT 训练流水线（集成版）"""

    def __init__(self):
        # 配置
        self.config = ModelConfig(
            d_model=128,
            nhead=8,
            num_layers=4,
            max_formula_len=64
        )

        # 数据验证器
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.analyzer = DataQualityAnalyzer()

        # 数据缓存
        self.cache = DataCache()

        # 训练监控
        self.metrics = TrainingMetrics()
        self.performance = PerformanceMonitor()
        self.early_stopping = EarlyStoppingMonitor(patience=10, min_delta=1e-6)

    async def run_full_pipeline(
        self,
        num_samples: int = 10000,
        num_epochs: int = 50,
        batch_size: int = 32
    ):
        """
        运行完整的训练流水线

        Args:
            num_samples: 样本数量
            num_epochs: 训练轮数
            batch_size: 批次大小
        """
        logger.info("=" * 60)
        logger.info("AlphaGPT 训练流水线启动")
        logger.info("=" * 60)

        # ====== 阶段 1: 数据生成 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 1: 数据生成")
        logger.info("=" * 60)

        start_time = self.performance.start_timer()

        generator = SyntheticDataGenerator(
            num_samples=num_samples,
            num_factors=self.config.num_basic_factors + self.config.num_advanced_factors,
            sequence_length=60
        )

        features, sentiment, targets = generator.generate()
        logger.info(f"✅ 数据生成完成：{num_samples} 样本，{features.shape[1]} 因子，{features.shape[2]} 时间步")

        # ====== 阶段 2: 数据验证 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 2: 数据验证")
        logger.info("=" * 60)

        # 将 tensor 转换为 DataFrame（用于验证）
        df_samples = []
        for i in range(min(1000, num_samples)):
            sample = {
                'return': targets['return'][i].item(),
                'sharpe': targets['sharpe'][i].item(),
                'drawdown': targets['drawdown'][i].item(),
                'volatility': features[i, :, 0].std().item()  # 简化的波动率
            }
            df_samples.append(sample)

        import pandas as pd
        df = pd.DataFrame(df_samples)

        # 验证数据
        is_valid, stats = self.validator.validate_dataframe(df)
        logger.info(self.validator.get_validation_report())

        # ====== 阶段 3: 数据清洗 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 3: 数据清洗")
        logger.info("=" * 60)

        # 移除重复行
        df_clean = self.cleaner.remove_duplicates(df)

        # 填充空值
        df_clean = self.cleaner.fill_nulls(df_clean, method="ffill")

        # 移除异常值
        df_clean = self.cleaner.remove_outliers(df_clean, column="return", method="iqr")

        logger.info(self.cleaner.get_cleaning_report())

        # ====== 阶段 4: 数据质量分析 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 4: 数据质量分析")
        logger.info("=" * 60)

        quality_report = self.analyzer.generate_quality_report(df_clean)
        print(quality_report)

        # 数据加载时间
        data_load_time = self.performance.end_timer(start_time, "data_load")

        # ====== 阶段 5: 创建数据集 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 5: 创建数据集")
        logger.info("=" * 60)

        from torch.utils.data import Dataset, DataLoader

        class OptimizedFactorDataset(Dataset):
            def __init__(self, features, sentiment, targets, seq_len=60):
                self.features = features
                self.sentiment = sentiment
                self.targets = targets
                self.seq_len = seq_len

            def __len__(self):
                return self.features.shape[0] - self.seq_len

            def __getitem__(self, idx):
                feat_seq = self.features[idx:idx+self.seq_len]
                sent = self.sentiment[idx + self.seq_len - 1]

                target_return = self.targets['return'][idx + self.seq_len]
                target_sharpe = self.targets['sharpe'][idx + self.seq_len]
                target_drawdown = self.targets['drawdown'][idx + self.seq_len]

                return {
                    'features': feat_seq,
                    'market_sentiment': sent,
                    'target_return': target_return,
                    'target_sharpe': target_sharpe,
                    'target_drawdown': target_drawdown
                }

        # 分割数据集
        train_size = int(0.8 * num_samples)

        train_dataset = OptimizedFactorDataset(
            features[:train_size],
            sentiment[:train_size],
            {
                'return': targets['return'][:train_size],
                'sharpe': targets['sharpe'][:train_size],
                'drawdown': targets['drawdown'][:train_size]
            },
            60
        )

        val_dataset = OptimizedFactorDataset(
            features[train_size:],
            sentiment[train_size:],
            {
                'return': targets['return'][train_size:],
                'sharpe': targets['sharpe'][train_size:],
                'drawdown': targets['drawdown'][train_size:]
            },
            60
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        logger.info(f"✅ 数据集创建完成：{len(train_dataset)} 训练样本，{len(val_dataset)} 验证样本")

        # ====== 阶段 6: 训练 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 6: 模型训练")
        logger.info("=" * 60)

        training_start_time = self.performance.start_timer()

        # 创建训练器
        trainer = Trainer(
            self.config,
            learning_rate=1e-4,
            weight_decay=1e-5,
            device='cuda',
            patience=10,
            min_delta=1e-6,
            save_dir='./checkpoints'
        )

        # 训练
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            save_dir='./checkpoints'
        )

        training_time = self.performance.end_timer(training_start_time, "training")

        # ====== 阶段 7: 生成报告 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 7: 生成报告")
        logger.info("=" * 60)

        # 训练指标
        metrics_report = self.metrics.generate_report()
        print(metrics_report)

        # 性能统计
        perf_stats = self.performance.get_performance_stats()
        logger.info(f"\n性能统计：")
        logger.info(f"  总训练时间：{perf_stats.get('total_training_time', 0):.1f} 秒")
        logger.info(f"  平均 Epoch 时间：{perf_stats.get('avg_epoch_time', 0):.2f} 秒")
        logger.info(f"  总请求数：{self.metrics.metrics_history[-1]['timestamp']} - {self.metrics.metrics_history[0]['timestamp']}")

        # ====== 阶段 8: 缓存统计 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 8: 缓存统计")
        logger.info("=" * 60)

        cache_stats = self.cache.get_cache_stats()
        cache_report = self.cache.get_cache_report()
        print(cache_report)

        # ====== 阶段 9: 清理过期缓存 ======
        logger.info("\n" + "=" * 60)
        logger.info("阶段 9: 清理过期缓存")
        logger.info("=" * 60)

        removed = self.cache.clean_cache(older_than_hours=24)
        logger.info(f"✅ 清理了 {removed} 个过期缓存文件")

        logger.info("\n" + "=" * 60)
        logger.info("✅ 完整流水线执行完成！")
        logger.info("=" * 60)

        # 总结
        logger.info("\n📊 流水线总结：")
        logger.info(f"  1. 数据生成：{num_samples} 样本")
        logger.info(f"  2. 数据验证：{stats['valid_rows']}/{stats['total_rows']} 有效")
        logger.info(f"  3. 数据清洗：{self.cleaner.cleaning_stats['rows_cleaned']} 行处理")
        logger.info(f"  4. 数据质量：完整度 {stats.get('total_cells', 0) > 0 and stats['completeness']:.1f}%")
        logger.info(f"  5. 数据加载时间：{data_load_time:.2f} 秒")
        logger.info(f"  6. 训练轮数：{num_epochs}")
        logger.info(f"  7. 最佳 Val Loss：{self.metrics.best_metrics['best_val_loss']:.4f} (Epoch {self.metrics.best_metrics['best_epoch']})")
        logger.info(f"  8. 缓存命中率：{cache_stats['cache_hit_rate']:.2f}%")
        logger.info(f"  9. 清理缓存：{removed} 个文件")

        return {
            'data_generated': num_samples,
            'data_valid': stats['valid_rows'],
            'data_cleaned': self.cleaner.cleaning_stats['rows_cleaned'],
            'training_epochs': num_epochs,
            'best_val_loss': self.metrics.best_metrics['best_val_loss'],
            'best_epoch': self.metrics.best_metrics['best_epoch'],
            'cache_hit_rate': cache_stats['cache_hit_rate']
        }


async def main():
    """主函数"""
    # 创建流水线
    pipeline = AlphaGPTPipeline()

    # 运行完整流水线
    results = await pipeline.run_full_pipeline(
        num_samples=10000,
        num_epochs=20,  # 使用较少的 epochs 进行演示
        batch_size=32
    )

    # 打印最终结果
    logger.info("\n" + "=" * 60)
    logger.info("🎉 AlphaGPT 训练流水线演示完成！")
    logger.info("=" * 60)
    logger.info("\n主要成果：")
    logger.info(f"  ✅ 数据验证和清洗流程")
    logger.info(f"  ✅ 数据缓存系统")
    logger.info(f"  ✅ 训练监控和早停")
    logger.info(f"  ✅ 性能监控和统计")
    logger.info(f"  ✅ 完整的流水线集成")
    logger.info("\n下一步：")
    logger.info("  1. 从 Tushare 加载真实市场数据")
    logger.info("  2. 实现数据验证和清洗的自动化")
    logger.info("  3. 优化模型超参数")
    logger.info("  4. 进行更长时间的训练")
    logger.info("  5. 部署到生产环境")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
