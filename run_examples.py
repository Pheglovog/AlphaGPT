#!/usr/bin/env python3
"""
AlphaQuant 运行示例
演示完整的量化交易流程
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from loguru import logger
from alphaquant.data_providers.tushare import TushareProProvider
from alphaquant.factors.china_factors import ChinaFactorEngine
from alphaquant.model.alpha_quant import AlphaQuant, ModelConfig
from alphaquant.backtest.backtester import BacktestEngine, Order, OrderSide, OrderType
from datetime import datetime


def example_factor_computation():
    """示例：因子计算"""
    print("\n" + "="*60)
    print("📊 示例 1: 因子计算")
    print("="*60)

    # 创建因子引擎
    engine = ChinaFactorEngine()

    # 模拟数据 [batch=5, time=60]
    batch_size = 5
    seq_len = 60

    raw_data = {
        'close': torch.rand(batch_size, seq_len) * 10 + 100,
        'open': torch.rand(batch_size, seq_len) * 10 + 100,
        'high': torch.rand(batch_size, seq_len) * 10 + 105,
        'low': torch.rand(batch_size, seq_len) * 10 + 95,
        'volume': torch.rand(batch_size, seq_len) * 1000000,
    }

    # 计算基础因子
    basic_factors = engine.compute_basic_factors(raw_data)
    print(f"✅ 基础因子: {basic_factors.shape}")  # [5, 6, 60]

    # 计算高级因子
    advanced_factors = engine.compute_advanced_factors(raw_data)
    print(f"✅ 高级因子: {advanced_factors.shape}")  # [5, 18, 60]

    return basic_factors, advanced_factors


def example_model_inference():
    """示例：模型推理"""
    print("\n" + "="*60)
    print("🤖 示例 2: 模型推理与因子公式生成")
    print("="*60)

    # 创建模型
    config = ModelConfig(d_model=64, nhead=4, num_layers=2)
    model = AlphaQuant(config)

    # 模拟输入
    batch_size = 2
    num_factors = 24
    time_steps = 60

    factor_features = torch.randn(batch_size, num_factors, time_steps)
    market_sentiment = torch.randn(batch_size, 15)

    # 前向传播
    output = model(factor_features, market_sentiment)

    print(f"✅ Logits shape: {output['logits'].shape}")      # [2, vocab_size]
    print(f"✅ Value shape: {output['value'].shape}")        # [2, 1]
    print(f"✅ Task probs: {output['task_probs'].shape}")    # [2, 3]

    # 生成因子公式
    formulas = model.generate_formula(
        factor_features,
        market_sentiment,
        max_length=15,
        temperature=1.0
    )

    print(f"\n✅ 生成的因子公式:")
    for i, formula in enumerate(formulas):
        print(f"   样本 {i}: {' '.join(formula[:10])}...")

    return model


def example_backtest():
    """示例：回测"""
    print("\n" + "="*60)
    print("📈 示例 3: 策略回测")
    print("="*60)

    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000.0)

    # 生成模拟数据
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    n = len(dates)

    np.random.seed(42)

    # 贵州茅台模拟数据
    data = pd.DataFrame({
        'open': 1700 + np.cumsum(np.random.randn(n) * 20),
        'high': 1750 + np.cumsum(np.random.randn(n) * 20),
        'low': 1650 + np.cumsum(np.random.randn(n) * 20),
        'close': 1700 + np.cumsum(np.random.randn(n) * 20),
        'volume': np.random.randint(50000, 200000, n),
    }, index=dates)

    # 添加前一收盘价（用于涨跌停计算）
    data['pre_close'] = data['close'].shift(1)
    data['pre_close'].fillna(data['close'].iloc[0], inplace=True)

    engine.add_data('600519.SH', data)

    # 简单策略
    from uuid import uuid4

    def simple_strategy(symbol: str, history: pd.DataFrame) -> list:
        """简单均线策略"""
        orders = []

        if len(history) < 20:
            return orders

        # 计算 MA
        ma5 = history['close'].rolling(5).mean()
        ma20 = history['close'].rolling(20).mean()

        latest = history.iloc[-1]
        prev = history.iloc[-2]

        # MA5 上穿 MA20 买入
        if prev['ma5'] < prev['ma20'] and latest['ma5'] > latest['ma20']:
            order = Order(
                id=str(uuid4()),
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100
            )
            orders.append(order)

        # MA5 下穿 MA20 卖出
        elif prev['ma5'] > prev['ma20'] and latest['ma5'] < latest['ma20']:
            order = Order(
                id=str(uuid4()),
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=100
            )
            orders.append(order)

        return orders

    # 运行回测
    results = engine.run(
        strategy=simple_strategy,
        start_date=dates[0],
        end_date=dates[-1]
    )

    # 打印结果
    print(f"\n📊 回测结果:")
    print(f"   初始资金: ¥{results['initial_capital']:,.0f}")
    print(f"   最终资金: ¥{results['final_equity']:,.0f}")
    print(f"   总收益: {results['total_return']:.2f}%")
    print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"   最大回撤: {results['max_drawdown']:.2f}%")
    print(f"   交易次数: {results['num_trades']}")

    return results


async def example_data_fetching():
    """示例：数据获取（需要 Tushare Token）"""
    print("\n" + "="*60)
    print("📥 示例 4: 数据获取")
    print("="*60)

    print("⚠️  此示例需要 Tushare Token")
    print("   请在 .env 文件中设置 TUSHARE_TOKEN")
    print("   或使用环境变量")

    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ 未找到 TUSHARE_TOKEN，跳过此示例")
        return

    # 创建数据提供者
    async with TushareProProvider(token) as provider:
        # 获取股票列表
        stocks = await provider.get_stock_list()
        print(f"✅ 获取到 {len(stocks)} 只股票")

        # 获取指数行情
        index = await provider.get_index_daily(
            ts_code='000001.SH',
            start_date='20240101',
            end_date='20240131'
        )
        print(f"✅ 上证指数数据: {len(index)} 条")
        print(index.head())

        # 获取涨跌停列表
        limits = await provider.get_limit_list(trade_date='20240131')
        print(f"✅ 涨停股票: {len(limits[limits['limit_type'] == 'U'])} 只")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 AlphaQuant 运行示例")
    print("="*60)
    print("\n本脚本演示 AlphaQuant 的核心功能：")
    print("1. 因子计算")
    print("2. 模型推理")
    print("3. 策略回测")
    print("4. 数据获取")
    print()

    try:
        # 示例 1: 因子计算
        example_factor_computation()

        # 示例 2: 模型推理
        example_model_inference()

        # 示例 3: 回测
        example_backtest()

        # 示例 4: 数据获取（可选）
        import os
        if os.environ.get('TUSHARE_TOKEN'):
            asyncio.run(example_data_fetching())

        print("\n" + "="*60)
        print("✅ 所有示例运行完成！")
        print("="*60)

    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
