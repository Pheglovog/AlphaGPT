#!/usr/bin/env python3
"""测试 Tushare Pro 连接"""
import os
import asyncio
from dotenv import load_dotenv
from alphaquant.data_providers.tushare import TushareProProvider

# 加载环境变量
load_dotenv()

token = os.getenv('TUSHARE_TOKEN')
if not token:
    print("❌ 未找到 TUSHARE_TOKEN，请检查 .env 文件")
    exit(1)

print(f"Token: {token[:20]}...{token[-20:]}")
print("="*60)


async def test_connection():
    """测试 Tushare Pro 连接"""

    # 创建 Provider（2000积分 = 5个并发）
    async with TushareProProvider(token, concurrent_limit=5) as provider:
        print("\n📊 测试 1: 获取股票列表")
        try:
            stocks = await provider.get_stock_list(exchange='SSE')
            print(f"✅ 成功获取 {len(stocks)} 只上交所股票")
            if len(stocks) > 0:
                print(f"   示例: {stocks.iloc[0]['name']} ({stocks.iloc[0]['ts_code']})")
        except Exception as e:
            print(f"❌ 失败: {e}")

        print("\n📊 测试 2: 获取指数行情")
        try:
            index = await provider.get_index_daily(
                ts_code='000001.SH',
                start_date='20240101',
                end_date='20240131'
            )
            print(f"✅ 成功获取 {len(index)} 条上证指数数据")
            if len(index) > 0:
                print(f"   最新: {index.iloc[-1]['trade_date']} 收盘 {index.iloc[-1]['close']:.2f}")
        except Exception as e:
            print(f"❌ 失败: {e}")

        print("\n📊 测试 3: 获取贵州茅台日线")
        try:
            data = await provider.get_daily_quotes(
                ts_code='600519.SH',
                start_date='20240101',
                end_date='20240131'
            )
            print(f"✅ 成功获取 {len(data)} 条贵州茅台数据")
            if len(data) > 0:
                print(f"   最新: {data.iloc[-1]['trade_date']} 收盘 {data.iloc[-1]['close']:.2f}")
        except Exception as e:
            print(f"❌ 失败: {e}")

        print("\n📊 测试 4: 获取涨跌停列表")
        try:
            limits = await provider.get_limit_list(trade_date='20240131')
            print(f"✅ 成功获取 {len(limits)} 条涨跌停数据")
            if len(limits) > 0:
                print(f"   示例: {limits.iloc[0]['name']} 涨幅 {limits.iloc[0]['pct_chg']:.2f}%")
        except Exception as e:
            print(f"❌ 失败: {e}")

    print("\n" + "="*60)
    print("✅ 所有测试完成！Tushare Pro 连接正常")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_connection())
