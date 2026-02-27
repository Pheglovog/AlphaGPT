#!/usr/bin/env python3
"""
每日选股脚本 - AlphaGPT
每天早上8点运行，挑选10只最有可能上涨的股票
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from alphaquant.data_providers.tushare import TushareProProvider
import torch


async def fetch_stock_data(provider: TushareProProvider, trade_date: str):
    """获取当日全市场数据"""
    print(f"📊 获取 {trade_date} 行情数据...")
    
    daily = await provider.get_daily_quotes(trade_date=trade_date)
    if daily is None or len(daily) == 0:
        prev_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        print(f"⚠️ {trade_date} 无数据，尝试 {prev_date}...")
        daily = await provider.get_daily_quotes(trade_date=prev_date)
    
    if daily is None or len(daily) == 0:
        print(f"❌ 无法获取行情数据")
        return None, None
    
    print(f"✅ 获取到 {len(daily)} 只股票数据")
    trade_date_used = daily['trade_date'].iloc[0] if len(daily) > 0 else trade_date
    return daily, trade_date_used


def calculate_scores(daily: pd.DataFrame) -> pd.DataFrame:
    """计算综合得分"""
    print("🔬 计算因子分数...")
    
    df = daily.copy()
    
    # 1. 涨跌幅
    df['pct_chg'] = df['pct_chg'].fillna(0)
    
    # 2. 价格相对位置 (收盘价在当日高低点中的位置)
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    
    # 3. 成交量相对强度
    avg_vol = df['vol'].mean()
    df['vol_ratio'] = df['vol'] / (avg_vol + 1e-6)
    
    # 4. 振幅
    df['amplitude'] = (df['high'] - df['low']) / (df['pre_close'] + 1e-6)
    
    # 5. 实体大小 (阳线/阴线)
    df['body'] = (df['close'] - df['open']) / (df['pre_close'] + 1e-6)
    
    # 综合得分
    df['score'] = (
        df['price_position'] * 30 +      # 价格位置权重 30
        df['body'].clip(-0.05, 0.1) * 200 +  # 阳线加分
        df['vol_ratio'].clip(0, 3) * 10 +    # 成交量权重 10
        (5 - df['amplitude'].clip(0, 5) * 50)  # 稳定性 (振幅小加分)
    )
    
    # 涨幅加成 (但不能涨停)
    df.loc[(df['pct_chg'] > 0) & (df['pct_chg'] < 9), 'score'] += df['pct_chg'] * 2
    
    print(f"✅ 计算了 {len(df)} 只股票的因子分数")
    return df


def select_top_stocks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """选出得分最高的股票"""
    print(f"🎯 筛选 Top {top_n} 股票...")
    
    # 过滤条件
    filtered = df[
        (df['score'] > 0) &
        (df['pct_chg'] < 9.5) &  # 未涨停
        (df['pct_chg'] > -9.5) &  # 未跌停
        (df['vol'] > 0)  # 有成交量
    ].copy()
    
    # 按分数排序
    top = filtered.nlargest(top_n, 'score')
    
    return top


def generate_reasons(row: pd.Series) -> str:
    """生成上涨原因"""
    reasons = []
    
    # 价格位置
    if row['price_position'] > 0.8:
        reasons.append("📈 收盘接近最高价，强势明显")
    elif row['price_position'] > 0.6:
        reasons.append("📊 收盘位置较好，买盘占优")
    
    # 涨幅
    if row['pct_chg'] > 5:
        reasons.append(f"🚀 大涨 {row['pct_chg']:.2f}%")
    elif row['pct_chg'] > 2:
        reasons.append(f"✅ 涨幅 {row['pct_chg']:.2f}%")
    elif row['pct_chg'] > 0:
        reasons.append(f"📈 微涨 {row['pct_chg']:.2f}%")
    
    # 成交量
    if row['vol_ratio'] > 2:
        reasons.append(f"💰 放量 {row['vol_ratio']:.1f}倍")
    elif row['vol_ratio'] > 1.2:
        reasons.append("💹 量能活跃")
    
    # 实体
    if row['body'] > 0.03:
        reasons.append("🔴 实体阳线")
    elif row['body'] > 0:
        reasons.append("📊 小阳线")
    
    # 稳定性
    if row['amplitude'] < 0.03:
        reasons.append("🎯 走势稳健")
    
    if not reasons:
        reasons.append("📊 综合因子得分较高")
    
    return " | ".join(reasons)


async def main():
    """主函数"""
    print("=" * 60)
    print(f"🚀 AlphaGPT 每日选股 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ 未找到 TUSHARE_TOKEN")
        return None
    
    async with TushareProProvider(token) as provider:
        # 获取最近交易日数据
        today = datetime.now().strftime('%Y%m%d')
        daily, trade_date = await fetch_stock_data(provider, today)
        
        if daily is None:
            return None
        
        # 计算分数
        df = calculate_scores(daily)
        
        # 选出 Top 10
        top_stocks = select_top_stocks(df, top_n=10)
        
        # 获取股票名称
        stock_list = await provider.get_stock_list()
        top_stocks = top_stocks.merge(stock_list[['ts_code', 'name']], on='ts_code', how='left')
        top_stocks['name'] = top_stocks['name'].fillna(top_stocks['ts_code'])
        
        # 输出结果
        print("\n" + "=" * 60)
        print(f"🎯 {trade_date} 推荐股票 Top 10")
        print("=" * 60)
        
        result_text = []
        for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
            reason = generate_reasons(row)
            line = f"{i}. {row['name']} ({row['ts_code']})\n   💰 现价: {row['close']:.2f} | 涨幅: {row['pct_chg']:+.2f}% | 分数: {row['score']:.1f}\n   📝 {reason}"
            print(f"\n{line}")
            result_text.append(line)
        
        print("\n" + "=" * 60)
        print("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！")
        print("=" * 60)
        
        # 保存结果
        result_file = Path(__file__).parent / 'daily_pick_result.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"AlphaGPT 每日选股报告\n")
            f.write(f"日期: {trade_date}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n\n".join(result_text))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！\n")
        
        print(f"\n✅ 结果已保存到: {result_file}")
        
        return top_stocks


if __name__ == '__main__':
    asyncio.run(main())
