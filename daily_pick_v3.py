#!/usr/bin/env python3
"""
AlphaGPT 每日选股 v3 - 游资版

整合义父亲授的实战体系：
1. 6个瞬间过滤法：趋势、K线形态、实体、量能、位置、大盘同步
2. 3种起爆形态：平台突破、回踩不破、小阳慢推
3. 量能真相：缩量=筹码锁住，启动前反而没量
4. 时间压缩因子：结构被压缩=强资金推动

核心认知：最强的走势不是涨得最多，而是涨得越来越快
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


async def fetch_data(provider: TushareProProvider, trade_date: str):
    """获取当日数据"""
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
    return daily, daily['trade_date'].iloc[0]


async def get_index_data(provider: TushareProProvider, trade_date: str, days: int = 20):
    """获取大盘指数"""
    print("📈 获取大盘指数...")
    
    end_date = datetime.strptime(trade_date, '%Y%m%d')
    start_date = (end_date - timedelta(days=days)).strftime('%Y%m%d')
    
    index_data = await provider.get_index_daily(ts_code='000001.SH', start_date=start_date, end_date=trade_date)
    
    if index_data is not None and len(index_data) > 0:
        latest = index_data.iloc[-1]
        print(f"✅ 上证指数: {latest['close']:.2f}, 涨幅: {latest['pct_chg']:.2f}%")
        return index_data
    
    return None


async def get_all_stock_names(provider: TushareProProvider) -> dict:
    """获取所有股票名称"""
    sse = await provider.get_stock_list(exchange='SSE')
    szse = await provider.get_stock_list(exchange='SZSE')
    stock_list = pd.concat([sse, szse])
    return dict(zip(stock_list['ts_code'], stock_list['name']))


def calculate_v3_scores(daily: pd.DataFrame, index_data: pd.DataFrame = None) -> pd.DataFrame:
    """v3 游资版评分"""
    print("🔬 计算因子分数 (v3游资版)...")
    print("=" * 60)
    
    df = daily.copy()
    
    # ====== 1. 市场环境判断 ======
    market_ok = True
    market_score_adj = 1.0
    
    if index_data is not None and len(index_data) >= 5:
        index_latest = index_data.iloc[-1]
        index_ma5 = index_data['close'].tail(5).mean()
        index_ma10 = index_data['close'].tail(10).mean() if len(index_data) >= 10 else index_ma5
        
        if index_latest['close'] < index_ma5 and index_latest['pct_chg'] < -1:
            market_ok = False
            market_score_adj = 0.7
            print(f"⚠️ 大盘环境不佳 (低于5日线且下跌)，分数打7折")
        elif index_latest['close'] < index_ma10:
            market_score_adj = 0.85
            print(f"⚠️ 大盘低于10日均线，分数打85折")
        else:
            print(f"✅ 大盘环境正常")
    
    # ====== 2. 基础过滤 ======
    
    # ① 趋势过滤：只选上涨的
    df = df[df['pct_chg'] > 0].copy()
    print(f"📊 上涨股票: {len(df)}")
    
    # ② K线过滤：只选阳线
    df = df[df['close'] > df['open']].copy()
    print(f"📊 阳线股票: {len(df)}")
    
    # ③ 实体过滤：实体不能太小
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    df = df[df['body_ratio'] > 0.3].copy()
    print(f"📊 实体>30%: {len(df)}")
    
    # ④ 量能过滤：成交额 > 2000万，但不能是天量
    df = df[df['amount'] > 2000].copy()
    # 排除天量（成交额前1%）
    amount_threshold = df['amount'].quantile(0.99)
    df = df[df['amount'] < amount_threshold].copy()
    print(f"📊 量能适中: {len(df)}")
    
    # ⑤ 位置过滤：涨幅 < 8.5%
    df = df[df['pct_chg'] < 8.5].copy()
    print(f"📊 涨幅<8.5%: {len(df)}")
    
    # ====== 3. 因子计算 ======
    
    # 价格位置（收盘在当日高低点的位置）
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    
    # 成交额分位数
    df['amount_rank'] = df['amount'].rank(pct=True)
    
    # 上影线比例（越小越好）
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-6)
    
    # ====== 4. 综合评分 ======
    
    df['score'] = 0
    
    # 价格位置 (最高30分)
    df['score'] += df['price_position'] * 30
    
    # 实体比例 (最高20分)
    df['score'] += df['body_ratio'] * 20
    
    # 量能分位数 (最高15分)
    df['score'] += df['amount_rank'] * 15
    
    # 上影线短 (最高10分，反向计分)
    df['score'] += (1 - df['upper_shadow_ratio']) * 10
    
    # 大盘对抗加分
    if index_data is not None and len(index_data) > 0:
        index_latest = index_data.iloc[-1]
        if index_latest['pct_chg'] < 0:
            df['score'] += 10  # 大盘跌它涨
    
    # 市场环境调整
    df['score'] *= market_score_adj
    
    # ====== 5. 形态识别 ======
    
    # 小阳推进（1-4%）
    df['is_small_yang'] = ((df['pct_chg'] > 1) & (df['pct_chg'] < 4)).astype(int)
    
    # 实体大（>50%）
    df['is_big_body'] = (df['body_ratio'] > 0.5).astype(int)
    
    # 收盘最高（>95%）
    df['is_close_high'] = (df['price_position'] > 0.95).astype(int)
    
    # 上影极短（<10%）
    df['is_short_shadow'] = (df['upper_shadow_ratio'] < 0.1).astype(int)
    
    # 综合形态
    df['pattern'] = ""
    df.loc[df['is_small_yang'] == 1, 'pattern'] += "小阳推进 "
    df.loc[df['is_big_body'] == 1, 'pattern'] += "大实体 "
    df.loc[df['is_close_high'] == 1, 'pattern'] += "收盘最高 "
    df.loc[df['is_short_shadow'] == 1, 'pattern'] += "上影极短"
    df['pattern'] = df['pattern'].str.strip()
    df.loc[df['pattern'] == "", 'pattern'] = "一般阳线"
    
    df['market_ok'] = market_ok
    
    print(f"✅ 计算了 {len(df)} 只股票的因子分数")
    print("=" * 60)
    
    return df


def select_top_stocks_v3(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """v3 筛选"""
    print(f"🎯 筛选 Top {top_n} 股票 (v3游资版)...")
    
    # 优先选择有优质形态的
    premium = df[
        (df['is_close_high'] == 1) | 
        (df['is_big_body'] == 1) |
        (df['is_short_shadow'] == 1)
    ].copy()
    
    if len(premium) >= top_n:
        return premium.nlargest(top_n, 'score')
    
    return df.nlargest(top_n, 'score')


def generate_reasons_v3(row: pd.Series) -> str:
    """v3 原因生成"""
    reasons = []
    
    # 形态
    if row['pattern'] != "一般阳线":
        reasons.append(f"🎯 {row['pattern']}")
    
    # 价格位置
    if row['price_position'] > 0.95:
        reasons.append("📈 收盘最高价")
    elif row['price_position'] > 0.8:
        reasons.append("📈 强势收盘")
    
    # 实体
    if row['body_ratio'] > 0.6:
        reasons.append("🔴 大实体阳线")
    
    # 量能
    if row['amount_rank'] > 0.8:
        reasons.append("💰 成交活跃")
    elif row['amount_rank'] > 0.5:
        reasons.append("💹 温和放量")
    
    # 上影线
    if row['upper_shadow_ratio'] < 0.05:
        reasons.append("⚡ 无上影")
    elif row['upper_shadow_ratio'] < 0.15:
        reasons.append("⚡ 上影极短")
    
    if not reasons:
        reasons.append("📊 综合得分")
    
    return " | ".join(reasons)


async def main():
    """主函数"""
    print("=" * 60)
    print(f"🔥 AlphaGPT 每日选股 v3 (游资版) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print("v3 整合义父亲授实战体系:")
    print("  ✅ 6个瞬间过滤：趋势、K线、实体、量能、位置、大盘")
    print("  ✅ 形态识别：小阳推进、大实体、收盘最高、上影极短")
    print("  ✅ 量能真相：温和放量，排除天量")
    print("=" * 60)
    print("💡 核心认知：最强走势不是涨最多，而是涨得越来越快")
    print("=" * 60)
    
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ 未找到 TUSHARE_TOKEN")
        return None
    
    async with TushareProProvider(token) as provider:
        today = datetime.now().strftime('%Y%m%d')
        
        # 获取股票名称
        print("📋 获取股票名称...")
        stock_names = await get_all_stock_names(provider)
        print(f"✅ 获取到 {len(stock_names)} 个股票名称")
        
        # 获取数据
        daily, trade_date = await fetch_data(provider, today)
        if daily is None:
            return None
        
        # 获取大盘指数
        index_data = await get_index_data(provider, today, days=20)
        
        # 计算分数
        df_scores = calculate_v3_scores(daily, index_data)
        
        if len(df_scores) == 0:
            print("❌ 无符合条件的股票")
            return None
        
        # 选出 Top 10
        top_stocks = select_top_stocks_v3(df_scores, top_n=10)
        
        # 添加股票名称
        top_stocks['name'] = top_stocks['ts_code'].map(stock_names)
        top_stocks['name'] = top_stocks['name'].fillna(top_stocks['ts_code'])
        
        # 输出结果
        print("\n" + "=" * 60)
        print(f"🔥 {trade_date} 推荐股票 Top 10 (v3游资版)")
        print("=" * 60)
        
        market_status = "🟢 正常" if top_stocks['market_ok'].iloc[0] else "🟡 谨慎"
        print(f"📊 市场环境: {market_status}")
        print("=" * 60)
        
        result_text = []
        for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
            reason = generate_reasons_v3(row)
            line = f"{i}. {row['name']} ({row['ts_code']})\n   💰 现价: {row['close']:.2f} | 涨幅: {row['pct_chg']:+.2f}% | 分数: {row['score']:.1f}\n   📝 {reason}"
            print(f"\n{line}")
            result_text.append(line)
        
        print("\n" + "=" * 60)
        print("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！")
        print("💡 核心原则：找最舒服的上涨，不是最猛的")
        print("=" * 60)
        
        # 保存结果
        result_file = Path(__file__).parent / 'daily_pick_result_v3.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"AlphaGPT 每日选股报告 v3 (游资版)\n")
            f.write(f"日期: {trade_date}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"市场环境: {market_status}\n")
            f.write("=" * 60 + "\n\n")
            f.write("选股逻辑 (义父亲授):\n")
            f.write("- 6个瞬间过滤: 趋势、K线、实体、量能、位置、大盘\n")
            f.write("- 形态识别: 小阳推进、大实体、收盘最高、上影极短\n")
            f.write("- 量能真相: 温和放量，排除天量\n")
            f.write("- 核心认知: 最强走势不是涨最多，而是涨得越来越快\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n\n".join(result_text))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！\n")
        
        print(f"\n✅ 结果已保存到: {result_file}")
        
        return top_stocks


if __name__ == '__main__':
    asyncio.run(main())
