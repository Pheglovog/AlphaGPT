#!/usr/bin/env python3
"""
AlphaGPT 每日选股 + 持仓分析 v3+

早上8点运行：
1. 选出 Top 10 买入候选
2. 分析持仓是否应该卖出
3. 生成完整交易建议

卖出逻辑（义父亲授）：
- 趋势破坏：跌破5日均线
- 放量滞涨：量放大但涨幅小
- 长上影：出货信号
- 市场环境变差
- 止损/止盈触发
"""
import asyncio
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from alphaquant.data_providers.tushare import TushareProProvider

# 持仓文件路径
HOLDINGS_FILE = Path(__file__).parent / 'holdings.json'


def load_holdings():
    """加载持仓"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"positions": [], "cash": 10000, "last_update": ""}


def save_holdings(holdings):
    """保存持仓"""
    holdings['last_update'] = datetime.now().strftime('%Y-%m-%d')
    with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


async def fetch_data(provider, trade_date):
    """获取当日数据"""
    daily = await provider.get_daily_quotes(trade_date=trade_date)
    if daily is None or len(daily) == 0:
        prev_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        daily = await provider.get_daily_quotes(trade_date=prev_date)
    
    if daily is not None and len(daily) > 0:
        trade_date_used = daily['trade_date'].iloc[0]
        return daily, trade_date_used
    return None, None


async def get_index_data(provider, trade_date, days=20):
    """获取大盘指数"""
    end_date = datetime.strptime(trade_date, '%Y%m%d')
    start_date = (end_date - timedelta(days=days)).strftime('%Y%m%d')
    return await provider.get_index_daily(ts_code='000001.SH', start_date=start_date, end_date=trade_date)


async def get_all_stock_names(provider):
    """获取所有股票名称"""
    sse = await provider.get_stock_list(exchange='SSE')
    szse = await provider.get_stock_list(exchange='SZSE')
    stock_list = pd.concat([sse, szse])
    return dict(zip(stock_list['ts_code'], stock_list['name']))


def analyze_position_sell(stock_data: pd.DataFrame, position: dict, market_ok: bool) -> dict:
    """
    分析持仓是否应该卖出
    
    卖出信号：
    1. 趋势破坏：收盘价 < 5日均线
    2. 放量滞涨：量能>2倍均量，但涨幅<1%
    3. 长上影：上影线 > 实体*2
    4. 止损：亏损超过-5%
    5. 止盈：盈利超过+15%
    6. 市场环境：大盘不好 + 持仓亏损
    """
    if stock_data is None or len(stock_data) == 0:
        return {"should_sell": False, "reason": "无法获取数据"}
    
    latest = stock_data.iloc[-1]
    signals = []
    should_sell = False
    
    # 基础数据
    close = latest['close']
    pct_chg = latest['pct_chg']
    high = latest['high']
    low = latest['low']
    open_price = latest['open']
    vol = latest['vol']
    
    # 买入价和盈亏
    buy_price = position.get('buy_price', close)
    pnl_pct = (close - buy_price) / buy_price * 100
    
    # 1. 趋势破坏
    if len(stock_data) >= 5:
        ma5 = stock_data['close'].tail(5).mean()
        if close < ma5:
            signals.append("📉 跌破5日均线")
            should_sell = True
    
    # 2. 放量滞涨
    if len(stock_data) >= 5:
        vol_ma5 = stock_data['vol'].tail(5).mean()
        if vol > vol_ma5 * 2 and abs(pct_chg) < 1:
            signals.append("⚠️ 放量滞涨")
            should_sell = True
    
    # 3. 长上影（出货信号）
    upper_shadow = high - max(open_price, close)
    body = abs(close - open_price)
    if body > 0 and upper_shadow > body * 2:
        signals.append("🔴 长上影出货")
        should_sell = True
    
    # 4. 止损
    if pnl_pct < -5:
        signals.append(f"🛑 触发止损 ({pnl_pct:.1f}%)")
        should_sell = True
    
    # 5. 止盈
    if pnl_pct > 15:
        signals.append(f"💰 触发止盈 ({pnl_pct:.1f}%)")
        should_sell = True
    
    # 6. 市场环境恶化 + 亏损
    if not market_ok and pnl_pct < 0:
        signals.append(f"🌊 大盘不好+亏损 ({pnl_pct:.1f}%)")
        should_sell = True
    
    # 7. 大阴线
    if pct_chg < -5:
        signals.append(f"⚫ 大阴线 ({pct_chg:.1f}%)")
        should_sell = True
    
    return {
        "should_sell": should_sell,
        "signals": signals,
        "pnl_pct": pnl_pct,
        "current_price": close,
        "buy_price": buy_price
    }


def calculate_v3_scores(daily: pd.DataFrame, index_data: pd.DataFrame = None) -> pd.DataFrame:
    """v3 游资版评分"""
    df = daily.copy()
    
    # 市场环境判断
    market_ok = True
    market_score_adj = 1.0
    
    if index_data is not None and len(index_data) >= 5:
        index_latest = index_data.iloc[-1]
        index_ma5 = index_data['close'].tail(5).mean()
        
        if index_latest['close'] < index_ma5 and index_latest['pct_chg'] < -1:
            market_ok = False
            market_score_adj = 0.7
        elif index_latest['close'] < index_ma5:
            market_score_adj = 0.85
    
    # 基础过滤
    df = df[df['pct_chg'] > 0].copy()
    df = df[df['close'] > df['open']].copy()
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    df = df[df['body_ratio'] > 0.3].copy()
    df = df[df['amount'] > 2000].copy()
    amount_threshold = df['amount'].quantile(0.99)
    df = df[df['amount'] < amount_threshold].copy()
    df = df[df['pct_chg'] < 8.5].copy()
    
    # 因子计算
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    df['amount_rank'] = df['amount'].rank(pct=True)
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-6)
    
    # 综合评分
    df['score'] = (
        df['price_position'] * 30 +
        df['body_ratio'] * 20 +
        df['amount_rank'] * 15 +
        (1 - df['upper_shadow_ratio']) * 10
    ) * market_score_adj
    
    # 形态识别
    df['is_small_yang'] = ((df['pct_chg'] > 1) & (df['pct_chg'] < 4)).astype(int)
    df['is_big_body'] = (df['body_ratio'] > 0.5).astype(int)
    df['is_close_high'] = (df['price_position'] > 0.95).astype(int)
    df['is_short_shadow'] = (df['upper_shadow_ratio'] < 0.1).astype(int)
    
    df['pattern'] = ""
    df.loc[df['is_small_yang'] == 1, 'pattern'] += "小阳推进 "
    df.loc[df['is_big_body'] == 1, 'pattern'] += "大实体 "
    df.loc[df['is_close_high'] == 1, 'pattern'] += "收盘最高 "
    df.loc[df['is_short_shadow'] == 1, 'pattern'] += "上影极短"
    df['pattern'] = df['pattern'].str.strip()
    df.loc[df['pattern'] == "", 'pattern'] = "一般阳线"
    
    df['market_ok'] = market_ok
    
    return df, market_ok


async def main():
    """主函数"""
    print("=" * 60)
    print(f"🔥 AlphaGPT 每日选股 + 持仓分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
        
        # 获取数据
        print("📊 获取行情数据...")
        daily, trade_date = await fetch_data(provider, today)
        if daily is None:
            return None
        
        # 获取大盘指数
        print("📈 获取大盘指数...")
        index_data = await get_index_data(provider, today, days=20)
        
        # 计算选股分数
        print("🔬 计算选股分数...")
        df_scores, market_ok = calculate_v3_scores(daily, index_data)
        
        # 加载持仓
        holdings = load_holdings()
        positions = holdings.get('positions', [])
        
        # ========== 第一部分：持仓分析 ==========
        sell_recommendations = []
        
        if positions:
            print("\n" + "=" * 60)
            print("📦 持仓分析")
            print("=" * 60)
            
            for pos in positions:
                ts_code = pos.get('ts_code')
                stock_name = stock_names.get(ts_code, ts_code)
                
                # 获取该股票数据
                stock_data = daily[daily['ts_code'] == ts_code].copy()
                
                # 分析卖出信号
                analysis = analyze_position_sell(stock_data, pos, market_ok)
                
                sell_rec = {
                    'ts_code': ts_code,
                    'name': stock_name,
                    'shares': pos.get('shares', 0),
                    'buy_price': pos.get('buy_price', 0),
                    **analysis
                }
                sell_recommendations.append(sell_rec)
                
                # 输出
                status = "🚨 建议卖出" if analysis['should_sell'] else "✅ 继续持有"
                print(f"\n{stock_name} ({ts_code})")
                print(f"   持仓: {pos.get('shares', 0)}股 @ ¥{pos.get('buy_price', 0):.2f}")
                print(f"   现价: ¥{analysis['current_price']:.2f} | 盈亏: {analysis['pnl_pct']:+.2f}%")
                print(f"   状态: {status}")
                if analysis['signals']:
                    print(f"   信号: {' | '.join(analysis['signals'])}")
        else:
            print("\n📦 当前无持仓")
        
        # ========== 第二部分：买入候选 ==========
        if len(df_scores) == 0:
            print("❌ 无符合条件的股票")
            return None
        
        # 选出 Top 10
        premium = df_scores[
            (df_scores['is_close_high'] == 1) | 
            (df_scores['is_big_body'] == 1) |
            (df_scores['is_short_shadow'] == 1)
        ].copy()
        
        if len(premium) >= 10:
            top_stocks = premium.nlargest(10, 'score')
        else:
            top_stocks = df_scores.nlargest(10, 'score')
        
        top_stocks['name'] = top_stocks['ts_code'].map(stock_names)
        top_stocks['name'] = top_stocks['name'].fillna(top_stocks['ts_code'])
        
        # 输出买入候选
        print("\n" + "=" * 60)
        print(f"🎯 {trade_date} 买入候选 Top 10")
        print("=" * 60)
        
        market_status = "🟢 正常" if market_ok else "🟡 谨慎"
        print(f"📊 市场环境: {market_status}")
        print("=" * 60)
        
        buy_text = []
        for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
            pattern = row['pattern'] if row['pattern'] != "一般阳线" else "阳线"
            line = f"{i}. {row['name']} ({row['ts_code']})\n   💰 ¥{row['close']:.2f} | {row['pct_chg']:+.2f}% | 分数 {row['score']:.1f}\n   📝 {pattern}"
            print(f"\n{line}")
            buy_text.append(line)
        
        # ========== 保存结果 ==========
        result_file = Path(__file__).parent / 'daily_pick_result_v3_plus.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"AlphaGPT 每日交易报告\n")
            f.write(f"日期: {trade_date}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"市场环境: {market_status}\n")
            f.write("=" * 60 + "\n\n")
            
            # 持仓分析
            if sell_recommendations:
                f.write("📦 持仓分析\n")
                f.write("-" * 60 + "\n")
                for rec in sell_recommendations:
                    status = "🚨 建议卖出" if rec['should_sell'] else "✅ 继续持有"
                    f.write(f"\n{rec['name']} ({rec['ts_code']})\n")
                    f.write(f"   持仓: {rec['shares']}股 @ ¥{rec['buy_price']:.2f}\n")
                    f.write(f"   现价: ¥{rec['current_price']:.2f} | 盈亏: {rec['pnl_pct']:+.2f}%\n")
                    f.write(f"   状态: {status}\n")
                    if rec['signals']:
                        f.write(f"   信号: {' | '.join(rec['signals'])}\n")
                f.write("\n" + "=" * 60 + "\n\n")
            
            # 买入候选
            f.write("🎯 买入候选 Top 10\n")
            f.write("-" * 60 + "\n\n")
            f.write("\n\n".join(buy_text))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！\n")
        
        print(f"\n✅ 结果已保存到: {result_file}")
        
        return {
            'sell_recommendations': sell_recommendations,
            'buy_candidates': top_stocks.to_dict('records'),
            'market_ok': market_ok
        }


if __name__ == '__main__':
    asyncio.run(main())
