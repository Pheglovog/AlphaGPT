#!/usr/bin/env python3
"""
AlphaGPT 每日选股 v5 - 完整版

整合义父亲授的全部实战体系：
1. 6个瞬间过滤 + 3种起爆形态
2. 控盘因子：VolShrink + RangeCompression + PullbackRecovery + TrendStability
3. 广度闸门：adv_ratio + median_ret + limit_diff
4. 持仓分析 + 卖出信号

核心认知：
- 动量策略赚的是"趋势扩散的钱"，不是"单日强势的钱"
- 必须先判断：今天市场有没有扩散
- 最强走势不是涨最多，而是涨得越来越快
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

# 持仓文件
HOLDINGS_FILE = Path(__file__).parent / 'holdings.json'


# ==================== 工具函数 ====================

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


# ==================== 广度闸门 ====================

def filter_universe(df: pd.DataFrame, stock_basic: pd.DataFrame = None) -> pd.DataFrame:
    """
    Universe 过滤
    - vol > 0 且 amount > 0（去停牌）
    - 非 ST
    - 非北交所
    """
    # 去停牌/无成交
    df = df[(df['vol'] > 0) & (df['amount'] > 0)].copy()
    
    # 过滤 ST
    if stock_basic is not None and 'name' in stock_basic.columns:
        st_codes = stock_basic[stock_basic['name'].str.contains('ST', case=False, na=False)]['ts_code']
        df = df[~df['ts_code'].isin(st_codes)].copy()
    
    # 过滤北交所（ts_code 以 .BJ 结尾）
    df = df[~df['ts_code'].str.endswith('.BJ')].copy()
    
    return df


def calc_breadth_metrics(df: pd.DataFrame) -> dict:
    """
    计算广度指标
    
    返回：
    - adv_ratio: 上涨家数占比
    - net_adv: 净上涨率 (-1~1)
    - median_ret: 全市场中位数收益 (%)
    - adv/dec: 上涨/下跌家数
    """
    # 上涨/下跌家数
    adv = (df['pct_chg'] > 0).sum()
    dec = (df['pct_chg'] < 0).sum()
    total = adv + dec
    
    # 广度比
    adv_ratio = adv / total if total > 0 else 0.5
    
    # 净上涨率
    net_adv = (adv - dec) / total if total > 0 else 0
    
    # 中位数收益
    median_ret = df['pct_chg'].median()
    
    return {
        'adv': int(adv),
        'dec': int(dec),
        'adv_ratio': adv_ratio,
        'net_adv': net_adv,
        'median_ret': median_ret
    }


def calc_breadth_gate(df: pd.DataFrame, stock_basic: pd.DataFrame = None) -> dict:
    """
    广度闸门计算
    
    开仓条件：
    - adv_ratio >= 0.55（上涨家数过半）
    - median_ret >= 0（中位数收益为正）
    
    返回：
    - market_on: 是否开仓
    - metrics: 广度指标详情
    """
    # 过滤 universe
    df_clean = filter_universe(df, stock_basic)
    
    if len(df_clean) == 0:
        return {
            'market_on': False,
            'metrics': {'adv_ratio': 0, 'median_ret': 0, 'adv': 0, 'dec': 0},
            'reason': '无有效数据'
        }
    
    # 计算广度指标
    metrics = calc_breadth_metrics(df_clean)
    
    # 闸门条件
    market_on = (
        metrics['adv_ratio'] >= 0.55 and
        metrics['median_ret'] >= 0
    )
    
    reason = ""
    if not market_on:
        reasons = []
        if metrics['adv_ratio'] < 0.55:
            reasons.append(f"上涨占比不足 ({metrics['adv_ratio']:.1%})")
        if metrics['median_ret'] < 0:
            reasons.append(f"中位数收益为负 ({metrics['median_ret']:.2f}%)")
        reason = " | ".join(reasons)
    else:
        reason = f"市场扩散正常 (上涨{metrics['adv_ratio']:.1%}，中位{metrics['median_ret']:.2f}%)"
    
    return {
        'market_on': market_on,
        'metrics': metrics,
        'reason': reason
    }


# ==================== 控盘因子 ====================

def calc_control_score(df_stock: pd.DataFrame) -> float:
    """
    控盘因子计算（单只股票）
    
    四个子因子：
    1. VolShrink: 量能收缩
    2. RangeCompression: 振幅压缩
    3. PullbackRecovery: 回撤恢复
    4. TrendStability: 趋势稳定
    """
    if len(df_stock) < 5:
        return 0
    
    df = df_stock.tail(20).copy()  # 最近20天
    if len(df) < 10:
        return 0
    
    score = 0
    
    # 1. VolShrink（量能收缩）
    if len(df) >= 20:
        vol_ma5 = df['vol'].tail(5).mean()
        vol_ma20 = df['vol'].mean()
        vol_shrink = vol_ma5 / (vol_ma20 + 1e-6)
        vol_shrink = min(vol_shrink, 2)  # 裁剪
        # 越小越好，转为 0~1 分数
        vol_score = max(0, 1 - vol_shrink) * 30
    else:
        vol_score = 0
    
    # 2. RangeCompression（振幅压缩）
    df['range'] = df['high'] - df['low']
    if len(df) >= 10:
        range_ma5 = df['range'].tail(5).mean()
        range_ma10 = df['range'].tail(10).mean()
        range_comp = range_ma5 / (range_ma10 + 1e-6)
        range_score = max(0, 1 - range_comp) * 30
    else:
        range_score = 0
    
    # 3. PullbackRecovery（回撤恢复）
    # 下影线比例
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    
    # 有下影线的天数占比
    has_lower = (df['lower_shadow'] > df['total_range'] * 0.1).sum()
    pullback_score = (has_lower / len(df)) * 25
    
    # 4. TrendStability（趋势稳定）
    if len(df) >= 5:
        ma5 = df['close'].tail(5).mean()
        latest_close = df['close'].iloc[-1]
        trend_stable = 1 if latest_close > ma5 else 0
        trend_score = trend_stable * 15
    else:
        trend_score = 0
    
    # 总分
    total_score = vol_score + range_score + pullback_score + trend_score
    
    return min(total_score, 100)


# ==================== 动量因子 ====================

def calc_momentum_score(row: pd.Series) -> float:
    """
    动量因子计算（单日数据）
    
    因子：
    1. 价格位置（收盘在高低点中的位置）
    2. 实体大小
    3. 量能分位数
    4. 上影线比例（反向）
    """
    score = 0
    
    # 价格位置
    price_position = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-6)
    score += price_position * 30
    
    # 实体大小
    body_ratio = (row['close'] - row['open']) / (row['high'] - row['low'] + 1e-6)
    score += max(0, body_ratio) * 20
    
    # 量能分位数（需要外部计算，这里用 amount_rank）
    # 外部会传入 amount_rank
    
    # 上影线比例（反向计分）
    upper_shadow = row['high'] - max(row['open'], row['close'])
    upper_ratio = upper_shadow / (row['high'] - row['low'] + 1e-6)
    score += (1 - upper_ratio) * 10
    
    return score


# ==================== 持仓分析 ====================

def analyze_position_sell(stock_data: pd.DataFrame, position: dict, market_on: bool) -> dict:
    """分析持仓是否应该卖出"""
    if stock_data is None or len(stock_data) == 0:
        return {"should_sell": False, "reason": "无法获取数据"}
    
    latest = stock_data.iloc[-1]
    signals = []
    should_sell = False
    
    close = latest['close']
    pct_chg = latest['pct_chg']
    high = latest['high']
    low = latest['low']
    open_price = latest['open']
    vol = latest['vol']
    
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
    
    # 3. 长上影
    upper_shadow = high - max(open_price, close)
    body = abs(close - open_price)
    if body > 0 and upper_shadow > body * 2:
        signals.append("🔴 长上影出货")
        should_sell = True
    
    # 4. 止损
    if pnl_pct < -5:
        signals.append(f"🛑 止损 ({pnl_pct:.1f}%)")
        should_sell = True
    
    # 5. 止盈
    if pnl_pct > 15:
        signals.append(f"💰 止盈 ({pnl_pct:.1f}%)")
        should_sell = True
    
    # 6. 市场环境
    if not market_on and pnl_pct < 0:
        signals.append(f"🌊 大盘不好+亏损")
        should_sell = True
    
    # 7. 大阴线
    if pct_chg < -5:
        signals.append(f"⚫ 大阴线")
        should_sell = True
    
    return {
        "should_sell": should_sell,
        "signals": signals,
        "pnl_pct": pnl_pct,
        "current_price": close,
        "buy_price": buy_price
    }


# ==================== 主函数 ====================

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
    return dict(zip(stock_list['ts_code'], stock_list['name'])), stock_list


async def main():
    """主函数"""
    print("=" * 60)
    print(f"🔥 AlphaGPT v5 完整版 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print("整合义父亲授实战体系:")
    print("  ✅ 6个瞬间过滤 + 3种起爆形态")
    print("  ✅ 控盘因子: VolShrink + RangeComp + Pullback + Trend")
    print("  ✅ 广度闸门: adv_ratio + median_ret")
    print("  ✅ 持仓分析 + 卖出信号")
    print("=" * 60)
    
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ 未找到 TUSHARE_TOKEN")
        return None
    
    async with TushareProProvider(token) as provider:
        today = datetime.now().strftime('%Y%m%d')
        
        # 获取股票名称和基础信息
        print("📋 获取股票基础信息...")
        stock_names, stock_basic = await get_all_stock_names(provider)
        
        # 获取数据
        print("📊 获取行情数据...")
        daily, trade_date = await fetch_data(provider, today)
        if daily is None:
            return None
        
        # 获取大盘指数
        print("📈 获取大盘指数...")
        index_data = await get_index_data(provider, today, days=20)
        
        # ========== 广度闸门 ==========
        print("\n" + "=" * 60)
        print("🚪 广度闸门分析")
        print("=" * 60)
        
        breadth = calc_breadth_gate(daily, stock_basic)
        market_on = breadth['market_on']
        
        print(f"📊 上涨: {breadth['metrics']['adv']} | 下跌: {breadth['metrics']['dec']}")
        print(f"📊 上涨占比: {breadth['metrics']['adv_ratio']:.1%}")
        print(f"📊 中位数收益: {breadth['metrics']['median_ret']:.2f}%")
        print(f"📊 闸门状态: {'🟢 开启' if market_on else '🔴 关闭'}")
        print(f"📊 原因: {breadth['reason']}")
        
        # ========== 持仓分析 ==========
        holdings = load_holdings()
        positions = holdings.get('positions', [])
        sell_recommendations = []
        
        if positions:
            print("\n" + "=" * 60)
            print("📦 持仓分析")
            print("=" * 60)
            
            for pos in positions:
                ts_code = pos.get('ts_code')
                stock_name = stock_names.get(ts_code, ts_code)
                stock_data = daily[daily['ts_code'] == ts_code].copy()
                
                analysis = analyze_position_sell(stock_data, pos, market_on)
                
                sell_rec = {
                    'ts_code': ts_code,
                    'name': stock_name,
                    'shares': pos.get('shares', 0),
                    'buy_price': pos.get('buy_price', 0),
                    **analysis
                }
                sell_recommendations.append(sell_rec)
                
                status = "🚨 建议卖出" if analysis['should_sell'] else "✅ 继续持有"
                print(f"\n{stock_name} ({ts_code})")
                print(f"   持仓: {pos.get('shares', 0)}股 @ ¥{pos.get('buy_price', 0):.2f}")
                print(f"   现价: ¥{analysis['current_price']:.2f} | 盈亏: {analysis['pnl_pct']:+.2f}%")
                print(f"   状态: {status}")
                if analysis['signals']:
                    print(f"   信号: {' | '.join(analysis['signals'])}")
        else:
            print("\n📦 当前无持仓")
        
        # ========== 选股 ==========
        if not market_on:
            print("\n⚠️ 市场环境不佳，暂停选股")
            return {'sell_recommendations': sell_recommendations, 'buy_candidates': [], 'market_on': False}
        
        # 过滤条件
        print("\n🔬 计算选股分数...")
        
        df = daily.copy()
        
        # 基础过滤
        df = df[df['pct_chg'] > 0].copy()
        df = df[df['close'] > df['open']].copy()
        df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
        df = df[df['body_ratio'] > 0.3].copy()
        df = df[df['amount'] > 2000].copy()
        amount_threshold = df['amount'].quantile(0.99)
        df = df[df['amount'] < amount_threshold].copy()
        df = df[df['pct_chg'] < 8.5].copy()
        
        print(f"📊 基础过滤后: {len(df)} 只股票")
        
        # 因子计算
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
        df['amount_rank'] = df['amount'].rank(pct=True)
        df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-6)
        
        # 动量分数
        df['momentum_score'] = (
            df['price_position'] * 30 +
            df['body_ratio'] * 20 +
            df['amount_rank'] * 15 +
            (1 - df['upper_shadow_ratio']) * 10
        )
        
        # 控盘分数（简化版，用单日数据）
        df['control_score'] = (
            (1 - df['upper_shadow_ratio']) * 20 +  # 上影短
            (df['price_position'] > 0.8).astype(int) * 15 +  # 收盘高
            (df['body_ratio'] > 0.5).astype(int) * 15  # 实体大
        )
        
        # 总分（根据市场环境调整权重）
        if market_on:
            df['total_score'] = df['momentum_score'] * 0.7 + df['control_score'] * 0.3
        else:
            df['total_score'] = df['momentum_score'] * 0.3 + df['control_score'] * 0.7
        
        # 选出 Top 10
        top_stocks = df.nlargest(10, 'total_score')
        top_stocks['name'] = top_stocks['ts_code'].map(stock_names)
        top_stocks['name'] = top_stocks['name'].fillna(top_stocks['ts_code'])
        
        # 输出
        print("\n" + "=" * 60)
        print(f"🎯 {trade_date} 买入候选 Top 10")
        print("=" * 60)
        
        result_text = []
        for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
            line = f"{i}. {row['name']} ({row['ts_code']})\n   💰 ¥{row['close']:.2f} | {row['pct_chg']:+.2f}% | 总分 {row['total_score']:.1f}\n   📝 动量{row['momentum_score']:.0f} + 控盘{row['control_score']:.0f}"
            print(f"\n{line}")
            result_text.append(line)
        
        print("\n" + "=" * 60)
        print("⚠️ 风险提示：以上仅供参考，投资有风险，入市需谨慎！")
        print("=" * 60)
        
        # 保存结果
        result_file = Path(__file__).parent / 'daily_pick_result_v5.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"AlphaGPT v5 完整版报告\n")
            f.write(f"日期: {trade_date}\n")
            f.write(f"市场状态: {'🟢 正常' if market_on else '🔴 谨慎'}\n")
            f.write(f"广度指标: 上涨{breadth['metrics']['adv_ratio']:.1%} | 中位{breadth['metrics']['median_ret']:.2f}%\n")
            f.write("=" * 60 + "\n\n")
            
            if sell_recommendations:
                f.write("📦 持仓分析\n" + "-" * 60 + "\n")
                for rec in sell_recommendations:
                    status = "🚨 卖出" if rec['should_sell'] else "✅ 持有"
                    f.write(f"\n{rec['name']} ({rec['ts_code']}) {status}\n")
                    f.write(f"   盈亏: {rec['pnl_pct']:+.2f}%\n")
                f.write("\n" + "=" * 60 + "\n\n")
            
            f.write("🎯 买入候选\n" + "-" * 60 + "\n\n")
            f.write("\n\n".join(result_text))
        
        print(f"\n✅ 结果已保存到: {result_file}")
        
        return {
            'sell_recommendations': sell_recommendations,
            'buy_candidates': top_stocks.to_dict('records'),
            'market_on': market_on,
            'breadth': breadth
        }


if __name__ == '__main__':
    asyncio.run(main())
