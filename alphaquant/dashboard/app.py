"""
可视化面板
使用 Streamlit 展示回测结果和策略表现
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any
import asyncio


def plot_equity_curve(equity_curve: pd.DataFrame) -> go.Figure:
    """
    绘制权益曲线

    Args:
        equity_curve: 权益曲线 DataFrame

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # 权益曲线
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['equity'],
        mode='lines',
        name='权益',
        line=dict(color='blue', width=2)
    ))

    # 最大回撤标记
    max_dd_idx = equity_curve['drawdown'].idxmin()
    if pd.notna(max_dd_idx):
        fig.add_trace(go.Scatter(
            x=[max_dd_idx, max_dd_idx],
            y=[equity_curve['equity'].loc[max_dd_idx], equity_curve['equity'].max()],
            mode='lines',
            name='最大回撤',
            line=dict(color='red', width=1, dash='dash')
        ))

    fig.update_layout(
        title='权益曲线',
        xaxis_title='日期',
        yaxis_title='权益',
        hovermode='x unified',
        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly'
    )

    return fig


def plot_drawdown(equity_curve: pd.DataFrame) -> go.Figure:
    """
    绘制回撤曲线

    Args:
        equity_curve: 权益曲线 DataFrame

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # 回撤区域
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['drawdown'],
        fill='tozeroy',
        mode='lines',
        name='回撤',
        line=dict(color='red', width=1)
    ))

    fig.update_layout(
        title='回撤曲线',
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        yaxis_tickformat='.2%',
        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly'
    )

    return fig


def plot_monthly_returns(returns: pd.Series) -> go.Figure:
    """
    绘制月度收益热图

    Args:
        returns: 日收益率 Series

    Returns:
        Plotly Figure
    """
    # 按年月聚合
    monthly = returns.groupby([returns.index.year, returns.index.month]).sum() * 100
    monthly = monthly.unstack()

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=monthly.columns,
        y=monthly.index,
        colorscale='RdYlGn',
        colorbar=dict(title='收益率 (%)')
    ))

    fig.update_layout(
        title='月度收益热图',
        xaxis_title='月份',
        yaxis_title='年份',
        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly'
    )

    return fig


def plot_returns_distribution(returns: pd.Series) -> go.Figure:
    """
    绘制收益分布

    Args:
        returns: 日收益率 Series

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # 直方图
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='收益分布',
        marker_color='blue',
        opacity=0.7
    ))

    # 平均收益线
    mean_return = returns.mean() * 100
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"平均: {mean_return:.2f}%"
    )

    fig.update_layout(
        title='收益分布',
        xaxis_title='日收益率 (%)',
        yaxis_title='频次',
        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly'
    )

    return fig


def plot_position_summary(positions: list) -> go.Figure:
    """
    绘制持仓汇总

    Args:
        positions: 持仓列表

    Returns:
        Plotly Figure
    """
    if not positions:
        return go.Figure()

    df = pd.DataFrame(positions)

    fig = go.Figure(data=[go.Pie(
        labels=df['symbol'],
        values=df['value'],
        hole=0.3,
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title='持仓分布',
        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly'
    )

    return fig


def display_metrics(metrics: Dict[str, float]):
    """
    显示关键指标

    Args:
        metrics: 指标字典
    """
    st.subheader("📊 回测指标")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="总收益率",
            value=f"{metrics.get('total_return', 0):.2f}%",
            delta=f"{metrics.get('total_return', 0):.2f}%"
        )

    with col2:
        st.metric(
            label="年化收益",
            value=f"{metrics.get('annual_return', 0):.2f}%",
            delta=f"{metrics.get('annual_return', 0):.2f}%"
        )

    with col3:
        st.metric(
            label="夏普比率",
            value=f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta="> 1 为优秀"
        )

    with col4:
        st.metric(
            label="最大回撤",
            value=f"{metrics.get('max_drawdown', 0):.2f}%",
            delta=f"{metrics.get('max_drawdown', 0):.2f}%"
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric(
            label="胜率",
            value=f"{metrics.get('win_rate', 0):.2f}%"
        )

    with col6:
        st.metric(
            label="盈亏比",
            value=f"{metrics.get('profit_loss_ratio', 0):.2f}"
        )

    with col7:
        st.metric(
            label="交易次数",
            value=f"{metrics.get('num_trades', 0)}"
        )


def display_trade_list(trades: pd.DataFrame):
    """
    显示交易列表

    Args:
        trades: 交易记录 DataFrame
    """
    st.subheader("📋 交易记录")

    if trades.empty:
        st.info("暂无交易记录")
        return

    # 格式化列
    display_cols = trades.copy()
    display_cols['pnl'] = display_cols['pnl'].round(2)
    display_cols['pnl_pct'] = (display_cols['pnl_pct'] * 100).round(2)

    # 颜色标记
    def color_pnl(val):
        color = '#90EE90' if val > 0 else '#FFB6C1'
        return f'background-color: {color}'

    styled = display_cols.style.applymap(color_pnl, subset=['pnl', 'pnl_pct'])

    st.dataframe(
        styled,
        use_container_width=True,
        height=400
    )


def display_factor_performance(factor_scores: pd.DataFrame):
    """
    显示因子表现

    Args:
        factor_scores: 因子分数 DataFrame
    """
    st.subheader("📈 因子表现")

    # 分组统计
    stats = factor_scores.groupby('factor_name').agg({
        'score': ['mean', 'std', 'count']
    })

    stats.columns = ['平均分数', '标准差', '样本数']

    st.dataframe(stats)


# Streamlit 主应用
def main():
    """主应用"""
    st.set_page_config(
        page_title="AlphaQuant Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 AlphaQuant 回测面板")

    # 侧边栏
    st.sidebar.header("配置")

    # 加载数据
    uploaded_file = st.sidebar.file_uploader(
        "上传回测结果 (CSV)",
        type=['csv']
    )

    if uploaded_file is None:
        st.info("请上传回测结果 CSV 文件")
        st.write("""
        CSV 文件应包含以下列：
        - date: 日期
        - equity: 权益
        - drawdown: 回撤
        - return: 日收益率
        - """)
        return

    # 读取数据
    results = pd.read_csv(uploaded_file, parse_dates=['date'])
    results.set_index('date', inplace=True)

    # 计算指标
    total_return = (results['equity'].iloc[-1] / results['equity'].iloc[0] - 1) * 100
    days = len(results)
    annual_return = total_return * 252 / days
    sharpe_ratio = results['return'].mean() / results['return'].std() * np.sqrt(252)
    max_drawdown = results['drawdown'].min()

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': 0.55,  # 示例
        'profit_loss_ratio': 1.5,  # 示例
        'num_trades': 42  # 示例
    }

    # 显示指标
    display_metrics(metrics)

    # 图表
    tab1, tab2, tab3, tab4 = st.tabs(["权益曲线", "回撤", "收益分布", "交易记录"])

    with tab1:
        fig_equity = plot_equity_curve(results)
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab2:
        fig_drawdown = plot_drawdown(results)
        st.plotly_chart(fig_drawdown, use_container_width=True)

        fig_monthly = plot_monthly_returns(results['return'])
        st.plotly_chart(fig_monthly, use_container_width=True)

    with tab3:
        fig_dist = plot_returns_distribution(results['return'])
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        # 模拟交易记录
        trades = pd.DataFrame({
            'date': pd.date_range(results.index[0], results.index[-1], freq='10D')[:42],
            'symbol': ['600519.SH'] * 42,
            'side': ['BUY', 'SELL'] * 21,
            'quantity': [100] * 42,
            'price': [1700 + np.random.randn() * 50 for _ in range(42)],
            'pnl': np.random.randn(42) * 5000,
            'pnl_pct': np.random.randn(42) * 0.02
        })

        display_trade_list(trades)


if __name__ == "__main__":
    main()
