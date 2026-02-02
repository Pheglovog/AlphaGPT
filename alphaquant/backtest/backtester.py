"""
中国股市回测引擎
考虑 T+1、涨跌停、交易成本等中国市场特性

参考：
- Backtrader 回测框架
- Qlib 回测引擎
- 中国市场交易规则
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"       # 市价单
    LIMIT = "limit"         # 限价单
    STOP = "stop"           # 止损单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    timestamp: datetime = None
    commission: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.entry_price

    @property
    def pnl_percent(self) -> float:
        """盈亏百分比"""
        if self.quantity == 0:
            return 0.0
        return self.unrealized_pnl / (self.quantity * self.entry_price) * 100


@dataclass
class Trade:
    """成交记录"""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float


class MarketRules:
    """中国市场交易规则"""

    # 涨跌停限制
    LIMIT_UP_RATES = {
        'main': 0.10,      # 主板 10%
        'star': 0.20,      # 创业板 20%
        'tech': 0.20,      # 科创板 20%
        'bj': 0.30,        # 北交所 30%
        'st': 0.05         # ST 股 5%
    }

    # 交易时间
    TRADING_HOURS = [
        (9, 30, 11, 30),  # 上午
        (13, 0, 15, 0),    # 下午
    ]

    # 交易费用
    COMMISSION_RATE = 0.0003  # 万三佣金
    STAMP_DUTY_RATE = 0.001   # 千一印花税（仅卖出）
    MIN_COMMISSION = 5.0       # 最低佣金 5 元

    # T+1 交易
    T_PLUS_ONE = True

    @staticmethod
    def get_limit_rate(symbol: str) -> float:
        """获取涨跌停限制"""
        if 'ST' in symbol or 'st' in symbol:
            return MarketRules.LIMIT_UP_RATES['st']
        elif symbol.startswith('688') or symbol.startswith('300'):
            return MarketRules.LIMIT_UP_RATES['tech']
        elif symbol.startswith('8'):
            return MarketRules.LIMIT_UP_RATES['bj']
        elif symbol.startswith('30'):
            return MarketRules.LIMIT_UP_RATES['star']
        else:
            return MarketRules.LIMIT_UP_RATES['main']

    @staticmethod
    def calculate_commission(amount: float, is_sell: bool = False) -> float:
        """
        计算交易费用

        Args:
            amount: 成交金额
            is_sell: 是否卖出（印花税）

        Returns:
            总费用
        """
        commission = max(amount * MarketRules.COMMISSION_RATE, MarketRules.MIN_COMMISSION)
        stamp_duty = amount * MarketRules.STAMP_DUTY_RATE if is_sell else 0
        transfer_fee = amount * 0.00001  # 过户费

        return commission + stamp_duty + transfer_fee

    @staticmethod
    def is_trading_time(dt: datetime) -> bool:
        """判断是否为交易时间"""
        hour, minute = dt.hour, dt.minute
        time_val = hour * 60 + minute

        for h1, m1, h2, m2 in MarketRules.TRADING_HOURS:
            start = h1 * 60 + m1
            end = h2 * 60 + m2
            if start <= time_val <= end:
                return True

        return False


class SlippageModel:
    """滑点模型"""

    @staticmethod
    def calculate_slippage(
        price: float,
        side: OrderSide,
        volume: float,
        avg_volume: float,
        order_type: OrderType = OrderType.MARKET
    ) -> float:
        """
        计算滑点

        Args:
            price: 基准价格
            side: 订单方向
            volume: 订单量
            avg_volume: 平均成交量
            order_type: 订单类型

        Returns:
            实际成交价格
        """
        # 交易占比
        volume_ratio = volume / (avg_volume + 1e-6)

        # 基础滑点（市价单更高）
        base_slippage = 0.001 if order_type == OrderType.MARKET else 0.0003

        # 大单滑点增加
        large_order_penalty = volume_ratio * 0.005

        total_slippage = base_slippage + large_order_penalty

        if side == OrderSide.BUY:
            return price * (1 + total_slippage)
        else:
            return price * (1 - total_slippage)


class BacktestEngine:
    """
    回测引擎
    考虑 T+1、涨跌停、交易成本、滑点
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage_model: Optional[SlippageModel] = None
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage_model: 滑点模型
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model or SlippageModel()

        # 持仓
        self.positions: Dict[str, Position] = {}

        # 订单
        self.orders: List[Order] = []
        self.pending_orders: List[Order] = []

        # 成交记录
        self.trades: List[Trade] = []

        # 账户记录
        self.equity_curve: List[Dict] = []

        # 数据缓存
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def add_data(self, symbol: str, data: pd.DataFrame):
        """
        添加历史数据

        Args:
            symbol: 股票代码
            data: OHLCV 数据，索引为日期
        """
        self.data_cache[symbol] = data.copy()

    def run(
        self,
        strategy: Callable[[str, pd.DataFrame], List[Order]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        运行回测

        Args:
            strategy: 策略函数，输入为 symbol 和数据，返回订单列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果字典
        """
        logger.info(f"开始回测: {start_date.date()} ~ {end_date.date()}")

        # 生成交易日期
        all_dates = []
        for df in self.data_cache.values():
            dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
            all_dates.extend(dates.tolist())
        all_dates = sorted(set(all_dates))

        for date in all_dates:
            # 检查是否为交易日
            if not MarketRules.is_trading_time(date):
                continue

            # 更新持仓市值
            self._update_positions(date)

            # 处理订单
            self._process_orders(date)

            # 生成策略信号
            for symbol, data in self.data_cache.items():
                # 获取到当前日期的数据
                history = data[data.index <= date]
                if len(history) < 2:
                    continue

                # 调用策略
                new_orders = strategy(symbol, history)
                self.pending_orders.extend(new_orders)

            # 记录账户
            self._record_equity(date)

        logger.info("回测完成")

        # 计算统计指标
        return self._calculate_stats()

    def _process_orders(self, date: datetime):
        """处理订单"""
        for order in self.pending_orders[:]:
            if not MarketRules.is_trading_time(date):
                continue

            symbol = order.symbol
            if symbol not in self.data_cache:
                order.status = OrderStatus.REJECTED
                self.pending_orders.remove(order)
                continue

            data = self.data_cache[symbol]
            if date not in data.index:
                continue

            # 获取当日数据
            row = data.loc[date]
            high, low, close, volume = row['high'], row['low'], row['close'], row['volume']

            # T+1 限制：当日买入的不能卖出
            if order.side == OrderSide.SELL:
                position = self.positions.get(symbol)
                if not position or position.quantity <= 0:
                    order.status = OrderStatus.REJECTED
                    self.pending_orders.remove(order)
                    continue

                # 检查是否达到 T+1
                if MarketRules.T_PLUS_ONE:
                    days_held = (date - position.entry_time).days
                    if days_held < 1:
                        continue

            # 涨跌停检查
            limit_rate = MarketRules.get_limit_rate(symbol)
            pre_close = row.get('pre_close', close)
            limit_up = pre_close * (1 + limit_rate)
            limit_down = pre_close * (1 - limit_rate)

            if order.side == OrderSide.BUY and close >= limit_up * 0.999:
                # 涨停板，无法买入
                continue
            if order.side == OrderSide.SELL and close <= limit_down * 1.001:
                # 跌停板，无法卖出
                continue

            # 执行订单
            if order.order_type == OrderType.MARKET:
                execution_price = self.slippage_model.calculate_slippage(
                    close, order.side, order.quantity, volume
                )

                # 检查是否在涨跌停范围内
                if execution_price > limit_up * 1.005 or execution_price < limit_down * 0.995:
                    continue

                # 计算成交
                self._execute_order(order, execution_price, date)

                self.pending_orders.remove(order)
            elif order.order_type == OrderType.LIMIT:
                # 限价单逻辑（简化）
                if order.side == OrderSide.BUY and low <= order.price <= high:
                    self._execute_order(order, order.price, date)
                    self.pending_orders.remove(order)
                elif order.side == OrderSide.SELL and low <= order.price <= high:
                    self._execute_order(order, order.price, date)
                    self.pending_orders.remove(order)

    def _execute_order(self, order: Order, price: float, date: datetime):
        """执行订单"""
        amount = order.quantity * price
        commission = MarketRules.calculate_commission(amount, order.side == OrderSide.SELL)

        if order.side == OrderSide.BUY:
            # 检查资金是否充足
            if amount + commission > self.capital:
                order.status = OrderStatus.REJECTED
                return

            self.capital -= amount + commission

            # 更新持仓
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                # 加权平均成本
                total_cost = pos.quantity * pos.entry_price + order.quantity * price
                pos.quantity += order.quantity
                pos.entry_price = total_cost / pos.quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=price,
                    entry_time=date
                )

        else:  # SELL
            position = self.positions.get(order.symbol)
            if not position:
                order.status = OrderStatus.REJECTED
                return

            # 更新持仓
            realized_pnl = (price - position.entry_price) * order.quantity
            position.quantity -= order.quantity
            position.realized_pnl += realized_pnl

            # 卖出所得
            self.capital += amount - commission

            # 清空持仓
            if position.quantity <= 0:
                del self.positions[order.symbol]

        # 记录成交
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=date,
            commission=commission
        )
        self.trades.append(trade)

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = price
        order.commission = commission
        self.orders.append(order)

    def _update_positions(self, date: datetime):
        """更新持仓市值"""
        for symbol, pos in self.positions.items():
            if symbol not in self.data_cache:
                continue

            data = self.data_cache[symbol]
            if date not in data.index:
                continue

            current_price = data.loc[date, 'close']
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity

    def _record_equity(self, date: datetime):
        """记录账户权益"""
        market_value = sum(pos.market_value for pos in self.positions.values())
        total_equity = self.capital + market_value

        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'market_value': market_value,
            'total_equity': total_equity,
            'positions': len(self.positions)
        })

    def _calculate_stats(self) -> Dict:
        """计算回测统计"""
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)

        # 基础指标
        final_equity = equity_df['total_equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 收益率序列
        returns = equity_df['total_equity'].pct_change().dropna()

        # 夏普比率（年化）
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min() * 100

        # 胜率
        winning_trades = [t for t in self.trades if t.side == OrderSide.SELL]
        win_rate = len([t for t in winning_trades
                        if self._get_trade_pnl(t) > 0]) / len(winning_trades) * 100 if winning_trades else 0

        # 交易次数
        num_trades = len(self.trades)

        stats = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': total_return,  # 简化计算
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'equity_curve': equity_df
        }

        logger.info(f"回测结果: 收益={total_return:.2f}%, 夏普={sharpe_ratio:.2f}, "
                   f"最大回撤={max_drawdown:.2f}%, 胜率={win_rate:.1f}%")

        return stats

    def _get_trade_pnl(self, trade: Trade) -> float:
        """获取交易盈亏"""
        for pos in self.positions.values():
            if pos.symbol == trade.symbol:
                return pos.realized_pnl
        return 0.0


def simple_strategy_example(symbol: str, data: pd.DataFrame) -> List[Order]:
    """
    简单的策略示例：均线交叉

    修复 Bug：确保数据至少有 21 行（prev 需要 -2）
    """
    from uuid import uuid4

    orders = []

    # 修复：检查数据行数
    if len(data) < 21:  # 需要至少 21 行（latest + prev）
        return orders

    # 计算 MA 并添加到数据
    data = data.copy()
    data['ma5'] = data['close'].rolling(5).mean()
    data['ma20'] = data['close'].rolling(20).mean()

    latest = data.iloc[-1]
    prev = data.iloc[-2]  # 修复：现在数据至少有 21 行，-2 是安全的

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


def backtest_example():
    """回测示例"""
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=1000000.0)

    # 添加模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    n = len(dates)
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n)) * 0.5,
        'high': 100 + np.cumsum(np.random.randn(n)) * 0.5 + 1,
        'low': 100 + np.cumsum(np.random.randn(n)) * 0.5 - 1,
        'close': 100 + np.cumsum(np.random.randn(n)) * 0.5,
        'volume': np.random.randint(100000, 1000000, n),
        'pre_close': 100
    }, index=dates)

    data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n))
    data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n))

    engine.add_data('600519.SH', data)

    # 运行回测
    results = engine.run(
        strategy=simple_strategy_example,
        start_date=dates[0],
        end_date=dates[-1]
    )

    print("回测统计:")
    for key, value in results.items():
        if key != 'equity_curve':
            print(f"{key}: {value}")


if __name__ == "__main__":
    backtest_example()
