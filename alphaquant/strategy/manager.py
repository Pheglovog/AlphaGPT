"""
策略管理器
管理交易策略、信号生成、持仓管理、风险控制
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger


class Signal(Enum):
    """交易信号"""
    BUY = 1
    SELL = -1
    HOLD = 0


class StrategyConfig:
    """策略配置"""

    def __init__(
        self,
        buy_threshold: float = 0.7,
        sell_threshold: float = 0.3,
        max_positions: int = 5,
        position_size: float = 0.2,
        stop_loss: float = 0.05,
        take_profit: float = 0.20,
        trailing_stop: float = 0.08
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop


class Position:
    """持仓信息"""

    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.highest_price = entry_price  # 用于移动止损

    @property
    def value(self, current_price: float) -> float:
        """当前市值"""
        return self.quantity * current_price

    @property
    def unrealized_pnl(self, current_price: float) -> float:
        """未实现盈亏"""
        return (current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """未实现盈亏百分比"""
        return (current_price / self.entry_price - 1) * 100


class StrategyManager:
    """策略管理器"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.cash: float = 0
        self.initial_capital: float = 0

    def initialize(self, initial_capital: float):
        """初始化账户"""
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions.clear()
        logger.info(f"Strategy initialized with capital: {initial_capital:,.2f}")

    def generate_signals(
        self,
        scores: pd.DataFrame,
        current_prices: pd.Series
    ) -> Dict[str, Signal]:
        """
        根据模型分数生成交易信号

        Args:
            scores: 股票分数 DataFrame {symbol: score}
            current_prices: 当前价格 Series {symbol: price}

        Returns:
            {symbol: Signal} 字典
        """
        signals = {}

        for symbol in scores.index:
            score = scores.loc[symbol]

            # 买入信号
            if score >= self.config.buy_threshold:
                # 检查是否已持仓
                if symbol not in self.positions:
                    signals[symbol] = Signal.BUY

            # 卖出信号
            elif score <= self.config.sell_threshold:
                if symbol in self.positions:
                    signals[symbol] = Signal.SELL

        return signals

    def check_stop_take(self, current_prices: pd.Series) -> Dict[str, Signal]:
        """
        检查止损止盈

        Args:
            current_prices: 当前价格 Series {symbol: price}

        Returns:
            {symbol: Signal} 字典
        """
        signals = {}

        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.entry_price)

            # 更新最高价（用于移动止损）
            position.highest_price = max(position.highest_price, current_price)

            # 移动止损
            trailing_stop_price = position.highest_price * (1 - self.config.trailing_stop)

            # 固定止损
            stop_loss_price = position.entry_price * (1 - self.config.stop_loss)

            # 止盈价格
            take_profit_price = position.entry_price * (1 + self.config.take_profit)

            # 检查止损
            if current_price <= min(stop_loss_price, trailing_stop_price):
                signals[symbol] = Signal.SELL
                logger.debug(f"{symbol} 触发止损: {current_price:.2f} < {stop_loss_price:.2f}")

            # 检查止盈
            elif current_price >= take_profit_price:
                signals[symbol] = Signal.SELL
                logger.debug(f"{symbol} 触发止盈: {current_price:.2f} >= {take_profit_price:.2f}")

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        total_capital: float
    ) -> float:
        """
        计算仓位大小

        Args:
            signal: 交易信号
            current_price: 当前价格
            total_capital: 总资产

        Returns:
            仓位大小（股数）
        """
        if signal != Signal.BUY:
            return 0

        # 目标仓位价值
        target_value = total_capital * self.config.position_size

        # 计算股数
        quantity = target_value / current_price

        # 取整（100股为1手）
        quantity = int(quantity / 100) * 100

        # 确保不超过剩余资金
        max_quantity = int(self.cash / current_price / 100) * 100
        quantity = min(quantity, max_quantity)

        return quantity

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        quantity: float,
        price: float,
        timestamp: datetime
    ) -> Optional[float]:
        """
        执行交易信号

        Args:
            symbol: 股票代码
            signal: 交易信号
            quantity: 数量
            price: 价格
            timestamp: 时间戳

        Returns:
            交易成本
        """
        # 交易成本（万三佣金 + 印花税）
        commission = max(price * quantity * 0.0003, 5.0)  # 最低5元
        stamp_duty = price * quantity * 0.001 if signal == Signal.SELL else 0
        transfer_fee = price * quantity * 0.0001

        total_cost = commission + stamp_duty + transfer_fee

        if signal == Signal.BUY:
            # 检查资金
            total_amount = price * quantity + total_cost
            if total_amount > self.cash:
                logger.warning(f"Insufficient cash for {symbol}: need {total_amount:.2f}, have {self.cash:.2f}")
                return None

            # 扣除资金
            self.cash -= total_amount

            # 创建持仓
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                stop_loss=price * (1 - self.config.stop_loss),
                take_profit=price * (1 + self.config.take_profit)
            )
            self.positions[symbol] = position

            logger.info(
                f"BUY {symbol} {quantity} @ {price:.2f}, "
                f"cost: {total_cost:.2f}, cash: {self.cash:.2f}"
            )

        elif signal == Signal.SELL:
            if symbol not in self.positions:
                logger.warning(f"No position for {symbol}")
                return None

            position = self.positions.pop(symbol)

            # 增加资金
            self.cash += price * quantity - total_cost

            # 记录盈亏
            pnl = (price - position.entry_price) * quantity
            logger.info(
                f"SELL {symbol} {quantity} @ {price:.2f}, "
                f"pnl: {pnl:.2f}, cash: {self.cash:.2f}"
            )

        return total_cost

    @property
    def total_value(self) -> float:
        """总资产"""
        return self.cash

    @property
    def equity(self) -> float:
        """权益（现金 + 持仓市值）"""
        equity = self.cash
        # 持仓市值需要传入当前价格，这里暂时只返回现金
        return equity

    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """获取持仓汇总"""
        summary = []
        for symbol, position in self.positions.items():
            summary.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'value': position.quantity * position.entry_price,  # 使用成本价
            })
        return summary


class BacktestStrategy:
    """回测策略基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.manager = StrategyManager(config)

    def on_data(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Signal]:
        """
        收到新数据时的回调

        Args:
            data: 包含股票分数和价格的 DataFrame
            timestamp: 当前时间

        Returns:
            {symbol: Signal} 字典
        """
        raise NotImplementedError

    def on_bar(self, bar_data: Dict[str, pd.DataFrame]):
        """
        每个K线回调

        Args:
            bar_data: {symbol: DataFrame} 字典
        """
        raise NotImplementedError


class FactorStrategy(BacktestStrategy):
    """基于因子分数的策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback_days = 20
        self.factor_scores: Optional[pd.DataFrame] = None

    def on_data(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Signal]:
        """
        根据因子分数生成信号

        Args:
            data: DataFrame columns=['symbol', 'factor_score', 'price']
            timestamp: 当前时间

        Returns:
            {symbol: Signal} 字典
        """
        # 保存因子分数
        scores = data.set_index('symbol')['factor_score']
        prices = data.set_index('symbol')['price']

        # 生成信号
        signals = self.manager.generate_signals(scores, prices)

        # 检查止损止盈
        stop_signals = self.manager.check_stop_take(prices)
        signals.update(stop_signals)

        return signals

    def on_bar(self, bar_data: Dict[str, pd.DataFrame]):
        """
        处理每个K线

        Args:
            bar_data: {symbol: DataFrame} 字典
        """
        # 计算因子分数（示例）
        scores = {}

        for symbol, df in bar_data.items():
            if len(df) < 5:
                continue

            # 简单动量因子
            recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)

            # 标准化到 0-1
            scores[symbol] = max(0, min(1, (recent_return + 0.1) / 0.2))

        # 创建DataFrame
        data = pd.DataFrame({
            'symbol': list(scores.keys()),
            'factor_score': list(scores.values()),
            'price': [bar_data[s]['close'].iloc[-1] for s in scores.keys()]
        })

        return data
