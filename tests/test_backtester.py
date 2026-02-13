"""
回测引擎单元测试

测试 Order、Position 等核心类
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquant.backtest.backtester import (
    OrderType,
    OrderSide,
    OrderStatus,
    Order,
    Position
)


class TestOrderType:
    """测试订单类型枚举"""

    def test_market_order_type(self):
        """测试市价单类型"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.MARKET.name == "MARKET"

    def test_limit_order_type(self):
        """测试限价单类型"""
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.LIMIT.name == "LIMIT"

    def test_stop_order_type(self):
        """测试止损单类型"""
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP.name == "STOP"


class TestOrderSide:
    """测试订单方向枚举"""

    def test_buy_side(self):
        """测试买入方向"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.BUY.name == "BUY"

    def test_sell_side(self):
        """测试卖出方向"""
        assert OrderSide.SELL.value == "sell"
        assert OrderSide.SELL.name == "SELL"


class TestOrderStatus:
    """测试订单状态枚举"""

    def test_pending_status(self):
        """测试待处理状态"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PENDING.name == "PENDING"

    def test_filled_status(self):
        """测试已成交状态"""
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.FILLED.name == "FILLED"

    def test_cancelled_status(self):
        """测试已取消状态"""
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.CANCELLED.name == "CANCELLED"


class TestOrder:
    """测试订单类"""

    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            id="order_001",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None
        )

        assert order.id == "order_001"
        assert order.symbol == "600000.SH"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.price is None
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.filled_price == 0.0
        assert order.commission == 0.0
        assert isinstance(order.timestamp, datetime)

    def test_limit_order_creation(self):
        """测试限价单创建"""
        order = Order(
            id="order_002",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=10.0
        )

        assert order.id == "order_002"
        assert order.price == 10.0
        assert order.status == OrderStatus.PENDING

    def test_order_status_update(self):
        """测试订单状态更新"""
        order = Order(
            id="order_003",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None
        )

        # 更新为已成交
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100.0
        order.filled_price = 10.0
        order.commission = 1.0

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100.0
        assert order.filled_price == 10.0
        assert order.commission == 1.0

    def test_order_partial_fill(self):
        """测试部分成交"""
        order = Order(
            id="order_004",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=200.0,
            price=10.0
        )

        # 部分成交
        order.status = OrderStatus.PARTIAL
        order.filled_quantity = 150.0
        order.filled_price = 10.0
        order.commission = 0.75

        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 150.0
        assert order.commission == 0.75

    def test_order_rejection(self):
        """测试订单拒绝"""
        order = Order(
            id="order_005",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None
        )

        # 拒绝订单
        order.status = OrderStatus.REJECTED

        assert order.status == OrderStatus.REJECTED

    def test_order_cancellation(self):
        """测试订单取消"""
        order = Order(
            id="order_006",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None
        )

        # 取消订单
        order.status = OrderStatus.CANCELLED

        assert order.status == OrderStatus.CANCELLED

    def test_order_commission_calculation(self):
        """测试佣金计算"""
        order = Order(
            id="order_007",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            price=10.0
        )

        # 计算佣金（假设千分之三）
        order.commission = order.quantity * order.price * 0.003

        assert order.commission == 30.0


class TestPosition:
    """测试持仓类"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        assert position.symbol == "600000.SH"
        assert position.quantity == 100.0
        assert position.entry_price == 10.0
        assert isinstance(position.entry_time, datetime)
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0

    def test_position_market_value(self):
        """测试持仓市值计算"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        market_value = position.market_value
        expected_value = 100.0 * 10.0

        assert market_value == expected_value

    def test_position_unrealized_pnl_long(self):
        """测试多头持仓未实现盈亏"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        # 当前价格上涨
        current_price = 12.0
        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity

        expected_pnl = (12.0 - 10.0) * 100.0
        assert position.unrealized_pnl == expected_pnl

    def test_position_unrealized_pnl_short(self):
        """测试空头持仓未实现盈亏"""
        position = Position(
            symbol="600000.SH",
            quantity=-100.0,  # 空头
            entry_price=10.0,
            entry_time=datetime.now()
        )

        # 当前价格上涨
        current_price = 12.0
        position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

        expected_pnl = (10.0 - 12.0) * 100.0
        assert position.unrealized_pnl == expected_pnl

    def test_position_realized_pnl(self):
        """测试已实现盈亏"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        # 部分平仓
        exit_price = 12.0
        exit_quantity = 50.0
        realized_pnl = (exit_price - position.entry_price) * exit_quantity

        position.realized_pnl = realized_pnl

        assert position.realized_pnl == realized_pnl

    def test_position_pnl_percent(self):
        """测试盈亏百分比"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        # 设置未实现盈亏
        position.unrealized_pnl = 200.0  # 盈利

        pnl_percent = position.pnl_percent
        expected_percent = 200.0 / (10.0 * 100.0) * 100

        assert pnl_percent == expected_percent

    def test_position_pnl_percent_loss(self):
        """测试亏损百分比"""
        position = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        # 设置未实现盈亏
        position.unrealized_pnl = -100.0  # 亏损

        pnl_percent = position.pnl_percent
        expected_percent = -100.0 / (10.0 * 100.0) * 100

        assert pnl_percent == expected_percent

    def test_multiple_positions(self):
        """测试多个持仓"""
        position1 = Position(
            symbol="600000.SH",
            quantity=100.0,
            entry_price=10.0,
            entry_time=datetime.now()
        )

        position2 = Position(
            symbol="600001.SH",
            quantity=200.0,
            entry_price=20.0,
            entry_time=datetime.now()
        )

        # 计算总市值
        total_value = position1.market_value + position2.market_value
        expected_value = 100.0 * 10.0 + 200.0 * 20.0

        assert total_value == expected_value
