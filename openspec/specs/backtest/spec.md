# Backtest Engine Specification

## Purpose
回测引擎负责在历史数据上模拟交易策略的执行，考虑中国市场的交易规则（T+1、涨跌停、交易成本等），并提供详细的回测统计结果。

## Requirements

### Requirement: 中国市场交易规则
系统 SHALL 实现中国股市特有的交易规则。

#### Scenario: T+1 交易限制
- GIVEN 策略在当天买入股票
- WHEN 策略在同一天尝试卖出该股票
- THEN 拒绝该卖出订单
- AND 订单状态设置为 REJECTED

#### Scenario: 涨跌停限制
- GIVEN 股票价格达到涨停板
- WHEN 策略尝试买入
- THEN 订单无法成交
- AND 订单保持 PENDING 状态

#### Scenario: 跌停限制
- GIVEN 股票价格达到跌停板
- WHEN 策略尝试卖出
- THEN 订单无法成交
- AND 订单保持 PENDING 状态

#### Scenario: 涨跌停计算
- GIVEN 股票代码
- WHEN 计算涨跌停价格
- THEN 根据股票类型使用不同限制：
  - 主板、中小板：10%
  - 创业板：20%
  - 科创板：20%
  - 北交所：30%
  - ST 股：5%

#### Scenario: ST 股识别
- GIVEN 股票代码或名称
- WHEN 识别 ST 状态
- THEN 检查代码中是否包含 "ST" 或 "st"

### Requirement: 交易成本计算
系统 SHALL 计算每笔交易的实际成本。

#### Scenario: 计算佣金
- GIVEN 成交金额和佣金率（万三）
- WHEN 计算佣金
- THEN 返回 max(成交金额 * 0.0003, 5.0)
- AND 最低 5 元

#### Scenario: 计算印花税
- GIVEN 成交金额
- WHEN 计算印花税
- THEN 仅卖出时收取 0.1%
- AND 买入不收印花税

#### Scenario: 计算过户费
- GIVEN 成交金额
- WHEN 计算过户费
- THEN 返回成交金额 * 0.0001
- AND 双向收取

#### Scenario: 计算总交易成本
- GIVEN 买入或卖出操作
- WHEN 计算总成本
- THEN 返回 佣金 + 印花税 + 过户费

### Requirement: 滑点模型
系统 SHALL 模拟实际交易中的滑点影响。

#### Scenario: 计算市价单滑点
- GIVEN 基准价格、订单方向、成交量
- WHEN 计算滑点后的成交价格
- THEN 买入价格 = 价格 * (1 + 滑点率)
- AND 卖出价格 = 价格 * (1 - 滑点率)

#### Scenario: 大单滑点惩罚
- GIVEN 订单量和平均成交量
- WHEN 订单量占比很大
- THEN 增加额外滑点
- AND 滑点 = 基础滑点 + (订单量 / 平均成交量) * 0.005

#### Scenario: 限价单滑点
- GIVEN 限价单
- WHEN 涨跌停范围内有匹配价格
- THEN 使用限价成交，无滑点
- OR 否则不成交

### Requirement: 持仓管理
系统 SHALL 管理交易账户的持仓状态。

#### Scenario: 开仓
- GIVEN 买入订单成交
- WHEN 创建新持仓
- THEN 记录股票代码、数量、成本价、开仓时间

#### Scenario: 加仓
- GIVEN 已有持仓
- WHEN 买入同一股票
- THEN 更新持仓数量
- AND 重新计算平均成本：(原数量*原成本 + 新数量*新价格) / (原数量 + 新数量)

#### Scenario: 减仓
- GIVEN 持仓数量 > 卖出数量
- WHEN 卖出部分持仓
- THEN 更新持仓数量
- AND 计算已实现盈亏

#### Scenario: 平仓
- GIVEN 卖出数量 >= 持仓数量
- WHEN 卖出所有持仓
- THEN 删除持仓记录
- AND 计算总盈亏

#### Scenario: 更新持仓市值
- GIVEN 持仓和最新价格
- WHEN 价格变化
- THEN 更新持仓的市值和未实现盈亏

### Requirement: 订单管理
系统 SHALL 管理订单的完整生命周期。

#### Scenario: 创建订单
- GIVEN 策略信号
- WHEN 创建订单
- THEN 记录订单 ID、股票、方向、类型、数量
- AND 初始状态为 PENDING

#### Scenario: 执行订单
- GIVEN PENDING 订单且满足成交条件
- WHEN 执行订单
- THEN 计算滑点和交易成本
- AND 更新持仓或账户余额
- AND 记录成交记录

#### Scenario: 订单部分成交
- GIVEN 订单数量无法全部成交
- WHEN 执行部分数量
- THEN 更新订单的已成交数量
- AND 订单状态变为 PARTIAL

#### Scenario: 撤销订单
- GIVEN PENDING 订单
- WHEN 撤销订单
- THEN 订单状态变为 CANCELLED
- AND 不执行任何交易

#### Scenario: 订单拒绝
- GIVEN 违反交易规则（如 T+1）
- WHEN 尝试执行订单
- THEN 订单状态变为 REJECTED
- AND 不执行任何交易

### Requirement: 回测统计
系统 SHALL 计算并返回详细的回测统计指标。

#### Scenario: 计算总收益
- GIVEN 初始资金和最终资金
- WHEN 计算总收益
- THEN 返回 (最终资金 / 初始资金 - 1) * 100%

#### Scenario: 计算年化收益
- GIVEN 总收益率和回测天数
- WHEN 计算年化收益
- THEN 返回 (1 + 总收益) ^ (365 / 天数) - 1

#### Scenario: 计算夏普比率
- GIVEN 收益率序列
- WHEN 计算夏普比率
- THEN 返回 mean(收益) / std(收益) * sqrt(252)

#### Scenario: 计算最大回撤
- GIVEN 权益曲线
- WHEN 计算最大回撤
- THEN 返回 (权益 / 历史最高权益 - 1) 的最小值
- AND 乘以 100 得到百分比

#### Scenario: 计算胜率
- GIVEN 所有成交记录
- WHEN 计算胜率
- THEN 返回 盈利交易次数 / 总交易次数 * 100%

#### Scenario: 计算波动率
- GIVEN 收益率序列
- WHEN 计算年化波动率
- THEN 返回 std(收益率) * sqrt(252)

### Requirement: 可视化输出
系统 SHALL 生成回测结果的可视化图表。

#### Scenario: 生成权益曲线
- GIVEN 每日权益数据
- WHEN 生成权益曲线图
- THEN 绘制总权益随时间变化的折线图
- AND 标注最大回撤

#### Scenario: 生成收益分布
- GIVEN 单笔交易收益
- WHEN 生成收益分布图
- THEN 绘制收益的直方图
- AND 标注平均收益

#### Scenario: 生成月度收益热图
- GIVEN 日收益序列
- WHEN 生成月度收益热图
- THEN 按年和月聚合收益
- AND 用颜色表示收益正负

## Data Models

### Order
```python
{
    "id": str,              # 订单 ID
    "symbol": str,          # 股票代码
    "side": "buy" | "sell", # 订单方向
    "type": "market" | "limit", # 订单类型
    "quantity": float,       # 数量
    "price": float,         # 限价（限价单）
    "status": str,          # PENDING | FILLED | PARTIAL | CANCELLED | REJECTED
    "filled_quantity": float, # 已成交数量
    "filled_price": float,   # 成交价格
    "timestamp": datetime,   # 下单时间
    "commission": float      # 佣金
}
```

### Position
```python
{
    "symbol": str,           # 股票代码
    "quantity": float,        # 持仓数量
    "entry_price": float,     # 开仓均价
    "entry_time": datetime,   # 开仓时间
    "unrealized_pnl": float,  # 未实现盈亏
    "realized_pnl": float     # 已实现盈亏
}
```

### BacktestResult
```python
{
    "initial_capital": float,      # 初始资金
    "final_equity": float,       # 最终资金
    "total_return": float,       # 总收益 (%)
    "annual_return": float,      # 年化收益 (%)
    "sharpe_ratio": float,      # 夏普比率
    "max_drawdown": float,      # 最大回撤 (%)
    "volatility": float,        # 年化波动率
    "win_rate": float,          # 胜率 (%)
    "num_trades": int,         # 交易次数
    "equity_curve": DataFrame  # 权益曲线
}
```

## Trading Rules

### 涨跌停限制
| 市场类型 | 限制 | 示例代码 |
|---------|------|----------|
| 主板 | 10% | 600xxx.SH, 000xxx.SZ |
| 中小板 | 10% | 002xxx.SZ |
| 创业板 | 20% | 300xxx.SZ |
| 科创板 | 20% | 688xxx.SH |
| 北交所 | 30% | 8xxxxx.BJ |
| ST 股 | 5% | STxxxxx |

### 交易时间
- 上午：9:30 - 11:30
- 下午：13:00 - 15:00

### 交易费用
| 费用类型 | 费率 | 说明 |
|---------|------|------|
| 佣金 | 万三（0.03%） | 最低 5 元 |
| 印花税 | 千一（0.1%） | 仅卖出收取 |
| 过户费 | 万一（0.01%） | 双向收取 |

## Performance Requirements

- 回测速度：处理 1 年数据，10 只股票 < 10 秒
- 内存占用：< 500MB
- 支持：至少 100 只股票并行回测
