# Data Pipeline Specification

## Purpose
数据管道负责从外部数据源获取股票行情、财务数据、市场情绪等信息，并进行清洗、转换、存储，为因子计算和模型训练提供高质量的数据。

## Requirements

### Requirement: Tushare Pro 数据获取
系统 SHALL 通过 Tushare Pro API 获取中国股市的历史和实时数据。

#### Scenario: 获取股票列表
- GIVEN 系统已初始化 Tushare Provider
- WHEN 请求获取股票列表
- THEN 返回所有上市股票的代码、名称、行业、上市日期等信息
- AND 支持按交易所（SSE/SZSE）过滤

#### Scenario: 获取日线行情
- GIVEN 股票代码和时间范围
- WHEN 请求日线行情数据
- THEN 返回包含 open, high, low, close, volume, amount, pct_chg 的数据
- AND 数据按日期升序排列

#### Scenario: 获取因子数据
- GIVEN 股票代码和时间范围
- WHEN 请求因子数据
- THEN 返回包含 PE、PB、PS、市值、换手率等基本面因子

#### Scenario: 获取资金流向
- GIVEN 股票代码和交易日期
- WHEN 请求资金流向数据
- THEN 返回大单、中单、小单的买卖量和金额

#### Scenario: 获取涨跌停列表
- GIVEN 交易日期
- WHEN 请求涨跌停列表
- THEN 返回当日所有涨停/跌停股票的详细信息
- AND 包含涨停时间、开板次数等信息

### Requirement: 异步数据获取
系统 SHALL 使用异步方式获取数据，提高并发性能。

#### Scenario: 并发获取多只股票数据
- GIVEN 10 只股票的代码列表
- WHEN 并发请求它们的日线数据
- THEN 所有请求在合理时间内完成
- AND 互不阻塞

#### Scenario: 错误重试机制
- GIVEN 数据请求失败（网络错误、API 限流等）
- WHEN 发生可恢复错误
- THEN 自动重试最多 3 次
- AND 采用指数退避策略

### Requirement: 数据存储
系统 SHALL 将获取的数据存储到数据库中，支持快速查询。

#### Scenario: 写入日线数据
- GIVEN 解析后的日线数据
- WHEN 写入数据库
- THEN 数据成功存储到 ohlcv 表
- AND 相同日期的相同股票数据会被覆盖（upsert）

#### Scenario: 查询历史数据
- GIVEN 股票代码和日期范围
- WHEN 查询历史行情
- THEN 返回指定时间范围内的所有数据
- AND 按日期排序

#### Scenario: 数据更新策略
- GIVEN 已有历史数据
- WHEN 执行每日数据同步
- THEN 只获取新增和更新的交易日数据
- AND 避免重复获取已有数据

### Requirement: 数据质量保证
系统 SHALL 确保存储的数据质量，包括去重、异常值处理。

#### Scenario: 数据去重
- GIVEN 包含重复记录的数据
- WHEN 写入数据库
- THEN 自动去重，同一股票同一日期只保留最新记录

#### Scenario: 异常值检测
- GIVEN 价格数据（开盘、最高、最低、收盘）
- WHEN 存在明显的异常值（如价格突变为 0 或负数）
- THEN 标记异常并跳过该条记录
- AND 记录异常日志

#### Scenario: 数据完整性校验
- GIVEN 从数据库读取的行情数据
- WHEN 发现缺失必要字段
- THEN 标记该记录为不完整
- AND 尝试重新获取

### Requirement: 市场情绪数据
系统 SHALL 收集市场层面的情绪数据，为策略提供宏观背景。

#### Scenario: 获取北向资金流向
- GIVEN 交易日期
- WHEN 请求北向资金数据
- THEN 返回当日北向资金的净流入/流出金额

#### Scenario: 获取市场宽基指标
- GIVEN 交易日期
- WHEN 请求市场统计数据
- THEN 返回涨跌家数、涨停跌停家数、成交额等

#### Scenario: 获取行业轮动数据
- GIVEN 交易日期
- WHEN 请求各行业涨跌幅
- THEN 返回主要行业的涨跌幅和资金流向

## Data Models

### OHLCV
```python
{
    "symbol": str,      # 股票代码
    "date": datetime,   # 交易日期
    "open": float,      # 开盘价
    "high": float,      # 最高价
    "low": float,       # 最低价
    "close": float,     # 收盘价
    "volume": float,    # 成交量
    "amount": float,    # 成交额
    "pct_chg": float,   # 涨跌幅
    "pre_close": float  # 前收盘价
}
```

### Factor Data
```python
{
    "symbol": str,
    "date": datetime,
    "pe": float,           # 市盈率
    "pb": float,           # 市净率
    "ps": float,           # 市销率
    "total_mv": float,     # 总市值
    "circ_mv": float,      # 流通市值
    "turnover_rate": float # 换手率
}
```

### Money Flow
```python
{
    "symbol": str,
    "date": datetime,
    "buy_large": float,   # 大单买入
    "sell_large": float,  # 大单卖出
    "buy_medium": float,  # 中单买入
    "sell_medium": float, # 中单卖出
    "buy_small": float,   # 小单买入
    "sell_small": float   # 小单卖出
}
```

## Performance Requirements

- 数据获取延迟：单只股票日线数据 < 500ms
- 并发能力：同时处理至少 50 只股票的请求
- 存储吞吐量：> 10000 条记录/秒
- 查询响应时间：单股票 1 年数据 < 100ms
