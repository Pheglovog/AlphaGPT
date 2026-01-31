# Factor Engine Specification

## Purpose
因子引擎负责从原始行情数据中计算各类技术指标和量化因子，包括基础因子、高级因子和中国市场特定因子，为模型训练和策略信号提供特征输入。

## Requirements

### Requirement: 基础因子计算
系统 SHALL 计算基于 OHLCV 数据的 6 维基础因子。

#### Scenario: 计算对数收益
- GIVEN 价格序列
- WHEN 计算对数收益
- THEN 返回 log(close[t] / close[t-1])
- AND 处理除零错误

#### Scenario: 计算买卖压力
- GIVEN 开盘、最高、最低、收盘价
- WHEN 计算买卖压力
- THEN 返回 (close - open) / (high - low)
- AND 使用 tanh 归一化到 [-1, 1]

#### Scenario: 计算成交量加速度
- GIVEN 成交量序列
- WHEN 计算成交量变化率及其差分
- THEN 返回 vol_chg - lag(vol_chg)
- AND 限制在 [-5, 5] 范围内

#### Scenario: 计算价格偏离
- GIVEN 价格序列和窗口长度（默认 20）
- WHEN 计算价格相对于均值的偏离
- THEN 返回 (close - ma(close, 20)) / ma(close, 20)
- AND 使用移动平均填充前 n-1 个值

#### Scenario: 计算对数成交量
- GIVEN 成交量序列
- WHEN 计算对数成交量
- THEN 返回 log(1 + volume)

#### Scenario: 计算振幅因子
- GIVEN 最高价、最低价、收盘价
- WHEN 计算日内振幅
- THEN 返回 (high - low) / close

### Requirement: 高级因子计算
系统 SHALL 计算 18 维高级因子，包括传统技术指标和中国市场特定因子。

#### Scenario: 计算 RSI
- GIVEN 价格序列和周期（默认 14）
- WHEN 计算相对强弱指标
- THEN 返回 100 - 100 / (1 + RS)
- AND 其中 RS = 平均涨幅 / 平均跌幅

#### Scenario: 计算 MACD
- GIVEN 价格序列和快/慢/信号周期（12/26/9）
- WHEN 计算 MACD 指标
- THEN 返回 DIF、DEA、MACD 柱状图
- AND DIF = EMA(12) - EMA(26)

#### Scenario: 计算布林带位置
- GIVEN 价格序列和参数（20 周期，2 倍标准差）
- WHEN 计算布林带位置
- THEN 返回 (close - lower) / (upper - lower - 1e-6) - 0.5
- AND 归一化到 [-1, 1]

#### Scenario: 计算 ATR
- GIVEN 最高、最低、收盘价和周期（14）
- WHEN 计算平均真实波幅
- THEN 返回 TR 的移动平均
- AND TR = max(high-low, |high-pre_close|, |low-pre_close|)

#### Scenario: 计算 KDJ
- GIVEN 最高、最低、收盘价和参数
- WHEN 计算 K、D、J 值
- THEN 返回标准化到 [-1, 1] 的值

#### Scenario: 计算动量因子
- GIVEN 价格序列和多个周期（5/10/20 日）
- WHEN 计算多周期动量
- THEN 返回 (close[t-n] / close[t] - 1) * 100
- AND 支持多个周期

#### Scenario: 计算价格相对位置
- GIVEN 价格序列和窗口（5/10/20）
- WHEN 计算价格在窗口内的相对位置
- THEN 返回 (close - min(close[n:])) / (max(close[n:]) - min(close[n:]) - 1e-6) - 0.5

### Requirement: 中国市场特定因子
系统 SHALL 计算适应中国市场交易规则的特定因子。

#### Scenario: 北向资金流向因子
- GIVEN 北向资金历史数据
- WHEN 计算北向资金流向因子
- THEN 返回北向资金净流入的标准化值
- AND 使用 5 日均值作为基准

#### Scenario: 融资融券余额因子
- GIVEN 融资融券历史数据
- WHEN 计算融资余额变化率
- THEN 返回 (fin_balance[t] - fin_balance[t-1]) / fin_balance[t-1]
- AND 异常值限制在 [-5, 5]

#### Scenario: 涨跌停统计因子
- GIVEN 行情数据和窗口
- WHEN 统计涨跌停次数
- THEN 返回涨跌停频率
- AND 区分涨停和跌停

#### Scenario: 换手率因子
- GIVEN 成交量和流通股本
- WHEN 计算换手率
- THEN 返回 volume / float_share
- AND 单位为百分比

### Requirement: 因子标准化
系统 SHALL 对计算出的因子进行标准化，使其具有可比性。

#### Scenario: 鲁棒标准化
- GIVEN 因子序列
- WHEN 计算鲁棒标准化
- THEN 返回 (x - median(x)) / mad(x)
- AND mad = median(|x - median(x)|)

#### Scenario: 限制范围
- GIVEN 标准化后的因子
- WHEN 发现超出范围的值
- THEN 限制在 [-5, 5] 范围内
- AND 使用 clamp 操作

### Requirement: 批量计算
系统 SHALL 支持批量计算多只股票的因子，提高效率。

#### Scenario: 批量计算基础因子
- GIVEN 多只股票的 OHLCV 数据
- WHEN 批量计算基础因子
- THEN 使用向量化操作同时处理所有股票
- AND 返回 [batch_size, 6, time_steps] 的张量

#### Scenario: 批量计算高级因子
- GIVEN 多只股票的 OHLCV 数据
- WHEN 批量计算高级因子
- THEN 使用 NumPy/Pandas 向量化操作
- AND 支持并行处理

### Requirement: 因子验证
系统 SHALL 验证计算出的因子，确保其正确性。

#### Scenario: 检查缺失值
- GIVEN 计算后的因子
- WHEN 发现 NaN 或 Inf 值
- THEN 标记这些值为无效
- AND 记录警告日志

#### Scenario: 检查因子范围
- GIVEN 因子序列
- WHEN 发现异常值（如 > 100 或 < -100）
- THEN 标记为潜在错误
- AND 不用于模型训练

## Factor Definitions

### Basic Factors (6 维)
| 因子名称 | 描述 | 计算公式 | 归一化 |
|---------|------|----------|--------|
| RET | 对数收益 | log(close[t] / close[t-1]) | 鲁棒标准化 |
| PRESSURE | 买卖压力 | (close - open) / (high - low) | tanh * 3 |
| FOMO | 成交量加速度 | vol_chg - lag(vol_chg) | clamp [-5, 5] |
| DEV | 价格偏离 | (close - ma) / ma | 鲁棒标准化 |
| VOL | 对数成交量 | log(1 + volume) | 鲁棒标准化 |
| AMP | 振幅 | (high - low) / close | 鲁棒标准化 |

### Advanced Factors (18 维)
| 因子名称 | 描述 | 来源 |
|---------|------|------|
| RSI | 相对强弱指标 | 传统技术分析 |
| MACD | MACD 柱状图 | 传统技术分析 |
| BB_POS | 布林带位置 | 传统技术分析 |
| ATR | 平均真实波幅 | 传统技术分析 |
| K, J | KDJ 指标 | 传统技术分析 |
| VPT | 成交量价格趋势 | 量价分析 |
| OBV | 能量潮 | 量价分析 |
| VOL_CLUSTER | 波动率聚集 | 高级技术分析 |
| MOM_5/10/20 | 多周期动量 | 动量分析 |
| POS_5/10/20 | 价格相对位置 | 位置分析 |
| NORTH_FLOW | 北向资金流向 | 中国市场 |
| MARGIN | 融资融券 | 中国市场 |
| LIMIT_UP | 涨跌停统计 | 中国市场 |

## Performance Requirements

- 单股票因子计算：< 10ms
- 批量 100 股：< 1s
- 内存占用：< 1GB（处理 1000 股，2 年数据）
