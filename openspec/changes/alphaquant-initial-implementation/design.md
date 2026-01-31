## Context

AlphaGPT 是一个使用符号回归进行因子挖掘的量化系统，针对 Solana meme 代币市场。我们需要将这个架构适配到中国股市，具有完全不同的数据源、交易规则和市场环境。

### AlphaGPT 架构分析

```
数据层 → 因子层 → 模型层 → 策略层 → 执行层
   ↓        ↓        ↓        ↓        ↓
Birdeye  StackVM  Transformer  Runner   Jupiter
```

### 中国股市差异

| 维度 | AlphaGPT (Solana) | AlphaQuant (中国） |
|-----|------------------|----------------|
| 数据源 | Birdeye/DexScreener | Tushare Pro |
| 交易规则 | 24/7 交易 | T+1，涨跌停 |
| 交易成本 | Gas fee | 佣金 + 印花税 + 过户费 |
| 滑点 | DEX 滑点 | 大单滑点 |
| 市场环境 | Meme 币 | 主板/创业板/科创板 |
| 特殊因子 | FDV/Liquidity | 北向资金/融资融券 |

## Goals / Non-Goals

**Goals:**
- 创建完整的中国股市量化交易框架
- 保留 AlphaGPT 的符号回归核心理念
- 实现中国市场特有的交易规则
- 提供易于使用的 API 和示例
- 使用 OpenSpec 规范记录设计和实现

**Non-Goals:**
- 保持与 AlphaGPT 完全兼容（架构差异太大）
- 支持期货、期权等衍生品（先专注股票）
- 提供完整的实盘对接（先实现模拟盘）

## Decisions

### Decision 1: 异步数据获取

选择使用 aiohttp 而非同步的 requests 库。

**Rationale:**
- Tushare Pro API 支持 HTTP，aiohttp 提供异步支持
- 批量获取数据时，异步可以显著提高吞吐量
- Python 3.9+ 原生支持 async/await
- 符合现代 Python 开发实践

### Decision 2: 因子维度选择

选择 24 维因子空间（6 维基础 + 18 维高级）。

**Rationale:**
- 6 维基础因子与 AlphaGPT 保持一致，便于对比
- 18 维高级因子补充技术分析能力
- 中国市场特定因子（3 维）提供本土化优势
- 总计 24 维足够丰富，但不过于复杂

### Decision 3: T+1 实现方式

在订单层面强制执行 T+1 规则。

**Rationale:**
- 买入记录开仓时间
- 卖出时检查 (当前时间 - 开仓时间).days >= 1
- 不满足条件的订单直接 REJECTED
- 在订单管理层面处理，而不是策略层面

### Decision 4: 涨跌停计算

使用预收盘价计算涨跌停，支持不同市场类型。

**Rationale:**
- 预收盘价是最准确的基准
- 不同市场类型（主板、创业板等）有不同限制
- ST 股特殊处理（5% 限制）
- 支持动态调整（注册制可能调整限制）

### Decision 5: 模型架构改进

在 AlphaGPT 基础上添加 QK-Norm 和 SwiGLU。

**Rationale:**
- QK-Norm 提高注意力稳定性
- SwiGLU 在 NLP 任务中表现优于 ReLU
- RMSNorm 比 LayerNorm 更稳定
- 市场情绪编码提供宏观背景

### Decision 6: 多任务学习

预测三个目标：回测收益、夏普比率、最大回撤。

**Rationale:**
- 单目标优化可能过拟合
- 夏普比率考虑风险
- 最大回撤衡量风险承受能力
- 任务路由让模型自动选择最相关的目标

### Decision 7: 配置文件格式

使用 YAML 格式的配置文件。

**Rationale:**
- 人类可读性好
- 支持注释
- 易于版本控制
- 可以用环境变量覆盖

### Decision 8: 项目结构

将功能模块化到 alphaquant/ 目录下。

**Rationale:**
- 清晰的模块边界
- 易于单独测试和维护
- 支持未来扩展
- 符合 Python 项目最佳实践

## Architecture

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AlphaQuant                          │
├─────────────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  DataPipe   │  │  Factors    │  │    Model    │      │
│  │             │  │             │  │             │      │
│  │ Tushare Pro │  │  China      │  │ AlphaQuant  │      │
│  │ Async API   │  │ Engine      │  │ Transformer │      │
│  │ MongoDB     │  │  24 factors │  │ QK-Norm     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │  Backtest   │  │  Strategy   │                      │
│  │             │  │             │                      │
│  │ T+1 Rules  │  │  Signals    │                      │
│  │ 涨跌停     │  │  Portfolio  │                      │
│  │ 滑点模型    │  │  Risk Mgmt   │                      │
│  └─────────────┘  └─────────────┘                      │
│                                                          │
│  ┌─────────────┐                                         │
│  │ Execution   │  (开发中）                        │
│  │             │                                         │
│  │ OpenCTP     │                                         │
│  │ 券商 API    │                                         │
│  └─────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
Tushare Pro → Data Pipeline → MongoDB
                          ↓
                    Factor Engine
                          ↓
                    Features [B, 24, T]
                          ↓
                    AlphaQuant Model
                          ↓
                    Generated Formula
                          ↓
                    Strategy Manager
                          ↓
                      Signals
                          ↓
                    Backtest Engine
                          ↓
                    Orders
                          ↓
                    Filled Trades
                          ↓
                    Portfolio + PnL
                          ↓
                    Statistics
```

### 模块依赖

```
DataPipeline
├── TushareProProvider (独立）
└── DataManager (依赖 Provider)

FactorEngine
├── ChinaFactors (独立）
└── (可选）StockFactors

BacktestEngine
├── MarketRules (独立）
├── SlippageModel (独立）
├── OrderManager (内部）
└── PortfolioManager (内部）

AlphaQuant Model
├── FactorEmbedding (依赖 Feature 定义）
├── MarketSentimentEncoder (独立）
├── Transformer (独立）
└── MultiTaskHead (独立）
```

## Technical Details

### 异步设计

```python
async with TushareProProvider(token) as provider:
    # 并发获取多只股票
    tasks = [provider.get_daily_quotes(code) for code in codes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 因子计算流程

```python
# 1. 数据加载
raw_data = load_ohlcv_data(symbol)

# 2. 基础因子 (6 维）
basic_factors = ChinaFactorEngine.compute_basic_factors(raw_data)
# shape: [batch, 6, time_steps]

# 3. 高级因子 (18 维）
advanced_factors = ChinaFactorEngine.compute_advanced_factors(raw_data)
# shape: [batch, 18, time_steps]

# 4. 合并
all_factors = torch.cat([basic_factors, advanced_factors], dim=1)
# shape: [batch, 24, time_steps]
```

### 回测订单处理

```python
def execute_order(order, market_data):
    # 1. 检查 T+1
    if order.side == SELL and not check_t_plus_one(order.symbol):
        order.status = REJECTED
        return

    # 2. 检查涨跌停
    if is_limit_up_or_down(market_data):
        order.status = PENDING  # 无法成交
        return

    # 3. 计算滑点
    filled_price = calculate_slippage(order, market_data)

    # 4. 计算交易成本
    commission = calculate_commission(order, filled_price)

    # 5. 执行
    fill_order(order, filled_price, commission)
```

### 模型推理

```python
# 1. 编码特征
feat_emb = feature_embed(factor_features)

# 2. 添加市场情绪
sent_emb = sentiment_encoder(market_sentiment)
x = feat_emb + sent_emb

# 3. Transformer 编码
x = transformer(x, mask=causal_mask)

# 4. 多任务预测
logits, value, task_probs = model(x)

# 5. 采样生成
formula_tokens = sample(logits, temperature=1.0, top_k=10)
```

## Performance Considerations

### 数据获取性能
- 并发请求数：50-100（避免 API 限流）
- 超时时间：10 秒
- 重试策略：指数退避，最多 3 次
- 预期吞吐量：5000 股/分钟

### 因子计算性能
- 批量计算：使用 NumPy/Pandas 向量化
- GPU 加速：使用 PyTorch 计算（支持 CUDA）
- 预期速度：100 股 x 2 年数据 < 5 秒

### 回测性能
- 单股票回测：< 1 秒（2 年数据）
- 批量回测：50 股 < 30 秒
- 内存占用：< 500MB

### 模型性能
- 推理延迟：< 50ms（batch_size=1）
- 批量推理：< 2 秒（batch_size=100）
- 模型大小：< 100MB

## Security Considerations

### API Key 管理
- Tushare Token 存储在 .env 文件或环境变量
- 不要将 .env 提交到版本控制
- 支持 secrets.txt 替代方案

### 交易风险
- 默认使用模拟交易
- 实盘对接需要额外确认
- 提供风控参数限制最大亏损

## Future Extensions

### 短期（1-2 个月）
1. 完成策略管理器
2. 实现可视化面板（Streamlit）
3. 编写训练脚本
4. 添加单元测试

### 中期（3-6 个月）
1. 对接 OpenCTP（期货）
2. 对接券商 API（股票）
3. 支持更多数据源（AkShare、Efinance）
4. 添加回测分析工具

### 长期（6+ 个月）
1. 策略优化平台
2. 实盘监控系统
3. 策略回测竞赛
4. 社区策略分享
