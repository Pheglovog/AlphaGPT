# AlphaQuant 中国股市量化系统 - 项目总结

## 项目概况

基于 AlphaGPT 符号回归架构开发的中国股市量化交易系统。

**项目状态**: ✅ 核心框架已完成

---

## 已完成模块

### 1. 数据管道 (DataPipeline)

**文件**: `alphaquant/data_providers/tushare.py`

**功能**:
- Tushare Pro 异步数据接口
- 股票列表获取
- 日线行情获取
- 因子数据（市值、市盈率等）
- 资金流向数据
- 涨跌停列表
- 指数行情

**API**:
```python
async with TushareProProvider(token) as provider:
    stocks = await provider.get_stock_list()
    daily = await provider.get_daily_quotes(ts_code='600519.SH', ...)
    factors = await provider.get_stock_factor(...)
    moneyflow = await provider.get_money_flow(...)
    limits = await provider.get_limit_list(...)
    index = await provider.get_index_daily(...)
```

---

### 2. 因子引擎 (FactorEngine)

**文件**: `alphaquant/factors/china_factors.py`

**功能**:

#### 基础因子 (6维)
- `RET`: 对数收益
- `PRESSURE`: 买卖压力
- `FOMO`: 成交量加速度
- `DEV`: 价格偏离
- `VOL`: 对数成交量
- `AMP`: 振幅

#### 高级因子 (18维)
- **技术指标**: RSI, MACD, 布林带, ATR, KDJ, VPT, OBV
- **动量因子**: 5/10/20日动量
- **位置因子**: 价格相对位置
- **波动率**: 波动率聚集
- **中国市场**: 北向资金, 融资融券, 涨跌停统计

**API**:
```python
engine = ChinaFactorEngine()

# 基础因子
basic = engine.compute_basic_factors(raw_data)  # [B, 6, T]

# 高级因子
advanced = engine.compute_advanced_factors(raw_data)  # [B, 18, T]
```

---

### 3. 回测引擎 (BacktestEngine)

**文件**: `alphaquant/backtest/backtester.py`

**功能**:
- 支持 T+1 交易
- 涨跌停限制（主板10%、创业板20%、科创板20%、北交所30%）
- 交易成本（佣金、印花税、过户费）
- 滑点模型
- 多头持仓管理
- 回测统计（收益、夏普、回撤、胜率）

**交易规则**:
- 市价单、限价单支持
- 订单状态管理
- 持仓盈亏跟踪
- 权益曲线记录

**API**:
```python
engine = BacktestEngine(initial_capital=1000000.0)
engine.add_data('600519.SH', data)

results = engine.run(strategy, start_date, end_date)

# 结果
{
    'total_return': 15.5,        # 总收益%
    'sharpe_ratio': 1.2,        # 夏普比率
    'max_drawdown': -8.3,       # 最大回撤%
    'win_rate': 55.0,           # 胜率%
    'num_trades': 42            # 交易次数
}
```

---

### 4. AlphaQuant 模型

**文件**: `alphaquant/model/alpha_quant.py`

**架构**:
- **QK-Norm Attention**: 查询键归一化
- **SwiGLU FFN**: Swish 门控线性单元
- **RMSNorm**: 根均方归一化
- **多任务学习**: 收益、夏普、回撤
- **市场情绪编码**: 宽基指数、行业轮动、资金流向

**生成算子 (16个)**:
- 算术: ADD, SUB, MUL, DIV, NEG, ABS, SIGN
- 聚合: MAX, MIN
- 条件: GATE
- 时序: DELAY1, DELAY5, SMA, EMA, STD

**词汇表** (50+):
- 6 个基础因子
- 18 个高级因子
- 16 个算子

**API**:
```python
model = AlphaQuant(config)

# 推理
output = model(factor_features, market_sentiment)
# {
#     'logits': [B, vocab_size],
#     'value': [B, 1],
#     'task_probs': [B, 3]
# }

# 生成公式
formulas = model.generate_formula(
    factor_features,
    market_sentiment,
    max_length=32,
    temperature=1.0
)
# ['RET', 'MUL', 'RSI', 'ADD', 'VOL', ...]
```

---

### 5. 配置系统

**文件**: `alphaquant/config/config.yaml`

**配置项**:
- 数据源（Tushare Token、MongoDB）
- 模型参数（维度、层数、学习率）
- 回测参数（初始资金、佣金、滑点）
- 策略参数（仓位、止损止盈）
- 风控参数（最大亏损、仓位限制）
- 交易执行（OpenCTP、券商 API）
- 日志配置
- 面板配置

---

### 6. 初始化脚本

**文件**: `init_alphaquant.py`

**功能**:
- 检查 Python 版本
- 安装依赖包
- 创建目录结构
- 生成配置文件
- 测试依赖导入

**使用**:
```bash
python init_alphaquant.py
```

---

### 7. 运行示例

**文件**: `run_examples.py`

**演示**:
1. 因子计算
2. 模型推理
3. 策略回测
4. 数据获取

**使用**:
```bash
python run_examples.py
```

---

## 待完成模块

### 1. 策略管理器
- 信号生成器
- 持仓管理
- 风控系统
- 组合优化

### 2. 交易执行
- OpenCTP 接口
- 券商 API 封装
- 订单管理
- 实时监控

### 3. 可视化面板
- Streamlit 面板
- K 线图表
- 回测结果展示
- 实时持仓监控

### 4. 训练脚本
- 数据准备
- 模型训练
- 回测奖励
- 模型评估

### 5. 单元测试
- 因子测试
- 模型测试
- 回测测试
- 集成测试

---

## 快速开始

### 1. 初始化
```bash
cd AlphaGPT
python init_alphaquant.py
```

### 2. 配置
```bash
nano .env
# 填入 TUSHARE_TOKEN
```

### 3. 运行示例
```bash
python run_examples.py
```

### 4. 自定义回测
```python
from alphaquant.backtest.backtester import BacktestEngine

# 创建策略
def my_strategy(symbol, data):
    # 你的策略逻辑
    return orders

# 运行回测
engine = BacktestEngine()
engine.add_data('600519.SH', data)
results = engine.run(my_strategy, start, end)
```

---

## 技术亮点

### 1. 中国市场适配
- ✅ T+1 交易规则
- ✅ 涨跌停机制
- ✅ 交易成本模型
- ✅ 中国市场因子

### 2. 高性能
- ✅ 异步数据获取
- ✅ 批量因子计算
- ✅ GPU 加速
- ✅ 向量化操作

### 3. 可扩展性
- ✅ 模块化设计
- ✅ 配置文件驱动
- ✅ 插件式架构
- ✅ 支持多数据源

### 4. 可解释性
- ✅ 符号回归公式
- ✅ 可执行因子表达式
- ✅ 清晰的信号来源

---

## 项目结构

```
AlphaGPT/
├── alphaquant/              # 核心
│   ├── data_providers/      # 数据提供者
│   │   └── tushare.py      # ✅ 完成
│   ├── factors/             # 因子引擎
│   │   └── china_factors.py # ✅ 完成
│   ├── model/               # 模型
│   │   └── alpha_quant.py  # ✅ 完成
│   ├── backtest/            # 回测
│   │   └── backtester.py   # ✅ 完成
│   ├── strategy/            # 策略 ⏳
│   ├── execution/           # 执行 ⏳
│   ├── dashboard/           # 面板 ⏳
│   └── config/             # 配置
│       └── config.yaml     # ✅ 完成
├── init_alphaquant.py      # ✅ 初始化脚本
├── run_examples.py         # ✅ 运行示例
├── OPENSPEC.md             # ✅ 开发文档
└── README_CN.md            # ✅ 使用文档
```

---

## 下一步计划

### Phase 1: 完善核心
1. 实现策略管理器
2. 实现交易执行模块
3. 开发可视化面板

### Phase 2: 训练与优化
1. 编写训练脚本
2. 准备训练数据
3. 调优模型参数

### Phase 3: 实盘对接
1. 对接券商 API
2. 模拟盘测试
3. 实盘验证

---

## 参考资料

- [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) - 原始架构
- [Qlib](https://github.com/microsoft/qlib) - 微软量化框架
- [Backtrader](https://github.com/mementum/backtrader) - 回测框架
- [Tushare Pro](https://tushare.pro) - 数据源

---

**最后更新**: 2026-01-31
**开发者**: 土哥 (Clawdbot AI Agent)
