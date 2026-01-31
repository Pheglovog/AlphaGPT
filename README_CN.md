# AlphaQuant 中国股市量化系统

基于 AlphaGPT 架构开发的 A 股量化交易系统，支持因子自动挖掘、回测验证和实盘交易。

## 项目结构

```
AlphaGPT/
├── alphaquant/                    # AlphaQuant 核心
│   ├── data_providers/           # 数据提供者
│   │   └── tushare.py          # Tushare Pro 接口
│   ├── factors/                   # 因子引擎
│   │   └── china_factors.py    # 中国市场因子
│   ├── model/                     # 模型核心
│   │   └── alpha_quant.py     # AlphaQuant 模型
│   ├── backtest/                  # 回测引擎
│   │   └── backtester.py      # 回测引擎
│   ├── strategy/                  # 策略管理
│   ├── execution/                 # 交易执行
│   ├── dashboard/                 # 可视化面板
│   └── config/                   # 配置文件
├── OPENSPEC.md                    # 开发文档
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据源

编辑 `config/config.yaml`：

```yaml
data:
  tushare_token: "your_tushare_token_here"
  enable_akshare: false

model:
  d_model: 128
  nhead: 8
  num_layers: 4
  dropout: 0.1

backtest:
  initial_capital: 1000000.0
  commission_rate: 0.0003
  slippage: true
```

### 3. 运行回测

```bash
python alphaquant/backtest/backtester.py
```

### 4. 训练模型

```bash
python alphaquant/model/train.py
```

## 核心特性

### 🎯 因子挖掘

基于符号回归自动生成可解释的因子公式：

- **基础因子** (6维): 对数收益、买卖压力、成交量加速度等
- **高级因子** (18维): RSI、MACD、布林带、KDJ、北向资金流向等
- **中国市场特定因子**: 涨跌停统计、融资融券余额、龙虎榜数据

### 📊 回测引擎

支持中国市场交易规则的回测：

- T+1 交易限制
- 涨跌停机制（主板10%、创业板20%、科创板20%、北交所30%）
- 交易成本（佣金、印花税、过户费）
- 滑点模型

### 🤖 AlphaQuant 模型

改进的 Transformer 模型：

- QK-Norm 注意力机制
- SwiGLU 激活函数
- 多任务学习（收益、夏普、回撤）
- 市场情绪编码

### 📈 策略执行

支持实盘交易：

- OpenCTP 接口
- 券商 API（东财、国泰君安等）
- 自动风控（止损、止盈、仓位管理）
- 实时监控面板

## 数据源

### Tushare Pro
- 股票列表、日线行情
- 财务数据、指标数据
- 北向资金、龙虎榜
- 融资融券数据

### AkShare
- 实时行情
- 新闻舆情
- 大股东数据
- 机构调研

## 使用示例

### 数据获取

```python
from alphaquant.data_providers.tushare import TushareProProvider

async with TushareProProvider(token) as provider:
    # 获取股票列表
    stocks = await provider.get_stock_list()

    # 获取日线行情
    daily = await provider.get_daily_quotes(
        ts_code='600519.SH',
        start_date='20240101',
        end_date='20241231'
    )
```

### 因子计算

```python
from alphaquant.factors.china_factors import ChinaFactorEngine

engine = ChinaFactorEngine()

# 计算基础因子
basic = engine.compute_basic_factors(raw_data)

# 计算高级因子
advanced = engine.compute_advanced_factors(raw_data)
```

### 回测

```python
from alphaquant.backtest.backtester import BacktestEngine

engine = BacktestEngine(initial_capital=1000000.0)
engine.add_data('600519.SH', data)

results = engine.run(strategy, start_date, end_date)
print(f"收益: {results['total_return']:.2f}%")
print(f"夏普: {results['sharpe_ratio']:.2f}")
```

### 模型生成因子公式

```python
from alphaquant.model.alpha_quant import AlphaQuant

model = AlphaQuant()

# 生成因子公式
formulas = model.generate_formula(
    factor_features,
    market_sentiment,
    max_length=20
)

print(f"生成的公式: {formulas[0]}")
```

## 开发文档

详细开发文档请查看 [OPENSPEC.md](OPENSPEC.md)

## 参考项目

- [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) - 原始架构
- [Qlib](https://github.com/microsoft/qlib) - 微软 AI 量化平台
- [Backtrader](https://github.com/mementum/backtrader) - 回测框架
- [VN.PY](https://github.com/vnpy/vnpy) - 量化交易平台

## License

MIT License

---

**注意**: 本系统仅供学习研究使用，实盘交易需谨慎！
