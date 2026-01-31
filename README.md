# AlphaQuant - 中国股市量化交易系统

基于 AlphaGPT 符号回归架构开发的中国股市量化交易系统，支持因子挖掘、回测模拟、模型训练和策略执行。

## ✨ 核心特性

### 📊 数据管道
- ✅ **Tushare Pro 异步接口** - 支持并发控制和自动重试
- ✅ **数据缓存** - MongoDB 本地存储
- ✅ **市场数据** - 股票行情、因子数据、资金流向、涨跌停列表

### 🔬 因子引擎
- ✅ **24维因子空间** - 6维基础 + 18维高级因子
- ✅ **中国市场因子** - 北向资金、融资融券、涨跌停统计
- ✅ **技术指标** - RSI、MACD、布林带、ATR、KDJ 等
- ✅ **批量计算** - 向量化处理，高效计算

### 📈 回测引擎
- ✅ **T+1 交易规则** - 符合中国股市规则
- ✅ **涨跌停限制** - 主板10%、创业板20%、科创板20%、北交所30%
- ✅ **交易成本** - 佣金、印花税、过户费
- ✅ **滑点模型** - 大单滑点惩罚
- ✅ **回测统计** - 夏普比率、最大回撤、胜率、盈亏比

### 🤖 AlphaQuant 模型
- ✅ **QK-Norm 注意力** - 更稳定的训练
- ✅ **SwiGLU 前馈** - 更好的性能
- ✅ **市场情绪编码** - 宽基、行业、资金流向
- ✅ **多任务学习** - 收益、夏普、回撤同时预测
- ✅ **符号回归** - 生成可解释的因子公式

### 🎯 策略管理
- ✅ **信号生成** - 基于因子分数自动生成买卖信号
- ✅ **持仓管理** - 开仓、加仓、减仓、平仓
- ✅ **风险控制** - 止损、止盈、移动止损
- ✅ **仓位计算** - 根据风险和资金自动计算仓位

### 📊 可视化面板
- ✅ **权益曲线** - 资金增长趋势
- ✅ **回撤分析** - 最大回撤和时间
- ✅ **收益分布** - 日收益率直方图
- ✅ **月度热图** - 按年月的收益分布
- ✅ **交易记录** - 详细的交易历史

---

## 🛠️ 环境要求

### 必需环境

| 组件 | 版本要求 | 说明 |
|-----|---------|------|
| Python | 3.9+ | 运行环境 |
| pip | 最新 | 包管理器 |
| Git | 任意 | 版本控制 |

### 可选环境

| 组件 | 版本要求 | 用途 |
|-----|---------|------|
| CUDA | 11.8+ | GPU 加速（可选） |
| MongoDB | 4.4+ | 数据缓存（可选） |
| Streamlit | 1.28+ | 可视化面板（开发中）|

---

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/Pheglovog/AlphaGPT.git
cd AlphaGPT
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 使用安装脚本
python init_alphaquant.py

# 或手动安装
pip install torch numpy pandas aiohttp python-dotenv loguru pyyaml

# 可选：Streamlit 面板
pip install streamlit plotly

# 可选：MongoDB 驱动
pip install pymongo
```

### 4. 配置环境变量

编辑 `.env` 文件：

```bash
# Tushare Pro 配置
TUSHARE_TOKEN=你的Tushare_Pro_Token
TUSHARE_CONCURRENT_LIMIT=5  # 2000积分=5, 5000积分=10

# MongoDB（可选）
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=alphaquant

# 模型参数
MODEL_D_MODEL=128
MODEL_NHEAD=8
MODEL_NUM_LAYERS=4

# 回测参数
BACKTEST_INITIAL_CAPITAL=1000000
BACKTEST_COMMISSION_RATE=0.0003

# 策略参数
STRATEGY_MAX_POSITIONS=5
STRATEGY_BUY_THRESHOLD=0.7
STRATEGY_SELL_THRESHOLD=0.3
```

### 5. 测试连接

```bash
python test_tushare.py
```

预期输出：
```
✅ 成功获取 2303 只上交所股票
✅ 成功获取 22 条上证指数数据
✅ 成功获取 22 条贵州茅台数据
✅ Tushare stats: 4 requests in 0.1s (31.80 req/s)
```

---

## 🚀 快速开始

### 1. 运行示例

```bash
python run_examples.py
```

这将演示：
- 因子计算
- 模型推理
- 回测引擎

### 2. 因子计算

```python
from alphaquant.factors.china_factors import ChinaFactorEngine
import torch

# 创建因子引擎
engine = ChinaFactorEngine()

# 准备数据
raw_data = {
    'close': torch.rand(5, 60) * 10 + 100,
    'open': torch.rand(5, 60) * 10 + 100,
    'high': torch.rand(5, 60) * 10 + 105,
    'low': torch.rand(5, 60) * 10 + 95,
    'volume': torch.rand(5, 60) * 1000000,
}

# 计算因子
basic = engine.compute_basic_factors(raw_data)  # [5, 6, 60]
advanced = engine.compute_advanced_factors(raw_data)  # [5, 18, 60]
```

### 3. 策略回测

```python
from alphaquant.backtest.backtester import BacktestEngine

# 创建回测引擎
engine = BacktestEngine(initial_capital=100000.0)

# 添加数据
data = pd.read_csv('stock_data.csv')
engine.add_data('600519.SH', data)

# 定义策略
def my_strategy(symbol, history):
    orders = []
    # 你的策略逻辑
    return orders

# 运行回测
results = engine.run(my_strategy, '2024-01-01', '2024-12-31')

# 查看结果
print(f"总收益: {results['total_return']:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2f}%")
```

### 4. 模型训练

```bash
python train_model.py --epochs 100 --batch-size 32 --lr 1e-4
```

### 5. 启动可视化面板（开发中）

```bash
streamlit run alphaquant/dashboard/app.py
```

---

## 📋 项目结构

```
AlphaGPT/
├── alphaquant/                 # 核心模块
│   ├── data_providers/        # 数据提供者
│   │   └── tushare.py     # Tushare Pro 接口
│   ├── factors/              # 因子引擎
│   │   └── china_factors.py # 中国市场因子
│   ├── model/                # 模型
│   │   └── alpha_quant.py  # AlphaQuant 模型
│   ├── backtest/             # 回测
│   │   └── backtester.py  # 回测引擎
│   ├── strategy/             # 策略
│   │   └── manager.py      # 策略管理器
│   ├── dashboard/            # 面板
│   │   └── app.py         # Streamlit 应用
│   └── config/              # 配置
│       └── config.yaml      # 配置文件
│
├── openspec/                 # OpenSpec 规范
│   ├── specs/              # 功能规范
│   └── changes/            # 变更记录
│
├── init_alphaquant.py        # 初始化脚本
├── run_examples.py          # 运行示例
├── test_tushare.py         # 连接测试
├── train_model.py          # 训练脚本
├── .env                    # 环境变量
├── venv/                  # 虚拟环境
│
├── OPENSPEC.md             # OpenSpec 工作流程
├── README_CN.md            # 中文文档
├── SUMMARY.md              # 项目总结
└── TUSHARE_429_SOLUTION.md # 429 错误解决方案
```

---

## 📚 文档

| 文档 | 说明 |
|-----|------|
| [OPENSPEC.md](OPENSPEC.md) | OpenSpec 工作流程 |
| [README_CN.md](README_CN.md) | 中文使用文档 |
| [SUMMARY.md](SUMMARY.md) | 项目总结 |
| [TUSHARE_429_SOLUTION.md](TUSHARE_429_SOLUTION.md) | 429 错误解决方案 |
| [openspec/specs/](openspec/specs/) | 功能规范文档 |

---

## 🔑 Tushare Pro 配置

### 获取 Token

1. 访问 https://tushare.pro
2. 注册/登录
3. 进入「用户中心」→「API Token」
4. 复制你的 Token

### 积分等级

| 等级 | 费用 | 并发 | 每分钟频次 | 适用场景 |
|-----|------|------|-----------|---------|
| 免费 | 0元 | 2 | 50次 | 学习测试 |
| 2000积分 | 200元/年 | 5 | 200次 | 日常投资 |
| 5000积分 | 500元/年 | 10 | 500次 | 量化研究 |

### 免费获取积分

1. 关注"挖地兔"公众号 + 小红书
2. 加入 Tushare QQ 群（1036369385）
3. @群主："已关注公众号和小红书"
4. 免费获得 200 积分

---

## 🎯 使用场景

### 场景 1: 学习量化

```bash
# 运行示例，了解系统架构
python run_examples.py

# 研究因子，理解市场特征
python -c "from alphaquant.factors.china_factors import ChinaFactorEngine; ..."
```

### 场景 2: 策略回测

```python
# 编写你的策略
def my_strategy(symbol, history):
    # 基于因子的策略
    return orders

# 回测验证
from alphaquant.backtest.backtester import BacktestEngine
engine = BacktestEngine()
results = engine.run(my_strategy, ...)
```

### 场景 3: 因子挖掘

```python
# 使用 AlphaQuant 模型自动生成因子公式
from alphaquant.model.alpha_quant import AlphaQuant
model = AlphaQuant(config)
formulas = model.generate_formula(features, sentiment)
```

### 场景 4: 实盘模拟

```python
# 使用策略管理器模拟实盘
from alphaquant.strategy.manager import StrategyManager
manager = StrategyManager(config)
# ... 每日更新持仓
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 License

MIT

---

## 🙏 致谢

- [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) - 原始架构
- [Tushare Pro](https://tushare.pro) - 数据支持
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**开发者**: Pheglovog
**最后更新**: 2026-01-31
