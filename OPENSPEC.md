# AlphaGPT 中国股市量化系统改进计划

## 项目概述

基于 AlphaGPT 的符号回归因子挖掘架构，开发一套支持中国股市的高性能量化交易系统。

### 改进目标
1. **支持中国股市** - A股、港股、北交所
2. **数据源优化** - 使用 Tushare Pro、AkShare 等
3. **因子扩展** - 添加适合中国市场的技术指标
4. **实盘接口** - 支持各大券商接口（OpenCTP、东财等）
5. **性能优化** - 高并发、低延迟
6. **可视化面板** - 实时监控和回测分析

---

## 设计文档

### 1. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     AlphaQuant 中国版                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  DataPipe   │  │  ModelCore  │  │   Trader    │       │
│  │   数据管道   │  │  模型核心    │  │   交易执行   │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │               │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐       │
│  │ Tushare Pro │  │ AlphaGPT   │  │  OpenCTP    │       │
│  │  AkShare    │  │ FactorGen  │  │  券商API    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2. 模块规划

#### 2.1 数据管道 (DataPipeline)
- `data_providers/tushare.py` - Tushare Pro 数据接口
- `data_providers/akshare.py` - AkShare 数据接口
- `data_providers/ctp.py` - CTP 实时行情
- `data_loader/stock_loader.py` - 股票数据加载器
- `storage/mongodb.py` - MongoDB 存储替代 Postgres

#### 2.2 模型核心 (ModelCore)
- `factors/china_factors.py` - 中国市场特定因子
- `factors/fundamental.py` - 基本面因子
- `model/alpha_quant.py` - 改进的 AlphaGPT 模型
- `backtest/backtester.py` - 回测引擎
- `optimization/pso.py` - 粒子群优化策略参数

#### 2.3 策略管理 (StrategyManager)
- `signals/signal_generator.py` - 信号生成器
- `portfolio/portfolio_cn.py` - 沪深持仓管理
- `risk/risk_cn.py` - 中国市场风控（涨跌停、T+1等）

#### 2.4 交易执行 (Execution)
- `brokers/openctp.py` - OpenCTP 接口
- `brokers/simulate.py` - 模拟交易
- `order/order_manager.py` - 订单管理
- `slippage/slippage_model.py` - 滑点模型

#### 2.5 可视化 (Dashboard)
- `streamlit/dashboard_cn.py` - Streamlit 面板
- `visualizer/charts.py` - K线图、指标图
- `analysis/analytics.py` - 收益分析

### 3. 中国市场特殊处理

#### 3.1 交易规则
- **T+1 交易** - 当日买入次日才能卖出
- **涨跌停限制** - 10% 主板、20% 创业板/科创板、30% 北交所
- **熔断机制** - 暂停交易
- **融券限制** - 创业板/科创板不能融券做空

#### 3.2 时间安排
- **交易时间**：9:30-11:30, 13:00-15:00
- **集合竞价**：9:15-9:25（可撤单）、9:25-9:30（不可撤单）
- **盘后交易**：15:05-15:30（仅科创板）

#### 3.3 特殊因子
- 北向资金流向
- 融资融券余额
- 股东人数变化
- 大宗交易
- 龙虎榜数据

---

## 开发日志

### 2026-01-31
- [x] 分析 AlphaGPT 原始代码
- [x] 理解符号回归因子挖掘架构
- [x] 创建 OpenSpec 文档
- [x] 设计中国股市数据接口
- [x] 实现基础因子工程
- [x] 开发回测引擎
- [x] 开发 AlphaQuant 模型
- [x] 创建初始化脚本
- [x] 创建运行示例
- [ ] 实现策略管理器
- [ ] 实现交易执行模块
- [ ] 开发可视化面板
- [ ] 编写训练脚本
- [ ] 单元测试

---

## 参考资料

### 开源量化项目
- [Qlib](https://github.com/microsoft/qlib) - 微软 AI 量化平台
- [Backtrader](https://github.com/mementum/backtrader) - Python 回测框架
- [VN.PY](https://github.com/vnpy/vnpy) - 量化交易平台
- [QuantAxis](https://github.com/yutiansut/quantaxis) - 量化金融框架

### 中国股市数据源
- [Tushare Pro](https://tushare.pro/) - 专业财经数据接口
- [AkShare](https://akshare.akfamily.xyz/) - 开源财经数据接口
- [Efinance](https://efinance.pydata.org/) - 东方财富接口

### 交易接口
- [OpenCTP](https://github.com/ctpbee/ctpbee) - CTP 接口
- [XtQuant](https://www.thinktrader.net/) - 迅投量化接口
- [SimNow](https://www.simnow.com.cn/) - CTP 模拟交易

---

## 待办事项

### Phase 1: 基础设施 (Week 1)
- [ ] 搭建项目框架
- [ ] 实现数据管道（Tushare Pro）
- [ ] 实现 MongoDB 存储
- [ ] 编写单元测试

### Phase 2: 因子工程 (Week 2)
- [ ] 实现基础技术因子
- [ ] 实现中国市场特定因子
- [ ] 因子有效性分析
- [ ] 因子组合优化

### Phase 3: 模型训练 (Week 3)
- [ ] 改进 AlphaGPT 模型
- [ ] 训练因子生成器
- [ ] 策略参数优化
- [ ] 回测验证

### Phase 4: 实盘对接 (Week 4)
- [ ] 接入券商接口
- [ ] 实现订单管理
- [ ] 风控系统
- [ ] 监控面板

---

## 更新历史
- 2026-01-31: 项目初始化，创建 OpenSpec 文档
