## Why

基于 AlphaGPT 的符号回归因子挖掘架构，我们需要开发一套适配中国股市的量化交易系统。AlphaGPT 原始设计专注于 Solana meme 代币市场，无法直接应用于中国股市。中国股市有独特的交易规则（T+1、涨跌停）和市场环境（北向资金、融资融券），需要专门的适配。

## What Changes

- 创建 AlphaQuant 核心框架，包含数据管道、因子引擎、回测引擎、模型核心
- 实现 Tushare Pro 异步数据接口，支持中国股市数据获取
- 开发中国市场特定因子（北向资金、融资融券、涨跌停）
- 实现支持 T+1 和涨跌停的回测引擎
- 改进 AlphaGPT 模型架构，添加市场情绪编码和多任务学习
- 提供初始化脚本和运行示例，降低使用门槛

## Capabilities

### New Capabilities
- `TushareProProvider`: 异步 Tushare Pro 数据接口，支持股票列表、日线行情、因子数据、资金流向、涨跌停列表、指数数据
- `ChinaFactorEngine`: 中国市场因子引擎，计算 24 维因子（基础 6 维 + 高级 18 维）
- `BacktestEngine`: 中国市场回测引擎，支持 T+1 交易、涨跌停限制、交易成本、滑点模型
- `AlphaQuant`: 改进的 Transformer 模型，包含 QK-Norm、SwiGLU、市场情绪编码、多任务学习
- `MarketRules`: 中国市场交易规则封装（涨跌停率、交易时间、交易费用）

### Modified Capabilities
<!-- 无现有功能被修改 - 这是一个全新的独立模块 -->

## Impact

- `alphaquant/data_providers/tushare.py`: 新增 Tushare Pro 数据提供者
- `alphaquant/factors/china_factors.py`: 新增中国市场因子引擎
- `alphaquant/backtest/backtester.py`: 新增回测引擎
- `alphaquant/model/alpha_quant.py`: 新增 AlphaQuant 模型
- `alphaquant/config/config.yaml`: 新增配置文件
- `init_alphaquant.py`: 新增初始化脚本
- `run_examples.py`: 新增运行示例
- `openspec/`: 新增 OpenSpec 规范文档（specs/ 和 changes/）

## Dependencies

### 新增依赖
```
torch>=2.0.0              # PyTorch 深度学习
numpy>=1.24.0              # 数值计算
pandas>=2.0.0              # 数据处理
aiohttp>=3.9.0             # 异步 HTTP 客户端
loguru>=0.7.0               # 日志
pyyaml>=6.0                 # 配置文件解析
tushare>=1.2.60            # Tushare Pro 数据接口
```

### 可选依赖
```
pymongo>=4.6.0             # MongoDB 存储（可选）
streamlit>=1.28.0           # 可视化面板（开发中）
```

## Migration Notes

### 从 AlphaGPT 迁移

AlphaGPT 原始设计针对 Solana 市场，迁移到中国股市需要注意：

1. **数据源**: 从 Birdeye/Jupiter API 迁移到 Tushare Pro
2. **交易规则**: 添加 T+1、涨跌停限制
3. **因子扩展**: 添加中国市场特定因子（北向资金、融资融券）
4. **模型改进**: 添加市场情绪编码
5. **回测适配**: 支持中国交易成本和规则

### 兼容性

- AlphaQuant 独立于 AlphaGPT，不影响原有代码
- 可以同时运行两套系统
- 未来可考虑共享回测框架

## Risks

- **数据质量**: Tushare Pro 数据延迟可能影响策略效果
  → 缓解：实现数据缓存和降级方案
- **模型训练**: 需要大量历史数据训练模型
  → 缓解：使用 Tushare 免费数据，逐步积累
- **实盘对接**: 券商 API 限制可能导致对接困难
  → 缓解：先完成模拟盘测试，再对接实盘

## Success Criteria

- [x] 数据管道能够获取中国股市数据
- [x] 因子引擎能够计算 24 维因子
- [x] 回测引擎支持 T+1 和涨跌停规则
- [x] AlphaQuant 模型能够生成因子公式
- [x] 初始化脚本能够设置开发环境
- [x] 运行示例展示完整工作流
- [x] OpenSpec 规范文档完整

## Related Work

- [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) - 原始架构参考
- [Qlib](https://github.com/microsoft/qlib) - 微软量化框架参考
- [Backtrader](https://github.com/mementum/backtrader) - 回测框架参考
