# AlphaGPT 贡献指南

欢迎参与 AlphaGPT 项目的开发！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 Bug 修复
- ✨ 新功能开发
- 📝 文档改进
- 🧪 测试用例添加
- 💡 代码优化

---

## 📋 目录

- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [文档规范](#文档规范)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)

---

## 🔄 开发流程

### 1. 环境准备

```bash
# 1. Fork 项目到你的 GitHub 账号

# 2. 克隆你的 Fork
git clone https://github.com/YOUR_USERNAME/AlphaGPT.git
cd AlphaGPT

# 3. 添加上游仓库
git remote add upstream https://github.com/Pheglovog/AlphaGPT.git

# 4. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 5. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖
```

### 2. 创建分支

```bash
# 同步上游仓库的最新代码
git fetch upstream
git checkout main
git merge upstream/main

# 创建特性分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

**分支命名规范**:
- 新功能: `feature/feature-name`
- Bug 修复: `fix/bug-name`
- 文档: `docs/doc-name`
- 重构: `refactor/refactor-name`
- 性能优化: `perf/perf-name`
- 测试: `test/test-name`

### 3. 开发

```bash
# 进行你的开发工作
# ...

# 运行测试
pytest

# 运行代码检查
pylint alphaquant/
# 或
flake8 alphaquant/
# 或
black alphaquant/
```

### 4. 提交更改

```bash
# 添加更改的文件
git add .

# 提交（遵循提交规范）
git commit -m "feat: add new factor calculation method"
```

### 5. 推送和创建 Pull Request

```bash
# 推送到你的 Fork
git push origin feature/your-feature-name

# 访问 GitHub 创建 Pull Request
```

---

## 📝 代码规范

### Python 代码风格

我们遵循 **PEP 8** 编码规范，并使用以下工具进行代码检查：

#### 格式化工具

```bash
# Black - 代码格式化
pip install black
black alphaquant/

# isort - import 排序
pip install isort
isort alphaquant/

# mypy - 类型检查
pip install mypy
mypy alphaquant/
```

#### 代码检查工具

```bash
# pylint - 代码质量检查
pip install pylint
pylint alphaquant/

# flake8 - 代码风格检查
pip install flake8
flake8 alphaquant/

# bandit - 安全检查
pip install bandit
bandit -r alphaquant/
```

### 代码组织

```python
"""
模块文档字符串

简要说明模块的功能
"""

# 标准库导入
import os
from typing import List, Dict, Optional

# 第三方库导入
import numpy as np
import pandas as pd

# 本地导入
from alphaquant.core.base import BaseClass


class MyNewClass:
    """类文档字符串"""

    def __init__(self, param1: int, param2: str) -> None:
        """初始化方法
        
        Args:
            param1: 参数1说明
            param2: 参数2说明
        """
        self.param1 = param1
        self.param2 = param2

    def my_method(self, arg1: List[int]) -> Dict[str, float]:
        """方法文档字符串
        
        Args:
            arg1: 参数说明
            
        Returns:
            返回值说明
            
        Raises:
            ValueError: 异常说明
        """
        # 实现
        return {"result": 1.0}
```

### 类型注解

我们强烈推荐使用类型注解，使用 `typing` 模块：

```python
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd


def calculate_factor(
    data: pd.DataFrame,
    period: int = 20,
    method: str = "sma"
) -> pd.Series:
    """计算因子
    
    Args:
        data: 价格数据
        period: 计算周期
        method: 计算方法
        
    Returns:
        因子值序列
    """
    # 实现
    return data['close'].rolling(period).mean()


def get_strategy_signals(
    data: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float
) -> Tuple[pd.Series, pd.Series]:
    """生成交易信号
    
    Returns:
        (buy_signals, sell_signals)
    """
    buy_signals = data['factor'] > buy_threshold
    sell_signals = data['factor'] < sell_threshold
    return buy_signals, sell_signals
```

### 文档字符串

使用 **Google 风格**的文档字符串：

```python
def complex_function(
    arg1: int,
    arg2: str,
    opt_arg: Optional[float] = None
) -> Dict[str, Union[int, float]]:
    """复杂函数的完整文档
    
    详细说明函数的功能和用途
    
    Args:
        arg1: 参数1的详细说明
        arg2: 参数2的详细说明
        opt_arg: 可选参数的详细说明
        
    Returns:
        返回字典，包含:
            - 'value1': 值1的说明
            - 'value2': 值2的说明
            
    Raises:
        ValueError: 当arg1小于0时
        TypeError: 当arg2不是字符串时
        
    Examples:
        >>> result = complex_function(10, "test")
        >>> print(result)
        {'value1': 10, 'value2': 0.5}
    """
    # 实现
    pass
```

---

## 🧪 测试要求

### 测试框架

我们使用 `pytest` 作为测试框架。

### 测试文件组织

```
AlphaGPT/
├── alphaquant/
│   ├── core/
│   │   ├── __init__.py
│   │   └── calculator.py
│   └── factors/
│       ├── __init__.py
│       └── technical.py
└── tests/
    ├── __init__.py
    ├── test_calculator.py
    └── test_technical.py
```

### 测试用例示例

```python
import pytest
import pandas as pd
import numpy as np
from alphaquant.factors.technical import TechnicalIndicators


class TestTechnicalIndicators:
    """技术指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=100)
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.randn(100) * 0.01) * 100
        
        return pd.DataFrame({
            'date': dates,
            'open': prices * 0.995,
            'high': prices * 1.005,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }).set_index('date')
    
    def test_sma_calculation(self, sample_data):
        """测试SMA计算"""
        indicators = TechnicalIndicators()
        
        sma_20 = indicators.calculate_sma(sample_data['close'], period=20)
        
        assert len(sma_20) == 100
        assert sma_20.isna().sum() == 19  # 前19个值为NaN
        assert sma_20.dropna().all() > 0
    
    def test_rsi_calculation(self, sample_data):
        """测试RSI计算"""
        indicators = TechnicalIndicators()
        
        rsi = indicators.calculate_rsi(sample_data['close'], period=14)
        
        assert len(rsi) == 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100
    
    @pytest.mark.parametrize("period", [5, 10, 20, 30])
    def test_sma_different_periods(self, sample_data, period):
        """测试不同周期的SMA"""
        indicators = TechnicalIndicators()
        
        sma = indicators.calculate_sma(sample_data['close'], period=period)
        
        assert sma.isna().sum() == period - 1
        assert sma.dropna().all() > 0
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_technical.py

# 运行特定测试函数
pytest tests/test_technical.py::TestTechnicalIndicators::test_sma_calculation

# 显示详细输出
pytest -v

# 显示测试覆盖率
pytest --cov=alphaquant --cov-report=html

# 只运行失败的测试
pytest -lf

# 并行运行测试
pytest -n auto
```

### 测试覆盖率要求

- **最低要求**: 60%
- **推荐目标**: 80%
- **优秀标准**: 90%+

### 添加新功能时的测试要求

1. **单元测试**: 每个新功能必须有对应的单元测试
2. **集成测试**: 对于涉及多个模块的功能，添加集成测试
3. **边界测试**: 测试边界情况和异常情况
4. **性能测试**: 对于性能敏感的代码，添加性能基准测试

---

## 📚 文档规范

### README.md

README.md 是用户了解项目的第一入口，必须包含：

1. 项目简介
2. 核心特性
3. 安装步骤
4. 快速开始
5. 使用示例
6. API 文档链接
7. 贡献指南链接
8. 许可证信息

### API 文档

API 文档应该包含：

1. 模块概述
2. 类和函数列表
3. 每个类/函数的详细说明
4. 参数说明
5. 返回值说明
6. 异常说明
7. 使用示例

### 代码注释

```python
# 单行注释：说明为什么这样做
# 不要说明在做什么

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """计算夏普比率
    
    夏普比率衡量每承担一单位风险所获得的超额收益
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        
    Returns:
        夏普比率（年化）
    """
    # 年化收益率
    annual_return = returns.mean() * 252
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 计算夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    
    return sharpe
```

---

## 📦 提交规范

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。

### 提交消息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type（类型）

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式化（不影响代码逻辑）
- `refactor`: 重构（既不是新功能也不是Bug修复）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具链相关
- `ci`: CI/CD相关

### Scope（范围）

- `data`: 数据相关
- `factor`: 因子相关
- `backtest`: 回测相关
- `model`: 模型相关
- `strategy`: 策略相关
- `execution`: 交易执行相关
- `docs`: 文档相关
- `test`: 测试相关

### 示例

```bash
# 新功能
git commit -m "feat(factor): add new technical indicator Bollinger Bands"

# Bug 修复
git commit -m "fix(backtest): correct T+1 trading rule implementation"

# 文档更新
git commit -m "docs(api): update API documentation for factor calculation"

# 性能优化
git commit -m "perf(data): optimize batch data fetching with async requests"

# 测试
git commit -m "test(factor): add unit tests for RSI calculation"

# 重构
git commit -m "refactor(model): simplify attention mechanism implementation"
```

### 多行提交消息

```bash
git commit -m "feat(factor): add MACD indicator

Implement Moving Average Convergence Divergence (MACD) indicator
with signal line and histogram.

- Calculate MACD line (12-26 EMA)
- Calculate signal line (9 EMA)
- Calculate histogram (MACD - signal)

Closes #123"
```

---

## 🔀 Pull Request 流程

### PR 标题格式

PR 标题应该遵循提交消息格式：

```
<type>(<scope>): <subject>
```

### PR 描述模板

创建 PR 时，请使用以下模板：

```markdown
## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 代码重构
- [ ] 文档更新
- [ ] 性能优化
- [ ] 测试

## 变更描述
简要描述这次PR做了什么...

## 相关 Issue
Closes #(issue number)
Related to #(issue number)

## 变更内容
- 列出主要的变更
- ...

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试
- [ ] 添加了新的测试用例

## 文档
- [ ] 更新了相关文档
- [ ] 添加了使用示例

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 已添加必要的测试
- [ ] 测试通过（pytest）
- [ ] 代码通过检查（pylint, flake8）
- [ ] 文档已更新
- [ ] Commit messages 遵循规范

## 截图（如果适用）
添加截图展示变更效果...

## 其他说明
其他需要说明的内容...
```

### PR 审查流程

1. **自动化检查**
   - 代码风格检查
   - 单元测试
   - 测试覆盖率

2. **人工审查**
   - 代码逻辑审查
   - 架构设计审查
   - 文档完整性审查

3. **反馈和修改**
   - 根据审查意见修改代码
   - 回应所有审查意见
   - 必要时更新文档

4. **合并**
   - 所有检查通过
   - 至少一名维护者批准
   - 合并到主分支

---

## 🎯 最佳实践

### 1. 保持代码简洁

```python
# 不好
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

# 好
def process_data(data: List[int]) -> List[int]:
    """处理数据，过滤正数并加倍"""
    return [item * 2 for item in data if item > 0]
```

### 2. 使用有意义的变量名

```python
# 不好
x = get_data()
y = process(x)
z = save(y)

# 好
raw_data = fetch_stock_data()
processed_data = calculate_factors(raw_data)
save_factors(processed_data)
```

### 3. 避免硬编码

```python
# 不好
def calculate_sma(data):
    return data.rolling(20).mean()

# 好
DEFAULT_SMA_PERIOD = 20

def calculate_sma(data: pd.Series, period: int = DEFAULT_SMA_PERIOD) -> pd.Series:
    """计算简单移动平均"""
    return data.rolling(period).mean()
```

### 4. 使用异常处理

```python
# 不好
def get_stock_price(code: str) -> float:
    data = fetch_data(code)
    return data['close'].iloc[-1]

# 好
def get_stock_price(code: str) -> float:
    """获取最新股价
    
    Args:
        code: 股票代码
        
    Returns:
        最新收盘价
        
    Raises:
        ValueError: 当股票代码无效时
        IndexError: 当没有数据时
    """
    try:
        data = fetch_data(code)
        return data['close'].iloc[-1]
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to get price for {code}: {e}")
        raise
```

---

## 📞 联系方式

如果你有任何问题或需要帮助：

- **GitHub Issues**: [提交问题](https://github.com/Pheglovog/AlphaGPT/issues)
- **Discussions**: [参与讨论](https://github.com/Pheglovog/AlphaGPT/discussions)
- **Email**: 3042569263@qq.com

---

**最后更新**: 2026-02-14
**文档版本**: 1.0.0
