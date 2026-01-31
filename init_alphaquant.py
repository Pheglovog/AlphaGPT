#!/usr/bin/env python3
"""
AlphaQuant 初始化脚本
检查环境、安装依赖、创建配置文件
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List


def run_command(cmd: List[str], description: str) -> bool:
    """运行命令"""
    print(f"\n📦 {description}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误: {e.stderr}")
        return False


def check_python_version() -> bool:
    """检查 Python 版本"""
    print("\n🔍 检查 Python 版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 符合要求")
        return True
    else:
        print(f"❌ 需要 Python 3.9+，当前版本: {version.major}.{version.minor}.{version.micro}")
        return False


def install_dependencies() -> bool:
    """安装依赖"""
    print("\n📥 安装依赖包...")

    requirements = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "aiohttp>=3.9.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0",
        "tushare>=1.2.60",
        "pymongo>=4.6.0",
    ]

    # 分批安装
    success = True
    for i in range(0, len(requirements), 5):
        batch = requirements[i:i+5]
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + batch
        if not run_command(cmd, f"安装依赖包 {i+1}-{min(i+5, len(requirements))}"):
            success = False

    return success


def create_directories() -> bool:
    """创建必要的目录"""
    print("\n📁 创建目录结构...")

    directories = [
        "logs",
        "data/cache",
        "data/historical",
        "models/checkpoints",
        "backtest/results",
        "strategies",
        "config"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

    return True


def create_env_file() -> bool:
    """创建 .env 文件"""
    print("\n🔧 创建 .env 文件...")

    env_content = """
# AlphaQuant 环境变量配置
# 请根据实际情况修改

# 数据源
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_CONCURRENT_LIMIT=2  # 并发限制：免费版=2, 2000积分=5, 5000积分=10
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=alphaquant

# 模型
MODEL_D_MODEL=128
MODEL_NHEAD=8
MODEL_NUM_LAYERS=4

# 回测
BACKTEST_INITIAL_CAPITAL=1000000
BACKTEST_COMMISSION_RATE=0.0003

# 策略
STRATEGY_MAX_POSITIONS=5
STRATEGY_BUY_THRESHOLD=0.7
STRATEGY_SELL_THRESHOLD=0.3

# 日志
LOG_LEVEL=INFO
"""

    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content.strip())
        print(f"✅ 创建 .env 文件")
        print(f"⚠️  请编辑 .env 文件，填入你的 Tushare Token")
        return True
    else:
        print(f"ℹ️  .env 文件已存在，跳过")
        return True


def create_gitignore() -> bool:
    """创建 .gitignore"""
    print("\n📝 创建 .gitignore...")

    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# 数据
data/cache/*
data/historical/*
!data/cache/.gitkeep
!data/historical/.gitkeep

# 模型
models/checkpoints/*
!models/checkpoints/.gitkeep

# 回测
backtest/results/*
!backtest/results/.gitkeep

# 日志
logs/*
!logs/.gitkeep

# 配置
.env
config.local.yaml

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db
"""

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(gitignore_content.strip())
        print(f"✅ 创建 .gitignore")
        return True
    else:
        print(f"ℹ️  .gitignore 已存在，跳过")
        return True


def create_gitkeep_files() -> bool:
    """创建 .gitkeep 文件"""
    print("\n📝 创建 .gitkeep 文件...")

    directories = [
        "data/cache",
        "data/historical",
        "models/checkpoints",
        "backtest/results",
        "logs",
        "strategies"
    ]

    for dir_path in directories:
        gitkeep_path = Path(dir_path) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()

    print(f"✅ 创建 .gitkeep 文件")
    return True


def test_imports() -> bool:
    """测试关键依赖"""
    print("\n🧪 测试依赖导入...")

    test_modules = [
        "torch",
        "numpy",
        "pandas",
        "aiohttp",
        "loguru",
        "yaml"
    ]

    success = True
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - 未安装")
            success = False

    return success


def print_next_steps():
    """打印下一步操作"""
    print("\n" + "="*60)
    print("🎉 AlphaQuant 初始化完成！")
    print("="*60)
    print("\n📋 下一步操作:")
    print()
    print("1. 编辑配置文件:")
    print("   nano .env")
    print("   # 或")
    print("   vim .env")
    print()
    print("2. 获取 Tushare Token:")
    print("   访问 https://tushare.pro")
    print("   注册账号并获取 Token")
    print("   填入 .env 文件中的 TUSHARE_TOKEN")
    print()
    print("3. 运行测试:")
    print("   python alphaquant/factors/china_factors.py")
    print("   python alphaquant/model/alpha_quant.py")
    print("   python alphaquant/backtest/backtester.py")
    print()
    print("4. 运行回测:")
    print("   python alphaquant/backtest/run_backtest.py")
    print()
    print("5. 启动面板:")
    print("   streamlit run alphaquant/dashboard/app.py")
    print()
    print("📚 更多信息:")
    print("   - 查看文档: cat OPENSPEC.md")
    print("   - 查看 README: cat README_CN.md")
    print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 AlphaQuant 初始化")
    print("="*60)

    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)

    # 安装依赖
    if not install_dependencies():
        print("\n⚠️  依赖安装失败，请检查网络连接")
        sys.exit(1)

    # 创建目录
    create_directories()

    # 创建配置文件
    create_env_file()
    create_gitignore()
    create_gitkeep_files()

    # 测试导入
    if not test_imports():
        print("\n⚠️  部分依赖未正确安装")
        sys.exit(1)

    # 打印下一步
    print_next_steps()


if __name__ == "__main__":
    main()
