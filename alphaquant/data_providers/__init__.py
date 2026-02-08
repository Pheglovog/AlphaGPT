"""
数据提供者模块

支持的数据源：
- Tushare Pro: 中国股市数据
"""

from .tushare import TushareProProvider

__all__ = ["TushareProProvider"]
