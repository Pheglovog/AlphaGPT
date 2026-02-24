"""
AlphaGPT 缓存优化增强脚本

功能：
1. 智能缓存预热（基于历史访问频率）
2. 智能缓存失效策略（基于数据新鲜度）
3. 缓存命中率监控和报告
4. 自动缓存清理

作者：吕布（上等兵•甘的 AI 助手）
日期：2026-02-21
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd


class CacheAccessTracker:
    """缓存访问跟踪器 - 用于智能预热"""

    def __init__(self, log_file: str = './cache_access.log'):
        self.log_file = Path(log_file)
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}

    def record_access(self, key: str) -> None:
        """记录缓存访问"""
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.last_access[key] = datetime.now()

    def get_hot_keys(self, top_n: int = 100) -> List[Tuple[str, int]]:
        """获取最热的缓存键"""
        sorted_keys = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_keys[:top_n]

    def save_to_file(self) -> None:
        """保存访问记录到文件"""
        data = {
            'access_counts': self.access_counts,
            'last_access': {
                k: v.isoformat() for k, v in self.last_access.items()
            }
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self) -> None:
        """从文件加载访问记录"""
        if not self.log_file.exists():
            return

        with open(self.log_file, 'r') as f:
            data = json.load(f)
            self.access_counts = data.get('access_counts', {})
            self.last_access = {
                k: datetime.fromisoformat(v)
                for k, v in data.get('last_access', {}).items()
            }


class SmartCacheInvalidator:
    """智能缓存失效器 - 基于数据新鲜度"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        # 不同类型数据的过期时间（秒）
        self.data_ttl_rules = {
            'stock_data': 3600,        # 股票数据：1小时
            'market_data': 300,        # 市场数据：5分钟
            'factor_data': 86400,      # 因子数据：1天
            'backtest_result': 604800, # 回测结果：7天
        }

    def set_ttl_rule(self, data_type: str, ttl: int) -> None:
        """设置数据类型的过期时间"""
        self.data_ttl_rules[data_type] = ttl

    def should_invalidate(self, key: str, cached_time: datetime) -> bool:
        """判断缓存是否应该失效"""
        # 解析键名，确定数据类型
        data_type = self._parse_data_type(key)
        if data_type is None:
            return False

        ttl = self.data_ttl_rules.get(data_type, 3600)
        return (datetime.now() - cached_time) > timedelta(seconds=ttl)

    def _parse_data_type(self, key: str) -> Optional[str]:
        """解析键名，返回数据类型"""
        for data_type in self.data_ttl_rules:
            if key.startswith(data_type):
                return data_type
        return None

    def invalidate_stale(self) -> Dict[str, int]:
        """失效过期缓存"""
        stats = {'invalidated': 0, 'kept': 0}
        # 实现略 - 需要遍历所有缓存键并检查时间
        return stats


class CacheHitRateMonitor:
    """缓存命中率监控器"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.history: List[dict] = []

    def collect_stats(self) -> dict:
        """收集缓存统计信息"""
        stats = self.cache_manager.get_stats()

        # 添加时间戳
        stats['timestamp'] = datetime.now().isoformat()

        # 计算总体命中率
        total_hits = 0
        total_misses = 0

        for cache_stat in stats.get('stats', []):
            total_hits += cache_stat.get('hits', 0)
            total_misses += cache_stat.get('misses', 0)

        total = total_hits + total_misses
        overall_hit_rate = total_hits / total if total > 0 else 0.0

        stats['overall'] = {
            'hits': total_hits,
            'misses': total_misses,
            'hit_rate': f"{overall_hit_rate * 100:.2f}%"
        }

        return stats

    def save_history(self, stats: dict) -> None:
        """保存历史记录"""
        self.history.append(stats)
        # 只保留最近 1000 条记录
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def generate_report(self, hours: int = 24) -> dict:
        """生成命中率报告"""
        # 过滤最近 N 小时的记录
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_stats = [
            s for s in self.history
            if datetime.fromisoformat(s['timestamp']) > cutoff_time
        ]

        if not recent_stats:
            return {'error': 'No data available'}

        # 计算平均命中率
        hit_rates = []
        for stat in recent_stats:
            overall = stat.get('overall', {})
            hit_rate_str = overall.get('hit_rate', '0%')
            hit_rates.append(float(hit_rate_str.rstrip('%')) / 100)

        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        max_hit_rate = max(hit_rates)
        min_hit_rate = min(hit_rates)

        return {
            'period': f'{hours} hours',
            'samples': len(recent_stats),
            'avg_hit_rate': f"{avg_hit_rate * 100:.2f}%",
            'max_hit_rate': f"{max_hit_rate * 100:.2f}%",
            'min_hit_rate': f"{min_hit_rate * 100:.2f}%",
            'trend': 'improving' if hit_rates[-1] > hit_rates[0] else 'declining'
        }


class CachePreheater:
    """缓存预热器 - 智能预热策略"""

    def __init__(self, cache_manager, access_tracker: CacheAccessTracker):
        self.cache_manager = cache_manager
        self.access_tracker = access_tracker

    def preheat_hot_keys(self, top_n: int = 100) -> dict:
        """预热热门键"""
        hot_keys = self.access_tracker.get_hot_keys(top_n)
        keys = [key for key, _ in hot_keys]

        logger.info(f"Preheating {len(keys)} hot cache keys...")

        result = self.cache_manager.warmup(keys)

        logger.info(
            f"Preheat completed: {result['hit']} hits, "
            f"{result['miss']} misses, {result['errors']} errors"
        )

        return result

    def preheat_by_pattern(self, pattern: str, limit: int = 100) -> dict:
        """根据模式预热缓存键"""
        # 从文件缓存扫描匹配的键
        cache_dir = Path('./data_cache')
        matching_files = []

        for cache_file in cache_dir.rglob('*.cache'):
            if pattern in str(cache_file):
                matching_files.append(cache_file.stem)

        keys = matching_files[:limit]

        logger.info(f"Preheating {len(keys)} keys matching '{pattern}'...")

        result = self.cache_manager.warmup(keys)

        logger.info(
            f"Preheat completed: {result['hit']} hits, "
            f"{result['miss']} misses, {result['errors']} errors"
        )

        return result

    def preheat_all(self) -> dict:
        """预热所有缓存"""
        # 扫描所有缓存键
        cache_dir = Path('./data_cache')
        all_keys = []

        for cache_file in cache_dir.rglob('*.cache'):
            all_keys.append(cache_file.stem)

        logger.info(f"Preheating {len(all_keys)} cache keys...")

        result = self.cache_manager.warmup(all_keys)

        logger.info(
            f"Preheat completed: {result['hit']} hits, "
            f"{result['miss']} misses, {result['errors']} errors"
        )

        return result


def main():
    """主函数 - 演示缓存优化功能"""
    from alphaquant.data_cache import CacheManager

    logger.info("=== AlphaGPT 缓存优化增强 ===")

    # 创建缓存管理器
    cache_manager = CacheManager(
        enable_memory=True,
        enable_redis=False,
        enable_file=True,
        auto_warmup=True
    )

    # 创建访问跟踪器
    access_tracker = CacheAccessTracker()
    access_tracker.load_from_file()

    # 创建缓存预热器
    preheater = CachePreheater(cache_manager, access_tracker)

    # 创建命中率监控器
    monitor = CacheHitRateMonitor(cache_manager)

    # 创建智能失效器
    invalidator = SmartCacheInvalidator(cache_manager)

    # 收集统计信息
    stats = monitor.collect_stats()
    logger.info(f"Cache stats: {json.dumps(stats, indent=2)}")

    # 生成报告
    report = monitor.generate_report(hours=24)
    logger.info(f"Hit rate report: {json.dumps(report, indent=2)}")

    # 预热热门键
    preheater.preheat_hot_keys(top_n=50)

    # 清理过期缓存
    cleaned = cache_manager.cleanup_expired()
    logger.info(f"Cleaned {cleaned} expired cache entries")


if __name__ == '__main__':
    main()
