"""
Tushare Pro 数据提供者
获取中国股市历史行情、财务数据、市场信息等

参考文档：https://tushare.pro/document/2
"""

import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger


class TushareProProvider:
    """Tushare Pro 数据接口"""

    # 并发限制（根据账号等级调整）
    FREE_CONCURRENT_LIMIT = 2      # 免费版：2个并发
    PRO2000_CONCURRENT_LIMIT = 5   # 2000积分：5个并发
    PRO5000_CONCURRENT_LIMIT = 10   # 5000积分：10个并发

    def __init__(self, token: str, concurrent_limit: Optional[int] = None):
        """
        初始化 Tushare Pro 接口

        Args:
            token: Tushare Pro API Token (从 https://tushare.pro 获取)
            concurrent_limit: 并发限制（默认使用免费版限制）
        """
        self.token = token
        self.api_url = "http://api.tushare.pro"
        self.session: Optional[aiohttp.ClientSession] = None

        # 并发控制
        self.concurrent_limit = concurrent_limit or self.FREE_CONCURRENT_LIMIT
        self._semaphore = asyncio.Semaphore(self.concurrent_limit)

        # 请求统计
        self._request_count = 0
        self._start_time = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self._start_time = datetime.now()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

        # 打印统计信息
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            rate = self._request_count / duration if duration > 0 else 0
            logger.info(f"Tushare stats: {self._request_count} requests in {duration:.1f}s ({rate:.2f} req/s)")

    async def _request(
        self,
        api_name: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict:
        """
        发送 API 请求（带并发控制和重试）

        Args:
            api_name: Tushare API 名称
            params: 请求参数
            max_retries: 最大重试次数

        Returns:
            API 响应数据
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # 使用信号量控制并发
        async with self._semaphore:
            payload = {
                'api_name': api_name,
                'token': self.token,
                'params': params,
                'fields': ''
            }

            # 指数退避重试
            for attempt in range(max_retries + 1):
                try:
                    async with self.session.post(self.api_url, json=payload, timeout=30) as resp:
                        # 处理 429 并发限制错误
                        if resp.status == 429:
                            delay = (2 ** attempt) * 2  # 2, 4, 8, 16...
                            logger.warning(
                                f"Tushare rate limit hit (429), "
                                f"retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})"
                            )
                            await asyncio.sleep(delay)
                            continue

                        # 其他错误
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(f"Tushare HTTP error: {resp.status} - {text}")
                            if attempt == max_retries:
                                return {}
                            await asyncio.sleep(2 ** attempt)
                            continue

                        data = await resp.json()

                        # 检查业务错误
                        if data.get('code') != 0:
                            error_msg = data.get('msg', 'Unknown error')
                            logger.error(f"Tushare API error [{api_name}]: {error_msg}")

                            # 积分不足错误
                            if '积分' in error_msg:
                                logger.error("Tushare 积分不足，请充值或等待重置")
                                return {}

                            if attempt == max_retries:
                                return {}

                            # 短暂等待后重试
                            await asyncio.sleep(2 ** attempt)
                            continue

                        # 成功
                        self._request_count += 1
                        return data.get('data', {})

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Tushare request timeout, "
                        f"retrying in {2 ** attempt}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                except aiohttp.ClientError as e:
                    logger.error(f"Tushare connection error: {e}")
                    if attempt == max_retries:
                        return {}
                    await asyncio.sleep(2 ** attempt)
                    continue

                except Exception as e:
                    logger.error(f"Tushare unexpected error: {e}")
                    return {}

            # 所有重试失败
            return {}

    async def get_stock_list(self, exchange: str = 'SSE') -> pd.DataFrame:
        """
        获取股票列表

        Args:
            exchange: 交易所 (SSE=上交所, SZSE=深交所)

        Returns:
            股票列表 DataFrame
        """
        params = {
            'exchange': exchange,
            'list_status': 'L'  # L=上市
        }

        data = await self._request('stock_basic', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'symbol': data['items'][1],
            'name': data['items'][2],
            'area': data['items'][3],
            'industry': data['items'][4],
            'market': data['items'][5],
            'list_date': data['items'][6]
        })

        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
        return df

    async def get_daily_quotes(
        self,
        ts_code: str = '',
        trade_date: str = '',
        start_date: str = '',
        end_date: str = ''
    ) -> pd.DataFrame:
        """
        获取日线行情

        Args:
            ts_code: 股票代码 (如: 000001.SZ)
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            日线行情 DataFrame
        """
        params = {
            'ts_code': ts_code,
            'trade_date': trade_date,
            'start_date': start_date,
            'end_date': end_date
        }

        data = await self._request('daily', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'trade_date': data['items'][1],
            'open': data['items'][2],
            'high': data['items'][3],
            'low': data['items'][4],
            'close': data['items'][5],
            'pre_close': data['items'][6],
            'change': data['items'][7],
            'pct_chg': data['items'][8],
            'vol': data['items'][9],
            'amount': data['items'][10]
        })

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.sort_values('trade_date')

    async def get_daily_quotes_batch(
        self,
        ts_codes: List[str],
        start_date: str = '',
        end_date: str = ''
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线行情（自动控制并发）

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {股票代码: DataFrame} 字典
        """
        logger.info(f"Fetching daily quotes for {len(ts_codes)} stocks (concurrent limit: {self.concurrent_limit})")

        tasks = [
            self.get_daily_quotes(ts_code=code, start_date=start_date, end_date=end_date)
            for code in ts_codes
        ]

        # 由于有并发控制，这里可以安全地并发
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        output = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {ts_codes[i]}: {result}")
            elif not result.empty:
                output[ts_codes[i]] = result

        logger.info(f"Successfully fetched {len(output)}/{len(ts_codes)} stocks")
        return output

    async def get_stock_factor(
        self,
        ts_code: str = '',
        start_date: str = '',
        end_date: str = ''
    ) -> pd.DataFrame:
        """
        获取日线因子（市值、市盈率等）

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            因子数据 DataFrame
        """
        params = {
            'ts_code': ts_code,
            'start_date': start_date,
            'end_date': end_date
        }

        data = await self._request('daily_basic', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'trade_date': data['items'][1],
            'turnover_rate': data['items'][2],
            'volume_ratio': data['items'][3],
            'pe': data['items'][4],
            'pe_ttm': data['items'][5],
            'pb': data['items'][6],
            'ps': data['items'][7],
            'ps_ttm': data['items'][8],
            'dv_ratio': data['items'][9],
            'dv_ttm': data['items'][10],
            'total_share': data['items'][11],
            'float_share': data['items'][12],
            'free_share': data['items'][13],
            'total_mv': data['items'][14],
            'circ_mv': data['items'][15]
        })

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.sort_values('trade_date')

    async def get_money_flow(
        self,
        ts_code: str = '',
        trade_date: str = ''
    ) -> pd.DataFrame:
        """
        获取个股资金流向

        Args:
            ts_code: 股票代码
            trade_date: 交易日期

        Returns:
            资金流向 DataFrame
        """
        params = {
            'ts_code': ts_code,
            'trade_date': trade_date
        }

        data = await self._request('moneyflow', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'trade_date': data['items'][1],
            'buy_lg_vol': data['items'][2],
            'buy_lg_amount': data['items'][3],
            'sell_lg_vol': data['items'][4],
            'sell_lg_amount': data['items'][5],
            'buy_elg_vol': data['items'][6],
            'buy_elg_amount': data['items'][7],
            'sell_elg_vol': data['items'][8],
            'sell_elg_amount': data['items'][9],
            'buy_md_vol': data['items'][10],
            'buy_md_amount': data['items'][11],
            'sell_md_vol': data['items'][12],
            'sell_md_amount': data['items'][13],
            'buy_sm_vol': data['items'][14],
            'buy_sm_amount': data['items'][15],
            'sell_sm_vol': data['items'][16],
            'sell_sm_amount': data['items'][17]
        })

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.sort_values('trade_date')

    async def get_limit_list(self, trade_date: str = '') -> pd.DataFrame:
        """
        获取涨跌停股票列表

        Args:
            trade_date: 交易日期

        Returns:
            涨跌停列表 DataFrame
        """
        params = {
            'trade_date': trade_date,
            'limit_type': 'U'  # U=涨停, D=跌停
        }

        data = await self._request('limit_list', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'trade_date': data['items'][1],
            'name': data['items'][2],
            'close': data['items'][3],
            'pct_chg': data['items'][4],
            'amplitude': data['items'][5],
            'up_limit': data['items'][6],
            'down_limit': data['items'][7],
            'open_times': data['items'][8],
            'limit_times': data['items'][9],
            'fd_amount': data['items'][10],
            'first_time': data['items'][11],
            'last_time': data['items'][12],
            'open_amount': data['items'][13],
            'limit_amount': data['items'][14],
            'lmt_ma': data['items'][15],
            'lmt_ma_5': data['items'][16]
        })

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.sort_values('pct_chg', ascending=False)

    async def get_index_daily(self, ts_code: str = '000001.SH', start_date: str = '', end_date: str = '') -> pd.DataFrame:
        """
        获取指数日线行情

        Args:
            ts_code: 指数代码 (000001.SH=上证指数, 399001.SZ=深证成指)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数行情 DataFrame
        """
        params = {
            'ts_code': ts_code,
            'start_date': start_date,
            'end_date': end_date
        }

        data = await self._request('index_daily', params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame({
            'ts_code': data['items'][0],
            'trade_date': data['items'][1],
            'close': data['items'][2],
            'open': data['items'][3],
            'high': data['items'][4],
            'low': data['items'][5],
            'vol': data['items'][6],
            'amount': data['items'][7]
        })

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.sort_values('trade_date')


class TushareDataManager:
    """Tushare 数据管理器 - 统一接口"""

    def __init__(self, token: str, concurrent_limit: Optional[int] = None):
        self.token = token
        self.concurrent_limit = concurrent_limit
        self._provider: Optional[TushareProProvider] = None

    async def initialize(self):
        """初始化数据连接"""
        self._provider = TushareProProvider(self.token, self.concurrent_limit)
        await self._provider.__aenter__()
        logger.info("TusharePro Provider initialized")

    async def close(self):
        """关闭数据连接"""
        if self._provider:
            await self._provider.__aexit__()
            logger.info("TusharePro Provider closed")
