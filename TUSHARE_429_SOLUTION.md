# Tushare API 429 错误解决方案

## 问题说明

当看到这个错误时：
```
HTTP 429: 您当前使用该API的并发数过高，请降低并发，或联系客服增加限额。
```

说明同时发起的请求数超过了 Tushare Pro 的并发限制。

---

## 原因分析

Tushare Pro 有**两种限制**：

| 限制类型 | 说明 | 免费版 | 2000积分 | 5000积分 |
|---------|------|--------|---------|---------|
| 频次限制 | 每分钟调用次数 | 50次/分钟 | 200次/分钟 | 500次/分钟 |
| **并发限制** | 同时进行的请求数 | **2个** | **5个** | **10个** |

**HTTP 429 是并发限制错误**，不是频次限制。

---

## 已修复

AlphaQuant v1.1 已添加：

### 1. 并发控制
```python
# 自动限制并发请求数
provider = TushareProProvider(token, concurrent_limit=2)
```

### 2. 429 错误重试
```python
# 自动重试，使用指数退避
# 第1次重试: 2秒
# 第2次重试: 4秒
# 第3次重试: 8秒
```

### 3. 批量获取
```python
# 自动控制并发的批量获取
results = await provider.get_daily_quotes_batch(codes)
```

---

## 配置方式

### 方式 1: 修改 .env 文件
```bash
nano .env
```
设置并发限制：
```bash
# 免费版
TUSHARE_CONCURRENT_LIMIT=2

# 2000积分
TUSHARE_CONCURRENT_LIMIT=5

# 5000积分
TUSHARE_CONCURRENT_LIMIT=10
```

### 方式 2: 代码中指定
```python
from alphaquant.data_providers.tushare import TushareProProvider

# 免费版
provider = TushareProProvider(token, concurrent_limit=2)

# 2000积分
provider = TushareProProvider(token, concurrent_limit=5)

# 5000积分
provider = TushareProProvider(token, concurrent_limit=10)
```

---

## 使用建议

### 如果你是免费用户（120积分）

```python
# 配置
concurrent_limit = 2

# 避免批量获取太多股票
# ❌ 一次性获取 100 只股票会触发 429
codes = ['000001.SZ', '000002.SZ', ... 98 more]
results = await provider.get_daily_quotes_batch(codes)  # 可能 429

# ✅ 分批获取
batch_size = 50
for i in range(0, len(codes), batch_size):
    batch = codes[i:i+batch_size]
    await provider.get_daily_quotes_batch(batch)
    await asyncio.sleep(5)  # 批次间休息
```

### 如果你有 2000 积分

```python
# 配置
concurrent_limit = 5

# 可以批量获取更多股票
batch_size = 100  # 每次最多 100 只
```

### 如果你有 5000 积分

```python
# 配置
concurrent_limit = 10

# 可以批量获取全市场
batch_size = 200  # 每次最多 200 只
```

---

## 升级账号的好处

| 功能 | 免费版 | 2000积分 | 5000积分 |
|-----|--------|---------|---------|
| 价格 | 0元/年 | 200元/年 | 500元/年 |
| 并发限制 | 2 | 5 | 10 |
| 每分钟频次 | 50次 | 200次 | 500次无上限 |
| 日线数据 | ✅ | ✅ | ✅ |
| 因子数据 | ❌ | ✅ | ✅ |
| 资金流向 | ❌ | ✅ | ✅ |
| 批量回测 | ❌ | ✅ | ✅ |
| 特色数据 | ❌ | ❌ | ✅ |

**推荐升级方案：**
- 学习测试 → 免费版
- 日常投资 → 2000积分（200元/年）
- 量化研究 → 5000积分（500元/年）

---

## 快速升级

1. 访问 https://tushare.pro/weborder/#/permission
2. 选择捐助金额
3. 完成支付
4. 等待积分到账（通常几分钟）
5. 更新 .env 中的 `TUSHARE_CONCURRENT_LIMIT`

---

## 免费获取积分

1. 关注"挖地兔"公众号 + 小红书
2. 加入 Tushare QQ 群（1036369385）
3. @群主："已关注公众号和小红书"
4. 免费获得 **200 积分**

---

## 故障排查

### 问题 1: 仍然频繁 429

**检查并发设置：**
```python
print(provider.concurrent_limit)  # 应该是 2, 5, 或 10
```

**检查是否有其他程序在调用：**
```bash
# 确保只有一个 AlphaQuant 实例在运行
ps aux | grep python
```

### 问题 2: 积分不足

**错误信息：**
```
积分不足，请充值
```

**解决：**
1. 等待每日积分重置（如果有）
2. 捐助充值（200元 = 2000积分）
3. 减少请求次数，只获取必要数据

### 问题 3: 请求超时

**检查网络：**
```bash
curl http://api.tushare.pro
```

**增加超时时间：**
```python
async def _request(..., timeout=30):  # 增加到 60
```

---

## 监控请求

AlphaQuant 会自动记录请求统计：

```python
async with TushareProProvider(token, concurrent_limit=2) as provider:
    # ... 执行请求 ...

# 关闭时会打印：
# Tushare stats: 1234 requests in 245.6s (5.02 req/s)
```

---

## 联系支持

如果问题持续：

1. Tushare 官网：https://tushare.pro
2. QQ 群：1036369385（免费）
3. 微信群：关注公众号获取

---

**版本**: AlphaQuant v1.1
**更新日期**: 2026-01-31
