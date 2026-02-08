# AlphaGPT 安全扫描报告

**扫描时间**: 2026-02-09
**更新时间**: 2026-02-09 13:05 (UTC+8)
**工具**: Bandit 1.9.3
**扫描范围**: AlphaGPT 项目（排除 venv）

---

## 摘要

| 严重程度 | 修复前 | 修复后 | 状态 |
|---------|--------|--------|------|
| 高 (High) | 1 | 0 | ✅ 已修复 |
| 中 (Medium) | 5 | 7 | 📝 待评估 |
| 低 (Low) | 0 | 0 | - |

---

## 已修复问题

### ✅ High: 使用弱 MD5 哈希 - 已修复

**ID**: B324:hashlib
**文件**: `./alphaquant/data_cache.py:154`
**CWE**: CWE-327

**问题代码**:
```python
cache_key = hashlib.md5(param_str.encode()).hexdigest()
```

**修复后**:
```python
# 添加 usedforsecurity=False 表明仅用于缓存键
cache_key = hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()
```

**状态**: ✅ 已修复

---

### ✅ SQL 注入防护 - 已加固

**文件**:
- `./dashboard/data_service.py:63`
- `./model_core/data_loader.py:16`
- `./model_core/data_loader.py:23`

**修复措施**:
- 添加参数类型验证 (`int(limit)`)
- 添加参数范围限制 (`max(1, min(limit, 1000))`)
- 添加地址字符串转义 (`addr.replace("'", "")`)

**状态**: ✅ 已加固（bandit 仍会报告但实际风险已降低）

---

## 剩余问题详情

### 🔴 High: 使用弱 MD5 哈希

**ID**: B324:hashlib
**文件**: `./alphaquant/data_cache.py:154`
**CWE**: CWE-327

**问题代码**:
```python
cache_key = hashlib.md5(param_str.encode()).hexdigest()
```

**说明**: 使用 MD5 哈希进行安全目的不够安全，存在碰撞风险。

**修复建议**:
- 如果仅用于缓存键（非安全目的），添加 `usedforsecurity=False`
- 如果用于安全目的，使用 SHA-256 或更强的哈希算法

**修复方案**:
```python
# 方案 1: 用于缓存（推荐）
cache_key = hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()

# 方案 2: 用于安全
cache_key = hashlib.sha256(param_str.encode()).hexdigest()
```

---

### 🟡 Medium: 不安全的 Pickle 反序列化 (3 处)

**ID**: B301:blacklist
**CWE**: CWE-502

#### 1. 文件: `./alphaquant/data_cache.py:165`
```python
with open(index_file, 'rb') as f:
    self.cache_index = pickle.load(f)
```

#### 2. 文件: `./alphaquant/data_cache.py:220`
```python
with gzip.open(cache_file, 'rb') as f:
    data = pickle.load(f)
```

#### 3. 文件: `./alphaquant/data_cache.py:223`
```python
with open(cache_file, 'rb') as f:
    data = pickle.load(f)
```

**说明**: Pickle 反序列化可能执行任意代码，如果数据来源不受信任则存在安全风险。

**修复建议**:
- 确保缓存文件来自可信来源
- 考虑使用 JSON 或其他安全格式
- 添加数据验证和签名检查

**当前风险评估**: **低** - 这些是本地缓存文件，不受外部影响

---

### 🟡 Medium/Low: SQL 注入风险 (4 处)

**ID**: B608:hardcoded_sql_expressions
**CWE**: CWE-89

#### 1. 文件: `./dashboard/data_service.py:63`
```python
query = f"""
SELECT t.symbol, o.address, o.close, o.volume, o.liquidity, o.fdv, o.time
FROM ohlcv o
JOIN tokens t ON o.address = t.address
WHERE o.time = (SELECT MAX(time) FROM ohlcv)
ORDER BY o.liquidity DESC
LIMIT {limit}
"""
```

#### 2. 文件: `./model_core/data_loader.py:16`
```python
top_query = f"""
SELECT address FROM tokens
LIMIT {limit_tokens}
"""
```

#### 3. 文件: `./model_core/data_loader.py:23`
```python
data_query = f"""
SELECT time, address, open, high, low, close, volume, liquidity, fdv
FROM ohlcv
WHERE address IN ({addr_str})
ORDER BY time ASC
"""
```

#### 4. 文件: `./strategy_manager/runner.py:76`
```python
query = f"""
SELECT address, count(*) as cnt
FROM ohlcv
GROUP BY address
ORDER BY cnt DESC
"""
```

**说明**: 使用 f-string 构建 SQL 查询可能存在 SQL 注入风险。

**修复建议**:
- 使用参数化查询
- 添加输入验证和类型检查
- 限制参数范围（如使用 `max(limit, 0)` 或 `min(limit, 1000)`）

**当前风险评估**: **低-中** - 参数来自内部代码，但仍有改进空间

---

## 剩余问题说明

### 🟡 Medium: 不安全的 Pickle 反序列化 (4 处)

**文件**:
- `./alphaquant/data_cache.py:165`
- `./alphaquant/data_cache.py:220`
- `./alphaquant/data_cache.py:223`
- `./train_real_data.py:123`

**说明**: Pickle 反序列化可能执行任意代码，但这些都是本地缓存文件，不受外部影响。

**风险评估**: **低** - 这些是可信的内部缓存/模型文件

**处理方案**:
- 当前保持不变（内部使用）
- 后续可考虑迁移到 JSON 或 HDF5 格式
- 添加 `# nosec` 注释抑制 bandit 警告

### 🟡 Medium: 不安全的 PyTorch load (1 处)

**文件**: `./train_real_data.py:555`

**说明**: PyTorch load 类似于 pickle，但用于加载本地训练的模型检查点。

**风险评估**: **低** - 仅加载自己训练的模型

**处理方案**:
- 当前保持不变
- 后续可使用 `torch.load(..., weights_only=True)` 增强安全性

---

## 修复优先级

### 高优先级
1. ✅ **修复 MD5 哈希** - 添加 `usedforsecurity=False`（缓存键非安全用途）

### 中优先级
2. ⚠️ **SQL 注入防护** - 添加参数验证和类型检查
3. 📝 **Pickle 反序列化** - 添加文档说明数据来源可信

---

## 修复计划

| 问题 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| MD5 哈希 | data_cache.py:154 | 高 | ✅ 已修复 |
| Pickle #1 | data_cache.py:165 | 低 | 📝 已评估（可信文件） |
| Pickle #2 | data_cache.py:220 | 低 | 📝 已评估（可信文件） |
| Pickle #3 | data_cache.py:223 | 低 | 📝 已评估（可信文件） |
| Pickle #4 | train_real_data.py:123 | 低 | 📝 已评估（可信文件） |
| PyTorch load | train_real_data.py:555 | 低 | 📝 已评估（可信文件） |
| SQL #1 | data_service.py:63 | 中 | ✅ 已加固 |
| SQL #2 | data_loader.py:16 | 中 | ✅ 已加固 |
| SQL #3 | data_loader.py:23 | 中 | ✅ 已加固 |
| SQL #4 | runner.py:76 | 低 | 📝 已评估（硬编码查询） |

---

## 下一步行动

- [x] 修复 MD5 哈希问题（已添加 `usedforsecurity=False`）
- [x] 为 SQL 查询添加参数验证
- [x] 设置 GitHub Actions 自动安全扫描
- [ ] 评估 pickle/PyTorch load 替代方案（可选）
- [ ] 考虑迁移到 JSON/HDF5 格式（可选）

---

**报告生成时间**: 2026-02-09 13:01 (UTC+8)
**工具版本**: Bandit 1.9.3
