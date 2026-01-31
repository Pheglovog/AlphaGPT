# AlphaQuant Model Specification

## Purpose
AlphaQuant 模型是基于 Transformer 的因子生成模型，使用符号回归的方式自动生成可解释的量化因子公式。它支持多任务学习（收益、夏普、回撤），并集成市场情绪信息。

## Requirements

### Requirement: 模型架构
系统 SHALL 使用改进的 Transformer 架构进行因子公式生成。

#### Scenario: QK-Norm 注意力
- GIVEN 查询（Q）和键（K）的嵌入
- WHEN 计算 QK-Norm
- THEN 对 Q 和 K 分别进行 L2 归一化
- AND 乘以可学习的 scale 参数

#### Scenario: SwiGLU 激活函数
- GIVEN 前馈层的输入
- WHEN 通过 SwiGLU 激活
- THEN 输入分为两部分 x 和 gate
- AND 返回 x * silu(gate)

#### Scenario: RMSNorm 归一化
- GIVEN 层输入
- WHEN 计算 RMSNorm
- THEN 返回 (x / sqrt(mean(x^2) + eps)) * weight
- AND weight 是可学习参数

#### Scenario: 多头注意力
- GIVEN 模型维度和头数
- WHEN 配置多头注意力
- THEN 将 d_model 分为 nhead * head_dim
- AND 并行计算多个注意力头

### Requirement: 因子嵌入
系统 SHALL 将因子特征和算子嵌入到统一空间。

#### Scenario: 特征嵌入
- GIVEN 因子特征（24 维基础 + 高级）
- WHEN 投影到模型空间
- THEN 返回 [batch, 2*24, d_model] 的嵌入
- AND 2*24 = 均值池化 + 最大池化

#### Scenario: Token 嵌入
- GIVEN 因子/算子的 Token ID
- WHEN 查询嵌入表
- THEN 返回 [batch, seq_len, d_model] 的嵌入

#### Scenario: 位置嵌入
- GIVEN 序列长度
- WHEN 获取位置嵌入
- THEN 返回 [1, max_len, d_model] 的可学习嵌入
- AND 使用正弦位置编码初始化

### Requirement: 市场情绪编码
系统 SHALL 编码市场层面的情绪信息，辅助因子生成。

#### Scenario: 编码宽基指标
- GIVEN 市场宽基数据（涨跌家数、涨停家数、北向资金、指数涨跌）
- WHEN 通过全连接层
- THEN 返回 [batch, d_model/4] 的嵌入

#### Scenario: 编码行业轮动
- GIVEN 主要行业的涨跌幅（8 个行业）
- WHEN 通过全连接层
- THEN 返回 [batch, d_model/4] 的嵌入

#### Scenario: 编码资金流向
- GIVEN 北向、融资、融券数据
- WHEN 通过全连接层
- THEN 返回 [batch, d_model/4] 的嵌入

#### Scenario: 融合情绪信息
- GIVEN 宽基、行业、资金流向嵌入
- WHEN 拼接并融合
- THEN 返回 [batch, d_model] 的情绪嵌入
- AND 加到特征嵌入上

### Requirement: 多任务学习
系统 SHALL 支持多目标优化，同时预测因子有效性和价值。

#### Scenario: 三任务预测头
- GIVEN 编码器输出
- WHEN 通过三个任务头
- THEN 分别预测回测收益、夏普比率、最大回撤
- AND 每个任务头是独立的 Linear 层

#### Scenario: 任务路由
- GIVEN 编码器输出
- WHEN 通过任务路由器
- THEN 返回每个任务的权重概率
- AND 使用 softmax 确保和为 1

#### Scenario: 加权组合
- GIVEN 三个任务的输出和权重
- WHEN 加权组合
- THEN 返回 weighted = sum(weight[i] * output[i])
- AND 作为最终预测的 logits

### Requirement: 符号回归算子
系统 SHALL 定义可用于生成因子的算子集。

#### Scenario: 算术算子
- GIVEN 两个操作数
- WHEN 应用算术算子
- THEN 支持：ADD, SUB, MUL, DIV
- AND DIV 使用 (a / (b + 1e-6)) 避免除零

#### Scenario: 聚合算子
- GIVEN 两个操作数
- WHEN 应用聚合算子
- THEN 支持：MAX, MIN
- AND 返回元素级最大/最小值

#### Scenario: 条件算子
- GIVEN 三个操作数（条件、真值、假值）
- WHEN 应用 GATE 算子
- THEN 返回 where(condition > 0, true_value, false_value)

#### Scenario: 一元算子
- GIVEN 一个操作数
- WHEN 应用一元算子
- THEN 支持：NEG, ABS, SIGN
- AND NEG = -a, ABS = |a|, SIGN = sign(a)

#### Scenario: 时序算子
- GIVEN 时间序列和参数
- WHEN 应用时序算子
- THEN 支持：
  - DELAY1：滞后 1 期
  - DELAY5：滞后 5 期
  - SMA：简单移动平均
  - EMA：指数移动平均
  - STD：滚动标准差

### Requirement: 公式生成
系统 SHALL 自动生成有效的因子公式序列。

#### Scenario: Top-K 采样
- GIVEN 预测的 logits 和温度
- WHEN 应用 Top-K 采样
- THEN 只保留 Top-K 个 token 的概率
- AND 重新归一化后采样

#### Scenario: 滥度采样
- GIVEN logits 和温度
- WHEN 应用温度
- THEN logits = logits / temperature
- AND 使用 softmax 得到概率

#### Scenario: 自回归生成
- GIVEN 初始 token 和特征
- WHEN 自回归生成
- THEN 循环：预测 → 采样 → 拼接
- AND 直到达到最大长度或结束符

#### Scenario: 语法验证
- GIVEN 生成的公式 token 序列
- WHEN 验证语法
- THEN 使用栈虚拟机验证可执行性
- AND 算子参数数量正确

### Requirement: 因子重要性学习
系统 SHALL 学习每个因子的重要性，自动加权。

#### Scenario: 因子权重
- GIVEN 计算后的因子
- WHEN 应用可学习的权重
- THEN 返回 factor * importance_weight
- AND weight 初始化为 1，可学习

#### Scenario: 正则化
- GIVEN 因子权重
- WHEN 应用 L2 正则化
- THEN 在损失函数中加入 ||w||^2 * lambda
- AND 防止权重过大

## Model Architecture

### 整体结构
```
输入层
├── 因子特征 [B, 24, T]
│   ├── 均值池化 → [B, 24]
│   └── 最大池化 → [B, 24]
├── 拼接 → [B, 48]
├── 特征嵌入 → [B, d_model]
└── 位置嵌入 → [1, L, d_model]

市场情绪编码器
├── 宽基指标 → [B, d_model/4]
├── 行业轮动 → [B, d_model/4]
├── 资金流向 → [B, d_model/4]
└── 融合 → [B, d_model]

Transformer 编码器
├── QK-Norm 多头注意力
├── SwiGLU 前馈网络
├── RMSNorm 层归一化
└── 残差连接

多任务预测头
├── 任务路由 → [B, 3]
├── 任务 1：回测收益 → [B, vocab_size]
├── 任务 2：夏普比率 → [B, vocab_size]
├── 任务 3：最大回撤 → [B, vocab_size]
└── 加权组合 → [B, vocab_size]

输出
├── logits：下一个 token 的预测
├── value：价值估计（Actor-Critic）
└── task_probs：任务权重
```

### 模型参数
| 参数 | 默认值 | 说明 |
|-----|--------|------|
| d_model | 128 | 模型维度 |
| nhead | 8 | 注意力头数 |
| num_layers | 4 | Transformer 层数 |
| dim_feedforward | 512 | FFN 隐藏层维度 |
| max_formula_len | 64 | 最大公式长度 |
| num_factors | 24 | 因子数量 |
| dropout | 0.1 | Dropout 比例 |
| vocab_size | ~50 | 词汇表大小 |

## Training Requirements

### Loss Function
系统 SHALL 使用多任务损失函数训练模型。

#### Scenario: 计算预测损失
- GIVEN 预测 logits 和真实 token
- WHEN 计算交叉熵损失
- THEN 返回 CE(logits, targets)

#### Scenario: 计算价值损失
- GIVEN 预测 value 和回测收益
- WHEN 计算价值损失
- THEN 返回 MSE(value, rewards)

#### Scenario: 计算总损失
- GIVEN 预测损失、价值损失、因子权重正则化
- WHEN 计算总损失
- THEN 返回 CE + lambda * MSE + reg

### Optimization
系统 SHALL 使用优化器训练模型。

#### Scenario: 学习率调度
- GIVEN 训练步数
- WHEN 调整学习率
- THEN 使用 warm-up + cosine decay

#### Scenario: 梯度裁剪
- GIVEN 模型梯度
- WHEN 裁剪梯度
- THEN 将梯度范数限制在 max_norm 以内

#### Scenario: 检查点保存
- GIVEN 训练周期
- WHEN 保存检查点
- THEN 保存模型权重、优化器状态、训练步数

## Inference Requirements

### Formula Generation
系统 SHALL 高效生成因子公式。

#### Scenario: 批量生成
- GIVEN 批量的因子特征
- WHEN 批量生成公式
- THEN 并行处理所有样本
- AND 返回批量公式列表

#### Scenario: Beam Search
- GIVEN 因子特征和 beam 宽度
- WHEN 使用 Beam Search 生成
- THEN 维持 top-k 候选
- AND 返回得分最高的公式

#### Scenario: 可控生成
- GIVEN 温度和 Top-K 参数
- WHEN 控制生成过程
- THEN 温度越低越确定性
- AND Top-K 越小越保守

## Performance Requirements

- 训练速度：> 1000 样本/秒（GPU）
- 推理延迟：单次生成 < 50ms
- 模型大小：< 100MB
- GPU 内存：< 2GB
- 公式验证：< 1ms
