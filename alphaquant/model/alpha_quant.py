"""
AlphaQuant 中国股市量化模型
基于 AlphaGPT 架构，适配中国市场

改进：
- 扩展因子维度（基础6维 + 高级18维）
- 添加市场情绪因子
- 支持多任务学习
- 改进的注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ModelConfig:
    """模型配置"""
    # 维度配置
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    max_formula_len: int = 64

    # 因子配置
    num_basic_factors: int = 6
    num_advanced_factors: int = 18
    total_factors: int = 24

    # 其他
    dropout: float = 0.1
    num_tasks: int = 3  # 多任务数（回测收益、夏普、最大回撤）

    # 新增：因子初始化配置
    factor_init_type: str = "constant"  # constant, uniform, normal


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for Attention"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1) * (d_model ** -0.5))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU Activation"""

    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.fc(x)


class MarketSentimentEncoder(nn.Module):
    """市场情绪编码器"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 市场宽基指标
        self.wide_market = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)  # 修复：保持维度一致
        )

        # 行业轮动
        self.sector = nn.Sequential(
            nn.Linear(8, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)  # 修复：保持维度一致
        )

        # 资金流向
        self.fund_flow = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)  # 修复：保持维度一致
        )

        # 融合层：将所有嵌入合并为 d_model 维度
        self.fusion = nn.Linear(d_model // 4 * 3, d_model)

    def forward(
        self,
        wide_market: torch.Tensor,  # [B, 4] - 涨跌家数、涨停家数、北向资金、指数涨跌
        sector: torch.Tensor,         # [B, 8] - 主要行业涨跌幅
        fund_flow: torch.Tensor        # [B, 3] - 北向、融资、融券
    ) -> torch.Tensor:
        """编码市场情绪"""
        wide_emb = self.wide_market(wide_market)
        sector_emb = self.sector(sector)
        flow_emb = self.fund_flow(fund_flow)

        emb = torch.cat([wide_emb, sector_emb, flow_emb], dim=-1)
        sentiment = self.fusion(emb)

        return sentiment  # [B, d_model]


class FactorAttention(nn.Module):
    """因子注意力机制"""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qk_norm = QKNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] - 因子序列
            mask: [L, L] - 因果掩码

        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # QK-Norm
        Q, K = self.qk_norm(Q, K)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return out


class AlphaQuantLayer(nn.Module):
    """AlphaQuant Transformer Layer"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.norm1 = RMSNorm(config.d_model)
        self.attn = FactorAttention(config.d_model, config.nhead)

        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.dim_feedforward)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
            mask: [L, L]

        Returns:
            [B, L, d_model]
        """
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class AlphaQuantTransformer(nn.Module):
    """AlphaQuant Transformer 编码器"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            AlphaQuantLayer(config) for _ in range(config.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
            mask: [L, L]

        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MultiTaskHead(nn.Module):
    """多任务预测头"""

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config

        # 3个任务头
        self.task_heads = nn.ModuleList([
            nn.Linear(config.d_model, vocab_size) for _ in range(config.num_tasks)
        ])

        # 任务权重（可学习）
        self.task_weights = nn.Parameter(torch.ones(config.num_tasks) / config.num_tasks)

        # 任务路由器
        self.task_router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_tasks)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, d_model] - 最后一个 token 的嵌入

        Returns:
            logits: [B, vocab_size] - 加权后的输出
            task_probs: [B, num_tasks] - 任务权重
        """
        # 任务路由
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)

        # 所有任务输出
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)  # [B, num_tasks, vocab_size]

        # 加权组合
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)

        return weighted, task_probs


class AlphaQuant(nn.Module):
    """
    AlphaQuant - 中国股市量化模型

    基于符号回归的因子生成模型
    """

    # 因子名称（基础）
    BASIC_FACTORS = ['RET', 'PRESSURE', 'MOMO', 'DEV', 'VOL', 'AMP']

    # 因子名称（高级）
    ADVANCED_FACTORS = [
        'RSI', 'MACD', 'BB_POS', 'ATR', 'K', 'J', 'VPT', 'OBV',
        'VOL', 'MOM_5', 'MOM_10', 'MOM_20', 'POS_5', 'POS_10', 'POS_20',
        'NORTH_FLOW', 'MARGIN', 'LIMIT_UP'
    ]

    # 算子（类似 AlphaGPT）
    OPS = [
        ('ADD', lambda a, b: a + b, 2),
        ('SUB', lambda a, b: a - b, 2),
        ('MUL', lambda a, b: a * b, 2),
        ('DIV', lambda a, b: a / (b + 1e-6), 2),
        ('NEG', lambda a: -a, 1),
        ('ABS', lambda a: torch.abs(a), 1),
        ('SIGN', lambda a: torch.sign(a), 1),
        ('MAX', lambda a, b: torch.maximum(a, b), 2),
        ('MIN', lambda a, b: torch.minimum(a, b), 2),
        ('GATE', lambda c, a, b: torch.where(c > 0, a, b), 3),
        ('DELAY1', lambda a: torch.roll(a, 1, dims=-1), 1),
        ('DELAY5', lambda a: torch.roll(a, 5, dims=-1), 1),
        ('SMA', lambda a, n: a.unfold(1, n, 1).mean(dim=-1), 2),
        ('EMA', lambda a, n: a.ewm(span=n, adjust=False).mean(), 2),
        ('STD', lambda a, n: a.unfold(1, n, 1).std(dim=-1), 2),
        ('CORR', lambda a, b: torch.corrcoef(a, b), 2)
    ]

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # 词汇表
        self.vocab = self.BASIC_FACTORS + self.ADVANCED_FACTORS + [op[0] for op in self.OPS]
        self.vocab_size = len(self.vocab)

        # 特征嵌入
        self.feature_emb = nn.Linear(self.config.total_factors, self.config.d_model)

        # 公式 Token 嵌入
        self.token_emb = nn.Embedding(self.vocab_size, self.config.d_model)

        # 位置嵌入
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.config.max_formula_len, self.config.d_model)
        )

        # 市场情绪编码器
        self.sentiment_encoder = MarketSentimentEncoder(self.config.d_model)

        # Transformer 编码器
        self.transformer = AlphaQuantTransformer(self.config)

        # 输出层归一化
        self.ln_f = RMSNorm(self.config.d_model)

        # 多任务预测头
        self.mtp_head = MultiTaskHead(self.config, self.vocab_size)

        # Value 头（用于 Actor-Critic）
        self.value_head = nn.Linear(self.config.d_model, 1)

        # 因子重要性（可学习）- 改进：根据配置初始化
        self._init_factor_importance()

        logger.info(f"AlphaQuant initialized: vocab_size={self.vocab_size}, "
                   f"d_model={self.config.d_model}, num_layers={self.config.num_layers}, "
                   f"factor_init_type={self.config.factor_init_type}")

    def _init_factor_importance(self):
        """初始化因子重要性"""
        if self.config.factor_init_type == "constant":
            # 所有因子同等重要
            self.factor_importance = nn.Parameter(torch.ones(self.config.total_factors))
        elif self.config.factor_init_type == "uniform":
            # 均匀分布初始化（0.5 ~ 1.5）
            self.factor_importance = nn.Parameter(torch.rand(self.config.total_factors) + 0.5)
        elif self.config.factor_init_type == "normal":
            # 正态分布初始化（均值1，标准差0.2）
            self.factor_importance = nn.Parameter(torch.randn(self.config.total_factors) * 0.2 + 1.0)
        else:
            raise ValueError(f"Unknown factor_init_type: {self.config.factor_init_type}")

        # 确保因子重要性为正数（使用 Softmax 归一化）
        self.factor_importance.data = F.softmax(self.factor_importance.data, dim=0) * self.config.total_factors

    def forward(
        self,
        factor_features: torch.Tensor,
        market_sentiment: Optional[torch.Tensor] = None,
        formula_tokens: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Dict:
        """
        前向传播

        Args:
            factor_features: [B, num_factors, T] - 因子特征序列
            market_sentiment: [B, 15] - 市场情绪（可选）
            formula_tokens: [B, L] - 已生成的公式 tokens（可选，用于推理）
            return_attentions: 是否返回注意力权重

        Returns:
            {
                'logits': [B, vocab_size] - 预测的下一个 token
                'value': [B, 1] - Value 估计
                'task_probs': [B, num_tasks] - 任务概率
                'embeddings': [B, L, d_model] - 嵌入序列（如果输入公式）
            }
        """
        B, _, T = factor_features.shape

        # 因子重要性加权
        weighted_features = factor_features * self.factor_importance.view(1, -1, 1)

        # 时间池化：平均池化 + 最大池化
        feat_mean = weighted_features.mean(dim=-1)  # [B, num_factors]
        feat_max = weighted_features.amax(dim=-1)   # [B, num_factors]
        feat_pooled = torch.cat([feat_mean, feat_max], dim=-1)  # [B, 2*num_factors]

        # 特征嵌入
        feat_emb = self.feature_emb(feat_pooled)  # [B, d_model]

        # 添加市场情绪（如果有）
        if market_sentiment is not None:
            sent_emb = self.sentiment_encoder(
                market_sentiment[:, :4],
                market_sentiment[:, 4:12],
                market_sentiment[:, 12:]
            )
            feat_emb = feat_emb + sent_emb  # 修复：使用加法而非 cat

        # 如果没有输入公式，则生成新的
        if formula_tokens is None:
            # 使用特征嵌入作为初始 token
            formula_tokens = torch.zeros(B, 1, dtype=torch.long, device=factor_features.device)

        # Token 嵌入 + 位置嵌入
        B, L = formula_tokens.shape
        tok_emb = self.token_emb(formula_tokens)  # [B, L, d_model]
        pos_emb = self.pos_emb[:, :L, :]
        x = tok_emb + pos_emb

        # 添加特征信息到第一个 token
        x[:, 0, :] = x[:, 0, :] + feat_emb

        # 因果掩码
        mask = torch.tril(torch.ones(L, L, device=factor_features.device))

        # Transformer 编码
        x = self.transformer(x, mask)
        x = self.ln_f(x)

        # 取最后一个 token
        last_emb = x[:, -1, :]  # [B, d_model]

        # 多任务预测
        logits, task_probs = self.mtp_head(last_emb)  # [B, vocab_size], [B, num_tasks]

        # Value 估计
        value = self.value_head(last_emb)  # [B, 1]

        result = {
            'logits': logits,
            'value': value,
            'task_probs': task_probs
        }

        if return_attentions:
            result['embeddings'] = x
            result['mask'] = mask

        return result

    def generate_formula(
        self,
        factor_features: torch.Tensor,
        market_sentiment: Optional[torch.Tensor] = None,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        生成因子公式

        Args:
            factor_features: [B, num_factors, T]
            market_sentiment: [B, 15]
            max_length: 最大长度
            temperature: 采样温度
            top_k: Top-K 采样

        Returns:
            生成的公式 token 列表
        """
        self.eval()
        with torch.no_grad():
            B = factor_features.shape[0]

            # 初始 token
            generated = torch.zeros(B, 1, dtype=torch.long, device=factor_features.device)

            for _ in range(max_length):
                # 前向传播
                output = self.forward(
                    factor_features,
                    market_sentiment,
                    generated
                )

                logits = output['logits'][:, -1, :] / temperature

                # Top-K 采样
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, indices, values)

                # 采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 停止条件（如果生成算子且栈为空，不能继续）
                # 这里简化处理：如果达到最大长度就停止
                generated = torch.cat([generated, next_token], dim=1)

            # 转换为 token 名称
            formulas = []
            for i in range(B):
                formula = [self.vocab[t.item()] for t in generated[i]]
                formulas.append(formula)

            return formulas


def model_test():
    """模型测试"""
    # 配置
    config = ModelConfig(
        d_model=128,
        nhead=8,
        num_layers=4,
        max_formula_len=64,
        factor_init_type="normal"  # 使用正态分布初始化因子重要性
    )

    # 创建模型
    model = AlphaQuant(config)

    # 模拟输入
    batch_size = 4
    num_factors = 24
    time_steps = 100

    factor_features = torch.randn(batch_size, num_factors, time_steps)
    market_sentiment = torch.randn(batch_size, 15)

    # 前向传播
    output = model(factor_features, market_sentiment)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Value shape: {output['value'].shape}")
    print(f"Task probs shape: {output['task_probs'].shape}")

    # 生成公式
    formulas = model.generate_formula(
        factor_features,
        market_sentiment,
        max_length=10
    )

    print(f"\nGenerated formula: {formulas[0]}")


if __name__ == "__main__":
    model_test()
