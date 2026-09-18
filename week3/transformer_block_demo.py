import torch
import math
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if hidden_size % num_heads != 0: #%取余数。每个 Head 必须获得相同数量的特征，因此 hidden_size 必须能被 num_heads 整除。
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads #4 // 2 = 2，每个 Head 得到 2 个特征

        #四个 nn.Linear(4,4) 分别对应 WQ、WK、WV、WO
        self.query_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.key_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.value_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        # WO：学习如何将多个 Head 的输出合并成最终的输出

    def _split_heads(self, tensor): #分离
        batch_size, sequence_length, _ = ( #_ 表示这里知道第三维存在，但不需要单独使用；它应当等于 hidden_size
            tensor.shape
        )

        tensor = tensor.reshape( #[1,4,4] -> [1,4,2,2]
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.transpose(1, 2) #[B,S,h,d] -> [B,h,S,d]

    def _merge_heads(self, tensor): #合并
        batch_size, _, sequence_length, _ = (
            tensor.shape
        )

        tensor = tensor.transpose(1, 2)
        tensor = tensor.contiguous() #contiguous() 创建连续布局，之后合并维度更明确

        return tensor.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        is_causal=False,
    ):
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        #[1,2,4,2] 四个维度分别是：[batch, head, sequence, head_dim]
        #这里并不是复制两份相同的 Q，而是把投影得到的 4 个特征分成两组：Head 0：前两个特征；Head 1：后两个特征
        #由于线性投影的每个输出特征都有独立参数，两个 Head 可以学习不同的表示空间

        key_transposed = key.transpose(-2, -1) #只交换最后两个维度：[B,h,S,d] → [B,h,d,S]

        attention_scores = torch.matmul(
            query,
            key_transposed,
        )
        # query           [1,2,4,2]
        # key_transposed  [1,2,2,4]
        # --------------------------------
        # attention       [1,2,4,4]

        scale = math.sqrt(self.head_dim) #现在每个 Head 的 Q/K 最后一维只有 head_dim=2，所以点积方差由 head_dim 决定
        scaled_scores = attention_scores / scale

        if attention_mask is not None:
            key_padding_mask = attention_mask.unsqueeze(
                dim=1,
            ).unsqueeze(
                dim=2,
            ) == 0
            # 广播对应关系：
            # Scores [1,2,4,4]
            # Mask   [1,1,1,4]
            #         │ │ │ └─ Key 位置
            #         │ │ └── Query 占位维，1 扩展为 4
            #         │ └──── Head 占位维，1 扩展为 2
            #         └────── Batch

            scaled_scores = scaled_scores.masked_fill(
                key_padding_mask,
                float("-inf"),
            )

        if is_causal:
            sequence_length = hidden_states.size(1)

            causal_mask = torch.ones(
                (
                    sequence_length,
                    sequence_length,
                ),
                dtype=torch.bool,
                device=hidden_states.device,
            )

            causal_mask = torch.triu(
                causal_mask,
                diagonal=1,
            )

            causal_mask = causal_mask.unsqueeze(
                dim=0,
            ).unsqueeze(
                dim=0,
            )

            scaled_scores = scaled_scores.masked_fill(
                causal_mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1,
        )

        context_by_head = torch.matmul(
            attention_weights,
            value,
        )

        merged_context = self._merge_heads(
            context_by_head
        )

        output = self.output_projection(
            merged_context
        )

        return output, attention_weights

#RMSNorm 不减均值，只控制整体尺度：RMS(x) = sqrt(mean(x²) + eps)；RMSNorm(x) = x / RMS(x) × weight
#计算出来的均方根为1，但均值不一定是0。RMSNorm 只控制整体尺度，允许均值偏移。它的参数 weight 形状 [4]，初始为 1。
#RMSNorm 通常只有缩放参数 weight，没有 LayerNorm 中的平移参数 bias，结构更简单。
class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter( #nn.Parameter 告诉 PyTorch：这个 Tensor 是模型需要训练的参数，请放入 model.parameters() 和 state_dict()
            torch.ones(hidden_size)
        )

    def forward(self, hidden_states):
        mean_square = hidden_states.pow(2).mean( #平方
            dim=-1,
            keepdim=True, #keepdim=True 保留最后一个长度为 1 的维度
        )

        inverse_rms = torch.rsqrt( #计算均方根的倒数
            mean_square + self.eps
        )

        normalized_hidden_states = (
            hidden_states * inverse_rms
        )

        output = (
            normalized_hidden_states
            * self.weight
        )

        return output

#比起 ReLU，现代大模型经常使用 SwiGLU。ReLU 会直接切断所有负数，而 SwiGLU 的激活函数更平滑，会保留一部分负值信息，使得梯度更稳定。
#SwiGLU(x) = SiLU(xW_gate) * (xW_up)，两条分支来自同一个输入经过两组独立线性投影。
#                   ┌→ gate_projection → SiLU ─┐
# 输入 hidden_states                           × → down_projection
#                   └→ up_projection ──────────┘
# gate = SiLU(xW_gate)
# up   = xW_up
# hidden = gate ⊙ up
# 其中 ⊙ 表示逐元素乘法，不是矩阵乘法。
# gate_projection：决定哪些特征应该通过，以及通过多少
# up_projection：提供真正需要传递的特征内容
class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
    ):
        super().__init__()

        self.gate_projection = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        )

        self.up_projection = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        )

        self.activation = nn.SiLU()

        self.down_projection = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, hidden_states):
        gate_values = self.activation(
            self.gate_projection(
                hidden_states
            )
        )

        up_values = self.up_projection(
            hidden_states
        )

        gated_values = (
            gate_values * up_values
        )

        output = self.down_projection(
            gated_values
        )

        return output

class FeedForward(nn.Module): #FFN 前馈神经网络
    def __init__(
        self,
        hidden_size,
        intermediate_size, #隐藏层
    ):
        super().__init__()

        #FFN的结构是：Linear → Activation → Linear
        #第一次 Linear：[1, 4, 4] → [1, 4, 16]
        self.input_projection = nn.Linear( #nn.Linear 只处理最后一个维度，因此它不会把不同 Token 混合
            in_features=hidden_size,
            out_features=intermediate_size,
        )

        self.activation = nn.ReLU() #激活函数引入非线性，ReLU(x) = max(0, x) [1, 4, 16]

        #第二次 Linear：[1, 4, 16] → [1, 4, 4]
        self.output_projection = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
        )

    def forward(self, hidden_states):
        intermediate_states = self.input_projection(
            hidden_states
        )

        activated_states = self.activation(
            intermediate_states
        )

        output = self.output_projection(
            activated_states
        )

        return output

# 多头注意力
# → 残差连接
# → LayerNorm
# → FFN
# → 残差连接
# → LayerNorm
#__init__() 负责准备组件，forward() 负责规定组件如何运行
class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout=0.1,
    ):
        super().__init__()

        #创建并登记模块，没有进行任何数据计算
        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.attention_dropout = nn.Dropout(
            p=dropout
        )

        #第一次归一化
        self.attention_layer_norm = nn.LayerNorm(hidden_size) #层归一化，让均值回归 0，方差回归 1
        # normalized =(x - mean) / sqrt(variance + eps)
        # output = normalized × gamma + beta
        # eps 防止除以零。
        # gamma 是可学习的缩放参数，形状 [4]。
        # beta 是可学习的平移参数，形状 [4]。
        # 初始时 gamma=1、beta=0。

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.feed_forward_dropout = nn.Dropout(
            p=dropout
        )

        #第二次归一化
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        is_causal=False,
    ):
        attention_output, attention_weights = (
            self.self_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
        )
        #第一次残差连接：保留的是进入 Attention 之前的 x0
        #残差连接：旧特征 + 注意力学到的新信息
        #残差连接有两个重要作用：
        # 1. 保留原来的 Token 信息，避免子层把旧信息完全覆盖。
        # 2. 给梯度提供更直接的传播路径，使深层网络更容易训练。
        # 这也是多头注意力最后必须恢复到 hidden_size=4 的原因之一，否则无法与原输入相加。
        hidden_states = self.attention_layer_norm( #x0 更新成了 x1
            hidden_states
            + self.attention_dropout(
                attention_output
            )
        )
        #这里采用嵌套写法，下面的FFN也同理
        # # 第一步：Attention 已经学习出的新信息
        # attention_output

        # # 第二步：对新信息做 Dropout
        # dropped_attention_output = self.attention_dropout(
        #     attention_output
        # )

        # # 第三步：加回 Attention 之前的原信息
        # residual_output = (
        #     hidden_states + dropped_attention_output
        # )

        # # 第四步：对残差结果做归一化
        # hidden_states = self.attention_layer_norm(
        #     residual_output
        # )

        feed_forward_output = self.feed_forward(
            hidden_states
        )

        ##第二次残差连接：保留的是进入 FFN 之前的 x1，不是最初的 x0
        output = self.feed_forward_layer_norm(
            hidden_states
            + self.feed_forward_dropout(
                feed_forward_output
            )
        )

        return output, attention_weights

# Post-LN：
# x ─→ Attention ─→ Add ─→ Norm ─→ 输出
# └─────────────────↑
# TransformerBlock即为Post-LN结构。残差连接的输出会经过归一化后再传给下一层。

# Pre-LN：
# x ─→ Norm ─→ Attention ─→ Add ─→ 输出
# └────────────────────────↑
# Pre-LN 中，原始的 x 可以沿残差路径直接传到输出，梯度也能沿这条路径直接反向传播，所以训练很多层时通常更稳定。
class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.attention_dropout = nn.Dropout(
            p=dropout
        )

        self.attention_layer_norm = nn.LayerNorm(
            hidden_size
        )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.feed_forward_dropout = nn.Dropout(
            p=dropout
        )

        self.feed_forward_layer_norm = nn.LayerNorm(
            hidden_size
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        is_causal=False,
    ):
        normalized_attention_input = (
            self.attention_layer_norm(
                hidden_states
            )
        )

        attention_output, attention_weights = (
            self.self_attention(
                hidden_states=normalized_attention_input,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
        )

        hidden_states = (
            hidden_states
            + self.attention_dropout(
                attention_output
            )
        )

        normalized_ffn_input = (
            self.feed_forward_layer_norm(
                hidden_states
            )
        )

        feed_forward_output = self.feed_forward(
            normalized_ffn_input
        )

        output = (
            hidden_states
            + self.feed_forward_dropout(
                feed_forward_output
            )
        )

        return output, attention_weights

#现代 Transformer Block：Pre-Norm + RMSNorm + SwiGLU
class ModernTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout=0.0,
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.attention_norm = RMSNorm(
            hidden_size=hidden_size
        )

        self.attention_dropout = nn.Dropout(
            p=dropout
        )

        self.feed_forward = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.feed_forward_norm = RMSNorm(
            hidden_size=hidden_size
        )

        self.feed_forward_dropout = nn.Dropout(
            p=dropout
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        is_causal=False,
    ):
        normalized_attention_input = (
            self.attention_norm(
                hidden_states
            )
        )

        attention_output, attention_weights = (
            self.self_attention(
                hidden_states=normalized_attention_input,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
        )

        hidden_states = (
            hidden_states
            + self.attention_dropout(
                attention_output
            )
        )

        normalized_ffn_input = (
            self.feed_forward_norm(
                hidden_states
            )
        )

        feed_forward_output = (
            self.feed_forward(
                normalized_ffn_input
            )
        )

        output = (
            hidden_states
            + self.feed_forward_dropout(
                feed_forward_output
            )
        )

        return output, attention_weights


def main():
    torch.manual_seed(42)

    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1, 0]],
        dtype=torch.long,
    )

    transformer_block = TransformerBlock(
        hidden_size=4,
        num_heads=2,
        intermediate_size=16,
        dropout=0.1,
    )

    transformer_block.eval()

    pre_norm_block = PreNormTransformerBlock(
        hidden_size=4,
        num_heads=2,
        intermediate_size=16,
        dropout=0.1,
    )

    pre_norm_block.load_state_dict( #把这些参数加载到另一个结构对应的模型中
        transformer_block.state_dict() #获得模型所有参数，例如：query_projection.weight、LayerNorm.weight、LayerNorm.bias、FFN 中的权重和偏置
    )
    #这样做是为了控制变量：两个模型的参数完全相同，区别只有 LayerNorm 的位置。这也是以后加载预训练模型权重的基础机制。
    
    pre_norm_block.eval()

    modern_block = ModernTransformerBlock(
        hidden_size=4,
        num_heads=2,
        intermediate_size=16,
        dropout=0.0,
    )

    final_norm = RMSNorm(
        hidden_size=4
    )

    modern_block.eval()
    final_norm.eval()

    with torch.no_grad():
        output, attention_weights = (
            transformer_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                is_causal=True,
            )
        )

        pre_norm_output, _ = pre_norm_block( #这里的 _ 表示本次不使用返回的注意力权重
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
        )

        modern_output, _ = (
            modern_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                is_causal=True,
            )
        )

        modern_final_output = final_norm(
            modern_output
        )

    print("input shape:", hidden_states.shape)
    print("output shape:", output.shape)
    print("output dtype:", output.dtype)
    print("output device:", output.device)

    print(
        "\nattention weights shape:",
        attention_weights.shape,
    )
    print(
        "attention weights:",
        attention_weights,
    )
    print(
        "row sums:",
        attention_weights.sum(dim=-1),
    )
    print(
        "padded key weights:",
        attention_weights[:, :, :, -1],
    )
    print(
        "future key weights:",
        torch.triu(
            attention_weights,
            diagonal=1,
        ),
    )

    print("\noutput:", output)

    print(
        "\ntoken means:",
        output.mean(dim=-1), #对每个 Token 最后的四个特征求平均
    )

    print(
        "token variances:",
        output.var( #对每个 Token 最后的四个特征计算总体方差。由于最后经过 LayerNorm，预期会看到：每个 Token 的均值接近 0，每个 Token 的方差接近 1
            dim=-1,
            unbiased=False,
        ),
    )

    print(
        "\npost-norm token means:",
        output.mean(dim=-1),
    )

    print(
        "post-norm token variances:",
        output.var(
            dim=-1,
            unbiased=False,
        ),
    )

    print(
        "\npre-norm token means:",
        pre_norm_output.mean(dim=-1),
    )

    print(
        "pre-norm token variances:",
        pre_norm_output.var(
            dim=-1,
            unbiased=False,
        ),
    )

    print(
        "\nmodern output shape:",
        modern_output.shape,
    )

    print(
        "modern output RMS:",
        torch.sqrt(
            modern_output.pow(2).mean(
                dim=-1
            )
        ),
    )

    print(
        "after final RMSNorm:",
        torch.sqrt(
            modern_final_output.pow(2).mean(
                dim=-1
            )
        ),
    )


if __name__ == "__main__":
    main()