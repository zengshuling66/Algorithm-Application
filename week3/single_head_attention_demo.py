import math
import torch
from torch import nn


#Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V
# 输入：
# query：[B,Sq,dₖ]
# key：  [B,Sk,dₖ]
# value：[B,Sk,dᵥ]
# Self-Attention 中 Sq 和 Sk 通常相等

# 输出：
# context：          [B,Sq,dᵥ]
# attention_weights：[B,Sq,Sk]
def scaled_dot_product_attention(query, key, value, attention_mask=None, is_causal=False): 
    # attention_mask=None是一个可选参数：
    # - 调用方传入 Mask：执行 Padding 屏蔽。
    # - 调用方不传：值为 None，保持之前的计算方式。
    key_transposed = key.transpose(-2, -1) #交换最后两个维度，为了沿 dₖ 做点积

    attention_scores = torch.matmul( #QKᵀ，[B, Sq, d_k] @ [B, d_k, Sk] -> [B, Sq, Sk]
        query,
        key_transposed,
    )

    scale = math.sqrt(query.size(-1)) #读取最后一维长度，维度越大，点积数值通常波动越大，Softmax 就容易变得非常极端，所以需要开根号。
    scaled_scores = attention_scores / scale #除以 √dₖ，Tensor 除以一个标量时，PyTorch 会把这个标量用于所有元素

    if attention_mask is not None: #它表示：只有调用者真正提供了 Mask，才进入屏蔽逻辑
        key_padding_mask = (
            attention_mask.unsqueeze(dim=1) == 0 #attention_mask.shape：[1,4] → [1,1,4]
            # 三个维度现在分别表示：[batch, query占位维, key位置]
            #Mask    [1,1,4]
            #Scores  [1,4,4]
            #           ↑
            #       长度 1 自动扩展成 4，变为 [1,4,4]，广播会把同一份 Key Mask 复制给所有 Query 行：广播不会计算新数值，只是逻辑上重复原来的值
            # 想遮住分数矩阵的列，就让 Mask 的最后一维对应列；缺少的 Query 行维放在中间并设为 1

            # 这里的 == 0 不是赋值，而是逐个检查张量中的元素是不是 0，它的类型是布尔类型true和false
            # masked_fill 根据布尔 Mask 选择位置：True 的位置被替换，False 的位置保留原值。
        )

        scaled_scores = scaled_scores.masked_fill(
            key_padding_mask,
            float("-inf"), #float("-inf") 表示负无穷，Softmax 时：e^(-∞) = 0，因此 Padding 权重会精确变成 0
            #不能把 Padding 分数填成 0，因为：e⁰ = 1，它仍会分走注意力权重。
        )

    if is_causal: #is_causal=False，表示普通的双向注意力，可以看前面和后面的 token；is_causal=True，表示因果注意力，不能看未来 token。
        query_length = query.size(-2) #获取 Query 数量，-2 是倒数第二维Sq=4
        key_length = key.size(-2) #获取 Key 数量，Sk = 4 = Sq

        causal_mask = torch.ones( #torch.ones()，布尔类型的 1 会表示为 True
            (query_length, key_length), #Causal Mask 是一个 [4,4] 矩阵
            dtype=torch.bool,
            device=query.device,
        )

        causal_mask = torch.triu( #triu 是 upper triangular，意思是取上三角部分
            causal_mask,
            diagonal=1, #diagonal=1，表示从主对角线的上一条对角线开始保留
            # False True  True  True
            # False False True  True
            # False False False True
            # False False False False
            # 这样只有未来位置是 True
            #如果使用：diagonal=0，主对角线也会被保留为 True，这样每个 Query 连自己都不能看，会出错
        )

        causal_mask = causal_mask.unsqueeze(
            dim=0, #数值不变，只是在最前面增加一个维度
            # 当前：
            # scaled_scores [1,4,4]
            # causal_mask   [1,4,4]
            # 两者可以直接对应。
        )

        #把未来分数替换为负无穷
        scaled_scores = scaled_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

    attention_weights = torch.softmax( #注意力权重，softmax(xᵢ) = e^xᵢ / Σe^xⱼ，每个 Query 对全部 Key 的权重和为 1
        scaled_scores,
        dim=-1, #dim=-1 表示沿最后一维，每个 Query 对全部 Sk 个 Key 计算 Softmax
    )

    context = torch.matmul( ## [B, Sq, Sk] @ [B, Sk, d_v] -> [B, Sq, d_v]，context 不是概率，因此里面的数不要求在 0 到 1 之间，它是 Value 向量的加权结果。
        attention_weights,
        value,
    )

    return context, attention_weights

class SingleHeadSelfAttention(nn.Module): #生成 Q/K/V
    def __init__(self, hidden_size): #hidden_size 表示每个 token 向量包含多少个特征
        super().__init__()

        #Query 投影层，它实现：Q = XWQᵀ；分别处理每个 token，不交换 token 信息
        self.query_projection = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False, #关闭偏置
        )

        #Key 投影层，它实现：K = XWKᵀ；分别处理每个 token，不交换 token 信息
        self.key_projection = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        )

        #Value 投影层，它实现：V = XWVᵀ；分别处理每个 token，不交换 token 信息
        self.value_projection = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        )

        # WQ：学习当前 token 想匹配什么
        # WK：学习当前 token 用什么特征接受匹配
        # WV：学习当前 token 真正提供什么信息
        # 如果三个变量都直接等于 hidden_states，模型无法分别学习“匹配方式”和“传递内容”。
        # 如果三个变量都调用同一个 nn.Linear 对象，它们就会共享完全相同的权重，也失去了明确分工。
        #Attention：通过 QKᵀ 和权重乘 V，让不同 token 交换信息

    def forward(self, hidden_states, attention_mask=None, is_causal=False):
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        context, attention_weights = (
            scaled_dot_product_attention( #使用 Q/K/V 计算权重和 context
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
        )

        return context, attention_weights

def main():
    torch.manual_seed(42) #固定 PyTorch 随机数生成器状态
    # 种子 42 没有数学含义，只是让每次运行产生相同权重，方便学习、debug 和复现实验。

    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0], #[PAD] 的演示向量
            ]
        ],
        dtype=torch.float32,
    )

    # 三层列表分别对应：
    # 最外层：1 条文本
    # 中间层：4 个 token
    # 最内层：每个 token 的 2 个特征

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0]
        ],
        dtype=torch.long,
    )

    attention_layer = SingleHeadSelfAttention( #三个 nn.Linear 创建时会随机初始化权重
        hidden_size=2, #必须等于输入最后一维 2。如果传入 hidden_size=4，线性层期望每个token有4个特征，但实际只有2个，运行时会发生矩阵维度错误。
    )

    print(attention_layer) #打印模型结构，预计显示三个线性层

    print("\nparameters:")
    for name, parameter in attention_layer.named_parameters():
        print(
            name,
            parameter.shape,
            parameter.requires_grad, #预计都是 True，表示这些权重可以通过 loss.backward() 计算梯度并被优化器更新
        )

    attention_layer.eval() #把模型切换为推理模式。当前只是演示前向传播，没有 loss 和训练数据，所以暂时不训练。

    with torch.no_grad():
        context, attention_weights = attention_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
        )

    print(
        "\nhidden states shape:",
        hidden_states.shape,
    )
    print(
        "hidden states dtype:",
        hidden_states.dtype,
    )
    print(
        "hidden states device:",
        hidden_states.device,
    )

    print(
        "\nattention mask shape:",
        attention_mask.shape,
    )
    print(
        "attention mask dtype:",
        attention_mask.dtype,
    )
    print(
        "attention mask device:",
        attention_mask.device,
    )
    print(
        "attention mask:",
        attention_mask,
    )

    print(
        "\nattention weights shape:",
        attention_weights.shape,
    )
    print(
        "attention weights:",
        attention_weights,
    )
    print(
        "weight row sums:",
        attention_weights.sum(dim=-1),
    )
    print(
        "padded key weights:",
        attention_weights[:, :, -1],
    )

    print("\ncontext shape:", context.shape)
    print("context dtype:", context.dtype)
    print("context device:", context.device)
    print("context:", context)

    print(
    "future key weights:",
    torch.triu(
        attention_weights,
        diagonal=1,
    ),
    )


if __name__ == "__main__":
    main()