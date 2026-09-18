import torch


def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(tensor)
    print("shape:", tensor.shape) #shape 表示每个维度的长度
    print("dtype:", tensor.dtype) #dtype 是 data type，即数据类型
    print("device:", tensor.device) #表示 Tensor 当前存储在什么设备上
    print("ndim:", tensor.ndim) #表示 Tensor 的维度数
    print("numel:", tensor.numel()) #表示 Tensor 中元素的总数
    #上面的是属性，所以不加圆括号；numel() 是方法，所以要加圆括号。

def demo_shape_operations():
    tensor_3d = torch.arange(
        24, #0-23
        dtype=torch.float32,
    ).reshape(2, 3, 4)
    # 2：batch_size 样本
    # 3：sequence_length token
    # 4：hidden_size 特征

    print_tensor_info("tensor_3d", tensor_3d)

    #整数索引会删除该维，切片会保留该维。
    first_sample = tensor_3d[0] #0表示取 batch 维中的第一个具体样本；[2,3,4] → [3,4]
    first_sample_keep_dim = tensor_3d[0:1] #使用切片保留 batch 维；[2,3,4] → [1,3,4]；0:1 表示从下标0开始，到下标1之前停止，因此仍然只取第一条数据
    #模型通常要求保留 batch 维，因此第二种写法经常更加安全

    first_token_each_sample = tensor_3d[:, 0, :] #中括号中的三个位置分别对应：tensor_3d[batch维, sequence维, hidden维]
    # 第一个: 选择所有样本
    # 中间的 0：选择每个样本的第一个token
    # 最后的: 选择该token的所有hidden特征
    #[2,3,4] → [2,4]：sequence 维使用了整数索引 0，所以该维消失

    second_feature = tensor_3d[:, :, 1] #取每个 token 的第二个特征；hidden 维因为整数索引而消失：[2,3,4] → [2,3]
    second_feature_keep_dim = tensor_3d[:, :, 1:2] #使用切片保留 hidden 维；[2,3,4] → [2,3,1]

    #上下作用一样，拆成多行是为了提高可读性
    print("\nfirst_sample shape:", first_sample.shape)
    print(
        "first_sample_keep_dim shape:",
        first_sample_keep_dim.shape,
    )
    print(
        "first_token_each_sample shape:",
        first_token_each_sample.shape,
    )
    print("second_feature shape:", second_feature.shape)
    print(
        "second_feature_keep_dim shape:",
        second_feature_keep_dim.shape,
    )

    flattened = tensor_3d.reshape(2, -1) #-1 表示：这一维的长度由 PyTorch 根据元素总数自动推断，一次 reshape() 中最多只能有一个 -1
    restored = flattened.reshape(2, 3, 4)
    same_values = torch.equal(tensor_3d, restored) #比较两个 Tensor：形状是否相同，每个位置的值是否相同，两个条件都满足才返回：True

    print("\nflattened shape:", flattened.shape)
    print("restored shape:", restored.shape)
    print("restored values unchanged:", same_values)

    #负数维度从右向左编号：-1：最后一维；-2：倒数第二维；-3：倒数第三维
    mean_without_keepdim = tensor_3d.mean(dim=-1) #dim=-1表示最后一维，即hidden维；每个token的4个特征求平均，最后一维默认消失；[2,3,4] → [2,3]
    mean_with_keepdim = tensor_3d.mean(
        dim=-1,
        keepdim=True, #keepdim=True被聚合的维度不删除，而是将长度变成1：[2,3,4] → [2,3,1]
    )

    #广播
    centered = tensor_3d - mean_with_keepdim
    #参与运算的形状：tensor_3d：[2,3,4]；mean_with_keepdim：[2,3,1]；
    #PyTorch 会自动将长度为1的维度扩展到与另一个 Tensor 相同的长度，从而实现广播机制：broadcasting
    #每个token的4个特征，都减去同一个平均值，这样每个token的特征就变成“以 0 为中心”的数列

    print(
        "\nmean_without_keepdim shape:",
        mean_without_keepdim.shape,
    )
    print(
        "mean_with_keepdim shape:",
        mean_with_keepdim.shape,
    )
    print(
        "centered mean:",
        centered.mean(dim=-1), #对“以 0 为中心”的数列的每个token的特征求平均，结果接近0
        # 预期接近：
        # [[0., 0., 0.],
        # [0., 0., 0.]]
    )

def demo_dimension_and_matrix_operations():
    token_vector = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
    )

    row_vector = token_vector.unsqueeze(dim=0) #unsqueeze 表示在指定位置插入一个长度为1的新维度；[4]->[1,4]
    three_dim_vector = row_vector.unsqueeze(dim=0) #[1,4]->[1,1,4]：这表示一批只有一条文本，文本只有一个 token，该 token 是四维向量

    restored_vector = (
        three_dim_vector
        .squeeze(dim=0) #删除第0维，但只有该维长度为1时才能删除
        .squeeze(dim=0) #再次删除当前第0维
    )
    #这里把 . 放在下一行，是 Python 的链式调用，与多行版本完全相同：three_dim_vector.squeeze(dim=0).squeeze(dim=0)
    #tensor.squeeze()会删除所有长度为1的维度，但存在风险，明确指定 dim 更安全

    print("\ntoken_vector shape:", token_vector.shape)
    print("row_vector shape:", row_vector.shape)
    print("three_dim_vector shape:", three_dim_vector.shape)
    print("restored_vector shape:", restored_vector.shape)

    hidden_states = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    transposed_states = hidden_states.transpose(1, 2) #transpose() 只交换两个指定维度
    # [2,3,4] → [2,4,3]
    # [B,S,H]  → [B,H,S]
    permuted_states = hidden_states.permute(2, 0, 1) #permute() 可以一次重新排列所有维度
    # 参数：(2,0,1)
    # 原第2维放到新第0维
    # 原第0维放到新第1维
    # 原第1维放到新第2维
    # 原形状：[2,3,4]；新形状：[4,2,3]

    print("\nhidden_states shape:", hidden_states.shape)
    print("transposed_states shape:", transposed_states.shape)
    print("permuted_states shape:", permuted_states.shape)

    matrix_a = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    matrix_b = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    matrix_product = torch.matmul(matrix_a, matrix_b) #矩阵乘法，中间两个维度必须相等，torch.matmul(a,b) 也可以写成：a @ b
    #a * b是逐元素乘法，不是矩阵乘法

    print("\nmatrix_a shape:", matrix_a.shape)
    print("matrix_b shape:", matrix_b.shape)
    print("matrix_product:", matrix_product)
    print("matrix_product shape:", matrix_product.shape)

    torch.manual_seed(42) #PyTorch的随机操作依赖随机数生成器。
    #设置相同随机种子后，CPU上的torch.randn()每次运行通常产生相同结果，方便调试、测试和复现实验，42没有特殊数学意义，只是常用示例值。

    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 3, 4)

    key_transposed = key.transpose(-2, -1) #交换最后两维，相当于转置 Key
    # query：          [2,3,4]
    # key_transposed： [2,4,3]
    attention_scores = torch.matmul(
        query,
        key_transposed,
    )
    #最后两个维度执行矩阵乘法：[3,4] @ [4,3] → [3,3]
    #前面的 batch 维保留：[2,3,4] @ [2,4,3] → [2,3,3]
    #结果中每个 [3,3] 矩阵表示：3个query token分别 与 3个key token计算点积
    #因此每个 token 都得到它与所有 token 的相关性分数。这就是 Attention 分数的核心来源：Q @ Kᵀ

    print("\nquery shape:", query.shape)
    print("key shape:", key.shape)
    print("key_transposed shape:", key_transposed.shape)
    print("attention_scores shape:", attention_scores.shape)

def main():
    #标量 0维 这一步将 Python 数据转换为 Tensor
    scalar = torch.tensor(3.0)

    #向量 1维
    vector = torch.tensor(
        [1.0, 2.0, 3.0], #使用三个浮点数创建一维 Tensor
        dtype=torch.float32, #float32 表示每个数字使用32位浮点数，是深度学习中最常见的数据类型之一
    )

    #矩阵 2维
    matrix = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    #调用打印函数
    print_tensor_info("scalar", scalar)
    print_tensor_info("vector", vector)
    print_tensor_info("matrix", matrix)

    batch = torch.arange( #调用 torch.arange() 创建等间隔的一维 Tensor
        12, #相当于：torch.arange(0, 12)    0, 1, 2, ..., 11    shape = [12]
        dtype=torch.float32,
    ).reshape(3, 4) #reshape() 可以改变形状，不能改变元素数量和排列顺序
    #[[ 0,  1,  2,  3],
    #[ 4,  5,  6,  7],
    #[ 8,  9, 10, 11]]

    print_tensor_info("batch", batch)
    print("第一行:", batch[0])
    print("第二列:", batch[:, 1]) #batch[行, 列]；: 表示选择所有行；1 表示选择下标为1的列，也就是第二列；输出tensor([1., 5., 9.])
    print("每一行的平均值:", batch.mean(dim=1)) #dim=0：三行这一维；dim=1：四列这一维，表示对每一行中的四个元素求平均

    hidden_states = torch.randn(2, 4, 8) #randn根据标准正态分布生成随机浮点数：均值约为0，标准差约为1
    #直接指定 Tensor 的形状：[2, 4, 8]：batch_size = 2，sequence_length = 4，hidden_size = 8
    print_tensor_info("hidden_states", hidden_states) #hidden_states指模型中每个token当前的向量表示。后续经过Attention和前馈网络，这些数值会不断变化

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("\na + b:", a + b)
    print("a * b:", a * b)
    print("a 和 b 的点积:", torch.dot(a, b))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    hidden_states = hidden_states.to(device) #原来的hidden_states在CPU， .to(device)创建或返回位于目标设备上的Tensor，结果重新赋值给 hidden_states

    print("\n当前计算设备:", device) #cuda 表示使用默认 CUDA 设备
    print("移动后的 device:", hidden_states.device) #cuda:0 表示实际使用编号为0的第一张 GPU

    demo_shape_operations()

    demo_dimension_and_matrix_operations()


if __name__ == "__main__":
    main()