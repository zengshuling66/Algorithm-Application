import torch

def demo_scalar_autograd():
    #x是我们直接创建、需要训练的 Tensor，称为叶子节点（leaf tensor）。
    x = torch.tensor(
        2.0,
        requires_grad=True, #后续凡是由这个Tensor参与的运算，PyTorch 都需要记录，以便反向计算梯度。
    )

    y = x ** 2 + 3 * x + 1 #这组运算关系称为计算图

    print("x:", x)
    print("y:", y)
    print("backward 前的 x.grad:", x.grad)

    y.backward() #反向传播：PyTorch 会自动计算 y 对 x 的梯度，并将结果存储在 x.grad 中

    expected_gradient = 2 * x.item() + 3 #x.item() 将只包含一个元素的 Tensor 转换成普通 Python 数字
    #手工验证 PyTorch 算得是否正确

    print("backward 后的 x.grad:", x.grad)
    print("手工计算的梯度:", expected_gradient)


def demo_gradient_accumulation():
    x = torch.tensor(
        2.0,
        requires_grad=True,
    )

    first_loss = x ** 2
    first_loss.backward()
    #当 x=2 时：d(x²)/dx = 2x = 4，所以 x.grad 是 4。

    print("\n第一次反向传播后的梯度:", x.grad)

    second_loss = 3 * x
    second_loss.backward()
    #第二个梯度是 3，但 PyTorch 默认不会覆盖旧梯度，而会累加：4 + 3 = 7

    print("第二次反向传播后的累计梯度:", x.grad)

    x.grad.zero_() #清零梯度。在神经网络训练里，如果忘记清零，当前批次梯度会与之前批次不断叠加，通常导致错误的参数更新。
    print("清零后的梯度:", x.grad)


def train_linear_model():
    torch.manual_seed(42)

    inputs = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]]
    )

    targets = 2 * inputs + 1

    weight = torch.randn( #torch.randn() 用来从标准正态分布中随机生成浮点数
        1, #这里的 1 表示 Tensor 的形状，而不是数值为1
        requires_grad=True,
    )

    bias = torch.zeros( #torch.zeros() 会创建一个所有元素都为 0 的 Tensor
        #偏置通常可以初始化为0，因为不同神经元的差异主要由随机初始化的权重产生，把偏置设为0一般不会造成权重全部相同带来的对称性问题。
        1, #同样的 1 代表形状
        requires_grad=True,
    )
    # inputs:     [4, 1]
    # weight:     [1]
    # bias:       [1]
    # predictions: [4, 1]
    #这里再次使用了广播机制：weight 和 bias 被自动应用到四个样本上。

    learning_rate = 0.05 #learning_rate 是学习率，控制每次更新的步长。
    num_steps = 200

    for step in range(num_steps):
        predictions = inputs * weight + bias #模型预测公式

        errors = predictions - targets
        loss = (errors ** 2).mean() #均方误差 MSE：所有样本预测误差平方的平均值
        #平方有两个作用：正负误差不会互相抵消；偏差越大，惩罚增长越快。

        loss.backward() #loss 对 weight和bias的偏导数，结果分别存入：weight.grad、bias.grad
        # eᵢ = wxᵢ + b - yᵢ；loss=ei**2
        # lossᵢ 对 w 的导数
        # = 2eᵢ × xᵢ = 2 × 误差 × 输入
        #当前有四个样本，因此使用平均损失：loss = (e₁² + e₂² + e₃² + e₄²) / 4；每一项对 weight 的梯度都是：2eᵢxᵢ
        #所以总梯度是：weight.grad = (1/4) × [2e₁x₁ + 2e₂x₂ + 2e₃x₃ + 2e₄x₄] = (1/2) × [e₁x₁ + e₂x₂ + e₃x₃ + e₄x₄]

        # lossᵢ 对 b 的导数
        # = 2eᵢ × 1
        # = 2eᵢ
        #因此四个样本的平均梯度是：bias.grad = (2/4) × (e₁ + e₂ + e₃ + e₄)

        with torch.no_grad(): #这是因为更新参数只是优化操作，不属于模型的前向计算。
            #若 PyTorch继续记录更新过程，就会建立无用的新计算图，还可能触发“对需要梯度的叶子 Tensor 原地修改”的错误。
            #参数更新公式是：新参数 = 旧参数 - 学习率 × 梯度
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad
            #梯度指向函数上升最快的方向。训练希望损失下降，因此沿梯度的反方向更新。

        weight.grad.zero_()
        bias.grad.zero_()

        if step % 50 == 0 or step == num_steps - 1:
            print(
                f"step={step:3d}, "
                f"loss={loss.item():.6f}, "
                f"weight={weight.item():.4f}, "
                f"bias={bias.item():.4f}"
            )


def main():
    demo_scalar_autograd()
    demo_gradient_accumulation()
    train_linear_model()


if __name__ == "__main__":
    main()