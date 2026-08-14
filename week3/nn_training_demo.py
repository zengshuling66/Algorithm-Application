import torch
from torch import nn #从PyTorch导入神经网络模块


#定义模型
#这里只说明模型由什么层组成、数据如何流动，还没有开始训练
class LinearRegressionModel(nn.Module): #定义自己的模型类，并继承 nn.Module
    def __init__(self): #构造方法，在创建模型时执行：model = LinearRegressionModel()
        super().__init__() #调用父类 nn.Module 的初始化方法，它会建立内部的参数、子模块和状态管理结构

        self.linear = nn.Linear( #创建线性层
            in_features=1, #每个样本有一个输入特征 x
            out_features=1, #每个样本输出一个预测值 y
        )
        # 输出 = 输入 × 权重转置 + 偏置
        #即使当前只有一个数字，PyTorch也统一使用矩阵表示，以便同一套代码支持批量计算、多输入和多输出。
        # 它会自动创建：
        # weight shape = [1, 1]
        # bias shape   = [1]

    def forward(self, inputs): #forward 定义数据进入模型后如何计算
        predictions = self.linear(inputs)
        return predictions


#准备训练数据
#数据与模型分开定义，是为了以后替换为 Dataset/DataLoader 时不需要修改模型。
def create_training_data():
    inputs = torch.tensor( #inputs：模型看到的输入
        [[1.0], [2.0], [3.0], [4.0]]
    )

    targets = 2 * inputs + 1 #targets：模型应该输出的正确答案

    return inputs, targets


#初始化训练所需对象
def train_model():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    inputs, targets = create_training_data()

    inputs = inputs.to(device)
    targets = targets.to(device)

    model = LinearRegressionModel() #model：怎么预测
    model = model.to(device)

    print(model)
    print(list(model.named_parameters()))

    #loss_fn：预测得有多差
    loss_fn = nn.MSELoss() #创建均方误差损失函数。它默认计算：所有预测误差平方的平均值

    #optimizer：怎样修改参数
    optimizer = torch.optim.SGD( #创建随机梯度下降优化器；这里虽然叫SGD，但当前每次使用全部4个样本，准确来说仍是全批量梯度下降。以后使用DataLoader的小批次后，才是常见的mini-batch SGD。
        model.parameters(), #把模型全部可训练参数交给优化器。
        lr=0.05, #学习率，控制每次更新的步长
    )

    num_epochs = 300

    #执行训练循环
    for epoch in range(num_epochs):
        #train → zero_grad → forward → loss → backward → step
        model.train() #把模型切换到训练模式。它不会自动开始训练，只会影响Dropout、BatchNorm等层的行为。当前只有Linear层，数值没有区别，但这是标准习惯。

        optimizer.zero_grad() #PyTorch默认会累加梯度，所以要清空上一轮梯度
        #zero_grad() 必须在本轮 backward() 之前，并且在上一轮 step() 之后。
        #放在开始的好处：即使上一轮中途异常或提前退出，也不会依赖上一轮是否成功清零。

        predictions = model(inputs) #前向传播，会进入模型的 forward()
        loss = loss_fn(predictions, targets) #计算预测与真实值之间的损失

        loss.backward() #反向传播，自动计算所有模型参数的梯度
        optimizer.step() #根据参数的 .grad 和学习率更新参数

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:3d}, "
                f"loss={loss.item():.6f}"
            )

    return model, inputs, targets


def main():
    model, inputs, targets = train_model()

    #训练完成后进入推理
    model.eval() #切换到评估模式。
    # 训练和推理是两个阶段：
    # 训练：需要计算梯度并更新参数。
    # 推理：只使用已经学好的参数，不更新。

    with torch.no_grad(): #关闭梯度记录，减少推理时的内存和计算开销
        predictions = model(inputs)

    #输出学习结果
    print("\npredictions:")
    print(predictions.cpu()) #把结果移回CPU，便于打印、转NumPy或保存

    print("\ntargets:")
    print(targets.cpu())

    print("\nlearned weight:", model.linear.weight.item())
    print("learned bias:", model.linear.bias.item())


if __name__ == "__main__":
    main()