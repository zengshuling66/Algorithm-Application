import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_training_demo import LinearRegressionModel


def create_dataloaders():
    inputs = torch.linspace( #在 -1 到 1 之间均匀生成20个数，原始形状为：[20]
        -1.0,
        1.0,
        steps=20,
    ).unsqueeze(dim=1) #指定位置插入一个长度为1的新维度：[20] → [20, 1]

    noise = torch.randn_like(inputs) * 0.05 #randn_like(inputs)：生成与 inputs 形状相同的随机数
    #* 0.05：把噪声幅度缩小；加噪声是为了模拟真实数据不会完美落在一条直线上。
    targets = 2 * inputs + 1 + noise

    # 训练集：前16条
    train_inputs = inputs[:16]
    train_targets = targets[:16]

    # 验证集：后4条
    val_inputs = inputs[16:]
    val_targets = targets[16:]

    #TensorDataset把多个Tensor按第一维对齐，这要求两个Tensor第一维长度相同
    #当访问：train_dataset[0]，会返回：第0条input、第0条target，长度都是16
    train_dataset = TensorDataset(
        train_inputs,
        train_targets,
    )

    val_dataset = TensorDataset(
        val_inputs,
        val_targets,
    )

    train_loader = DataLoader(
        train_dataset, #训练集共16条
        batch_size=4, #每批4条，即每个epoch产生4个batch
        shuffle=True, #每个epoch重新打乱训练样本顺序，可以降低样本原始排列对训练的影响。验证集：shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
    )

    return train_loader, val_loader

# 一个epoch = 遍历全部16条训练数据
# 一个epoch包含4个batch
# 每处理一个batch，optimizer.step()一次

def evaluate_model(
    model,
    data_loader,
    loss_fn,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)

            batch_size = batch_inputs.size(dim=0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            #nn.MSELoss()默认返回当前batch的平均损失，如果不同batch大小不一样，不能直接对各batch的平均损失再次求平均。
            #因此先还原成这个batch的损失总和：loss.item() * batch_size，累计所有样本后再除以总样本数，这样即使最后一个batch不足4条，结果仍然正确。

    average_loss = total_loss / total_samples

    return average_loss


def train_model():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = create_dataloaders()

    model = LinearRegressionModel().to(device)
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets) #计算当前batch损失

            loss.backward()
            optimizer.step()

            batch_size = batch_inputs.size(dim=0)

            total_train_loss += loss.item() * batch_size #累计训练损失
            total_train_samples += batch_size

        average_train_loss = (
            total_train_loss / total_train_samples
        )

        average_val_loss = evaluate_model(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:3d}, "
                f"train_loss={average_train_loss:.6f}, "
                f"val_loss={average_val_loss:.6f}"
            )

    return model


def main():
    model = train_model()

    print(
        "\nlearned weight:",
        model.linear.weight.item(),
    )
    print(
        "learned bias:",
        model.linear.bias.item(),
    )


if __name__ == "__main__":
    main()