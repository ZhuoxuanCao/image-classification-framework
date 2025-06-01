import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(
    data_dir,
    batch_size=32,
    val_split=0.2,
    input_size=224,
    num_workers=2,
    shuffle=True,
    seed=42
):
    """
    加载数据集并返回训练、验证的 DataLoader。

    参数:
        data_dir (str): 数据集根目录
        batch_size (int): 每个batch的图片数量
        val_split (float): 验证集占比
        input_size (int): 输入图片的尺寸
        num_workers (int): DataLoader进程数
        shuffle (bool): 是否在划分前打乱数据
        seed (int): 随机种子，保证划分一致

    返回:
        train_loader, val_loader, class_names
    """

    # 图像增强与标准化
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用ImageFolder自动按文件夹划分标签
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # 保证每次划分一致
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 验证集使用不同transform（去除随机增强）
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = full_dataset.classes

    return train_loader, val_loader, class_names

# 用于测试数据加载是否正常
if __name__ == "__main__":
    import torch

    data_dir = "./image_train"  # 修改为你的数据路径
    train_loader, val_loader, class_names = get_data_loaders(data_dir)
    print("类别名:", class_names)
    for images, labels in train_loader:
        print("图片 batch shape:", images.shape)
        print("标签 batch:", labels)
        break
