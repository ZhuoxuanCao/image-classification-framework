import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import math

from utils import save_best_model
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from datetime import datetime
from dataset import get_data_loaders
from model import ResNet34, ResNet50
from config import get_config

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

def get_model(model_type, num_classes, pretrained=True):
    if model_type == 'resnet34':
        return ResNet34(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'resnet50':
        return ResNet50(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# def save_best_model(model, save_path, best_loss, current_loss):
#     if current_loss < best_loss[0]:
#         # 删除旧文件
#         if os.path.exists(save_path):
#             os.remove(save_path)
#         torch.save(model.state_dict(), save_path)
#         best_loss[0] = current_loss

def main():
    opt = get_config()  # 获取所有配置参数（dict）
    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, val_loader, class_names = get_data_loaders(
        opt['dataset'],
        batch_size=opt['batch_size'],
        val_split=0.2,
        input_size=224,
        num_workers=2,
        shuffle=True,
        seed=42
    )

    # 模型
    model = get_model(opt['model_type'], num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=opt['learning_rate'])

    if opt['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt['learning_rate'])
    elif opt['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("Unsupported optimizer: {}".format(opt['optimizer']))

    total_steps = opt['epochs'] * len(train_loader)

    # warmup + 余弦退火
    def lr_lambda(current_step):
        warmup_steps = int(0.1 * total_steps)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer = SummaryWriter(log_dir=opt['log_dir'])
    best_loss = [float('inf')]
    save_path = os.path.join(opt['model_save_dir'], 'best_model.pth')

    global_step = 0
    for epoch in range(opt['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[Epoch {epoch+1}/{opt['epochs']}] Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{opt['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        print(f"[Epoch {epoch+1}/{opt['epochs']}] Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

        writer.add_scalar('Val/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_epoch_acc, epoch)
        # save_best_model(model, save_path, best_loss, val_epoch_loss)
        exp_info = f"lr{opt['learning_rate']}_bs{opt['batch_size']}"

        save_best_model(
            model=model,
            save_dir=opt['model_save_dir'],
            model_type=opt['model_type'],
            epoch=epoch,
            current_loss=val_epoch_loss,
            best_loss=best_loss,
            run_id=run_id,
            extra_info=exp_info
        )

    print("训练结束。最小Val Loss:", best_loss[0])
    writer.close()

if __name__ == "__main__":
    main()
