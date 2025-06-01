import argparse
import os


def parse_option():
    parser = argparse.ArgumentParser('Image Classification with ResNet and Configurable Optimizer')

    # 训练核心参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'],
                        help='Optimizer to use (adam or sgd)')

    # 设备和路径
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--dataset', type=str, default='./image_train', help='Path to dataset (root folder)')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints',
                        help='Directory to save best model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory for TensorBoard logs')

    # 模型和数据相关
    parser.add_argument('--model_type', type=str, default='resnet34', choices=['resnet34', 'resnet50'],
                        help='Model type')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set split ratio')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use ImageNet pretrained weights (default True)')

    # 学习率 warmup/调度
    parser.add_argument('--warmup_epochs', type=float, default=2.0, help='Warmup epochs for LR scheduler')

    parser.add_argument('--input', type=str, required=False, default=None,
                        help='Path to image or directory for prediction')
    parser.add_argument('--save_path', type=str, default=None, help='Path to model weight for prediction (optional)')

    args = parser.parse_args()

    args.num_classes = 5

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 只有在 save_path 未指定时才生成默认路径
    if args.save_path is None:
        args.save_path = os.path.join(args.model_save_dir, 'best_model.pth')

    return vars(args)


def get_config():
    return parse_option()
