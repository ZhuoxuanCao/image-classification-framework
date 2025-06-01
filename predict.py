# python predict.py --input ./image_test/BlueUp3.jpg --model_type resnet34 --save_path ./checkpoints/20250601_130141_resnet34_epoch23_val0.0005_lr0.001_bs16_20250601_130938.pth

import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import ResNet34, ResNet50
from config import get_config


def load_model(opt, device):
    if opt['model_type'] == 'resnet34':
        model = ResNet34(num_classes=opt['num_classes'], pretrained=False)
    elif opt['model_type'] == 'resnet50':
        model = ResNet50(num_classes=opt['num_classes'], pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {opt['model_type']}")
    model.load_state_dict(torch.load(opt['save_path'], map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # 增加batch维度 [1, C, H, W]


def main():
    opt = get_config()
    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else "cpu")
    model = load_model(opt, device)

    # 获取类别名（按训练时顺序，通常是按文件夹排序）
    class_names = sorted(os.listdir(opt['dataset']))

    # # 支持命令行传入文件夹或单张图片
    # parser = argparse.ArgumentParser(description="Predict with trained ResNet model.")
    # parser.add_argument('--input', type=str, required=True, help='Path to image or directory')
    # args = parser.parse_args()
    # input_path = args.input

    input_path = opt['input']

    image_list = []
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_list.append(os.path.join(input_path, fname))
    elif os.path.isfile(input_path):
        image_list = [input_path]
    else:
        print("输入路径无效")
        return

    for img_path in image_list:
        img_tensor = preprocess_image(img_path, opt['input_size']).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred = prob.argmax(dim=1).item()
            print(f"{os.path.basename(img_path)} --> 预测类别: {class_names[pred]}，概率: {prob[0, pred]:.4f}")


if __name__ == '__main__':
    main()
