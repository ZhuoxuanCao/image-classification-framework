import os
import torch
from datetime import datetime

def save_best_model(model, save_dir, model_type, epoch, current_loss, best_loss, run_id, extra_info=""):
    if current_loss < best_loss[0]:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 文件名以 run_id 开头，确保能正确删除旧文件
        filename = f"{run_id}_{model_type}_epoch{epoch + 1}_val{current_loss:.4f}"
        if extra_info:
            filename += f"_{extra_info}"
        filename += f"_{time_str}.pth"
        save_path = os.path.join(save_dir, filename)

        # 只删除本 run_id 的历史最优
        for f in os.listdir(save_dir):
            if f.startswith(f"{run_id}_{model_type}_") and f.endswith(".pth"):
                os.remove(os.path.join(save_dir, f))

        torch.save(model.state_dict(), save_path)
        print(f"保存新最优模型：{save_path}")
        best_loss[0] = current_loss
