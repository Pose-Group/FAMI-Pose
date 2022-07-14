"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/4/16
    Description：
        将训练的参数文件，保存为 pretrain的格式
"""
import torch

model_name = "out_model"

source_pth_path = "/media/T/fengrunyang/MPPE/output/HRNet-Train/PoseTrack17/bbox_1.25_rot_45_scale_0.65-1.35/checkpoints/epoch_0_state.pth"
des_pth_path = f"/media/T/chenhaoming/Code/DcPose_supp_files/pretrained_models/{model_name}.pth"

checkpoint = torch.load(source_pth_path)
state_dict = checkpoint['state_dict']
## 删除多GPU训练导致的多余前缀
state_dict = {k.replace('module.', '') if k.find('module.') == 0 else k: v for k, v in state_dict.items()}
print(f"{des_pth_path} save finish")
torch.save(state_dict, des_pth_path)
