import torch
# 加载完整模型（实际上是字典）
checkpoint = torch.load('/home/share/FM_Code/Stage1/PanDerm/Model_Weights/original/panderm_bb_data6_checkpoint-499.pth')

# 检查字典结构
print(f"Keys in checkpoint: {checkpoint.keys()}")

# 提取模型权重（假设键名为'model'或'state_dict'）
if 'model' in checkpoint:
    model_weights = checkpoint['model']
elif 'state_dict' in checkpoint:
    model_weights = checkpoint['state_dict']
else:
    # 如果本身就是权重字典，则直接使用
    model_weights = checkpoint

# 保存只有权重的版本
torch.save(model_weights, '/home/share/FM_Code/Stage1/PanDerm/Model_Weights/panderm_bb_data6_checkpoint-499.pth')

print("已成功保存纯权重文件")