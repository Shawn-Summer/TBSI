import torch
print("PyTorch 能识别的 GPU 数量：", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"逻辑 GPU {i}：", torch.cuda.get_device_name(i))
