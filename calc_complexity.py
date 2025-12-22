import torch
from thop import profile, clever_format
import time
from model import UNET   # ← 确保这个路径对，和你 train.py 导入方式一致

# ---------------------
# 1. 创建模型（不用加载权重）
# ---------------------
model = UNET(in_channels=1, out_channels=1).cuda()  # 如果你训练时是3通道，就写3
model.eval()

# ---------------------
# 2. 定义模型输入大小，例如 1024×1024
# ---------------------
dummy_input = torch.randn(1, 1, 1024, 1024).cuda()  # batch=1，可换为3通道

# ---------------------
# 3. 计算 Params & FLOPs
# ---------------------
macs, params = profile(model, inputs=(dummy_input,), verbose=False)
flops = macs * 2  # 论文常用 FLOPs=MACs×2

macs_str, params_str = clever_format([macs, params], "%.3f")
flops_str, _ = clever_format([flops, params], "%.3f")

print(f"模型参数量 Params: {params_str}")
print(f"MACs: {macs_str}")
print(f"FLOPs: {flops_str}")

# ---------------------
# 4. 计算推理速度 FPS
# ---------------------
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    total_time = time.time() - start
    fps = 50 / total_time

print(f"推理速度 FPS: {fps:.2f}")
