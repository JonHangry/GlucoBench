import torch

print("可用 GPU 数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(torch.cuda.is_available())
# from mamba_ssm import Mamba

# import torch
# print("PyTorch 版本:", torch.__version__)
# print("CUDA 编译版本:", torch.version.cuda)
# print("cuDNN 版本:", torch.backends.cudnn.version())
# print("GPU 是否可用:", torch.cuda.is_available())
# print("GPU 数量:", torch.cuda.device_count())
# print(torch._C._GLIBCXX_USE_CXX11_ABI)
# if torch.cuda.is_available():
#     print("当前 GPU 名称:", torch.cuda.get_device_name(0))

# pip install /home/limu-pytorch/Downloads/causal_conv1d-1.5.0.post8+cu12torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# pip install /home/limu-pytorch/Downloads/mamba_ssm-2.2.4+cu12torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
