import torch
print(f'CUDA Availability: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
print(f'CUDA Version: {torch.version.cuda}')