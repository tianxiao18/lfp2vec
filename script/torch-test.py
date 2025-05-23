import torch

print(torch.__file__)
print(torch.__version__)

# How many GPUs are there?
print(torch.cuda.device_count())

# Get the name of the current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Is PyTorch using a GPU?
print(torch.cuda.is_available())