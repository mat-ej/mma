import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print('Device name:', torch.cuda.get_device_name(0))
    print('Memory Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    if hasattr(torch.cuda, 'memory_reserved'):
        # Only for new PyTorch version
        print('Memory Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
