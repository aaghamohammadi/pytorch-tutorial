import torch

# Check PyTorch installation and CUDA support
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Demonstrate CPU vs GPU tensor operations
x_cpu = torch.rand(3, 3)
print("\nTensor on CPU:\n", x_cpu)

# Only move to GPU if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
x_device = x_cpu.to(device)
print(f"\nTensor on {device.upper()}:\n", x_device)

# Show tensor device location
print(f"\nTensor device: {x_device.device}")
