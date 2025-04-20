import numpy as np
import torch

# 1. Basic tensor creation from Python lists
print("\n--- Basic Tensor Creation ---")
basic_tensor = torch.tensor([1, 2, 3, 4, 5])
print("1D tensor:", basic_tensor)
print("Shape:", basic_tensor.shape)  # (5,)
print("Data type:", basic_tensor.dtype)  # torch.int64

# 2. Tensor with floating point numbers
float_tensor = torch.tensor([[1.1, 2.7, 3.4, 4.8, 5.2]])
print("\n--- Floating Point Tensor ---")
print("2D tensor:", float_tensor)
print("Shape:", float_tensor.shape)  # (1, 5)
print("Data type:", float_tensor.dtype)  # torch.float32

# 3. Converting NumPy array to tensor with specific dtype
print("\n--- NumPy to Tensor Conversion ---")
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
numpy_tensor = torch.tensor(numpy_array, dtype=torch.float64)
print("From NumPy:", numpy_tensor)
print("Shape:", numpy_tensor.shape)  # (2, 3)
print("Data type:", numpy_tensor.dtype)  # torch.float64

# 4. Scalar tensor
print("\n--- Scalar Tensor ---")
scalar_tensor = torch.tensor(2)
print("Scalar:", scalar_tensor)
print("Shape:", scalar_tensor.shape)  # ()

# 5. Automatic type inference
print("\n--- Type Inference ---")
mixed_tensor = torch.tensor([False, 5, 7.3])
print("Mixed types:", mixed_tensor)
print("Resulting type:", mixed_tensor.dtype)  # torch.float32

# 6. Factory methods for tensor creation
print("\n--- Factory Methods ---")
# Create tensor filled with zeros
zeros_tensor = torch.zeros((4, 5))
print("Zeros tensor:\n", zeros_tensor)

# Create tensor filled with ones
ones_tensor = torch.ones((3, 4))
print("\nOnes tensor:\n", ones_tensor)

# Create tensor with random integers
random_int_tensor = torch.randint(low=0, high=10, size=(3, 4))
print("\nRandom integers tensor:\n", random_int_tensor)

# Create tensor with random values from uniform distribution [0, 1)
random_uniform_tensor = torch.rand((3, 4))
print("\nRandom uniform tensor:\n", random_uniform_tensor)

# Create tensor with random values from normal distribution
random_normal_tensor = torch.randn((3, 4))
print("\nRandom normal tensor:\n", random_normal_tensor)
