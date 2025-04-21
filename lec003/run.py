import torch

# 1. Basic Arithmetic Operations
print("\n--- Basic Arithmetic Operations ---")
basic_matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original matrix:\n", basic_matrix)
print("Multiplication by scalar:\n", basic_matrix * 5)
print("Addition with scalar:\n", basic_matrix + 7)

# 2. Reshaping Operations
print("\n--- Reshaping Operations ---")
# reshape changes the shape while preserving the number of elements
reshaped_matrix = basic_matrix.reshape(3, 2)
print("Reshaped from (2,3) to (3,2):\n", reshaped_matrix)

# 3. Dimension Operations
print("\n--- Dimension Operations ---")
# squeeze removes dimensions of size 1
multi_dim_tensor = torch.randn(1, 2, 1, 3)
print("Original shape:", multi_dim_tensor.shape)
print("After squeeze (all dims):", multi_dim_tensor.squeeze().shape)
print("After squeeze (dim 0):", multi_dim_tensor.squeeze(0).shape)
print("After squeeze (dim 2):", multi_dim_tensor.squeeze(2).shape)

# unsqueeze adds a dimension of size 1
image_tensor = torch.randn(28, 28)  # Simulating an image
batched_image = image_tensor.unsqueeze(0)  # Add batch dimension
print("Image shape:", image_tensor.shape)
print("Batched image shape:", batched_image.shape)

# 4. Matrix Multiplication
print("\n--- Matrix Multiplication ---")
matrix1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
matrix2 = torch.tensor([[1, 2], [4, 5], [6, 7]])
print("Matrix multiplication (@):\n", matrix1 @ matrix2)
print("Using torch.matmul:\n", torch.matmul(matrix1, matrix2))

# 5. Concatenation
print("\n--- Concatenation ---")
tensor3d = torch.randn(3, 5, 6)
# Concatenate along dimension 1 (doubling the middle dimension)
concatenated = torch.cat([tensor3d, tensor3d], dim=1)
print("Original shape:", tensor3d.shape)
print("After concatenation:", concatenated.shape)

# 6. Maximum Values and Indices
print("\n--- Maximum Operations ---")
example_tensor = torch.tensor(
    [[[11, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 13]], [[4, 5], [6, 17], [8, 9]]]
)
print("Original tensor shape:", example_tensor.shape)

# Get maximum along first dimension (dim=0)
max_values, max_indices = example_tensor.max(dim=0)
print("Maximum values:\n", max_values)
print("Indices of maximum values:\n", max_indices)

# 7. Permutation
print("\n--- Permutation ---")
original_tensor = torch.randn(5, 6, 7)
# Rearrange dimensions: 7,5,6 instead of 5,6,7
permuted_tensor = original_tensor.permute(2, 0, 1)
print("Original shape:", original_tensor.shape)
print("After permute:", permuted_tensor.shape)
