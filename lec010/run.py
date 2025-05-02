import torch

x = torch.tensor([3], dtype=torch.float32, requires_grad=True)
y = 3 * x**2 + 4 * x
y.backward()
print(x.grad)

x = torch.tensor([2, 5], dtype=torch.float32, requires_grad=True)
y = torch.dot(x, x)
y.backward()
print(x.grad)

x = torch.tensor([[1, 3]], dtype=torch.float32)
w = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1, 4, 5], dtype=torch.float32, requires_grad=True)
y = torch.matmul(x, w) + b
out = y.sum()
out.backward()
print(w.grad, b.grad)
