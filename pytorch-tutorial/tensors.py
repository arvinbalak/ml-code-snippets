import numpy as np
import torch
from setuptools.ssl_support import is_available
from torch.onnx.symbolic_opset9 import new_zeros

# Initialization
# ---------------------
x = torch.tensor([5, 3])
print(x)

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# using other existing tensors
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

y = torch.randn_like(x, dtype=torch.double)
print(y.size())  # size is tuple
print(y)

# Operations
# -----------
print(x+y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Any operation that mutates a tensor in-place is post-fixed with an _
y.add_(x)
print(y)

# slices
print(y[:, 1])

# resize
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x)
print(y)
print(z)

# Numpy porting
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)  # change is reflected in both

a = np.ones(5)
b = torch.from_numpy(a)

np.add(a, 1, out=a)
print(a)
print(b)  # change is reflected in both

# Cuda tensors
print("executing on cuda...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)

    z = x + y
    print(x)
    print(y)
    print(z)
    print(z.to("cpu", torch.double))
