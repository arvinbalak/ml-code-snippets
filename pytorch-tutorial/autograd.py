import torch

# grad_fn
x = torch.ones(2, 2, requires_grad=True)
print(x)

print("one operation")
y = x + 2
print(y)
print(y.grad_fn)

print("more operations...")
z = y * y * 3
out = z.mean()

print(z)
print(out)

# backprop
print("backprop...")
out.backward()

print(x.grad)
