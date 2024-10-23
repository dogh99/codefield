import torch

x=torch.tensor([1.],requires_grad=True)

y=x**2

res=torch.autograd.grad(outputs=y,inputs=x,create_graph=True)#返回的是一个元组

print(res[0])
