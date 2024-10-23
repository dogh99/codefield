import torch
import numpy as np

x=torch.tensor([2.],requires_grad=True)
y=3*x**2

#一阶导数
y.backward(create_graph=True)
print(x.grad)
x.grad.data.zero_()
x.grad.backward()
print(x.grad)

