import torch

x=torch.tensor([2.],requires_grad=True)

y=3*x**2

grad_x=torch.autograd.grad(outputs=y,inputs=x,create_graph=True)
print(grad_x)
grad_xx=torch.autograd.grad(outputs=grad_x[0],inputs=x)
print(grad_xx[0])