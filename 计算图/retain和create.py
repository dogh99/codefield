import torch

x=torch.tensor([2.],requires_grad=True)

y=x**2

grad_x1=torch.autograd.grad(outputs=y,inputs=x,retain_graph=True)

print(grad_x1)
print(grad_x1[0].requires_grad)



grad_x2=torch.autograd.grad(outputs=y,inputs=x,create_graph=True)

print(grad_x2)
print(grad_x2[0].requires_grad)

