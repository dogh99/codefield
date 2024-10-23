import numpy as np
import torch

x=torch.tensor([2.],requires_grad=True)

def func(x):
    return x**2+2*x+5

#步长控制
a=0.1

    
def grad_xx(x,y):
    grad_x=torch.autograd.grad(outputs=y,inputs=x,create_graph=True)[0]
    hessian=torch.autograd.grad(outputs=grad_x,inputs=x,create_graph=True)[0]
    return hessian

for i in range(100):
    y=func(x)
    y.backward(retain_graph=True)#保存计算图，否则会被释放掉以节省内存
    hessian=grad_xx(x,y)
    with torch.no_grad():
        x-=a * x.grad / hessian
    x.grad.zero_()
    '''
    x_new = x-a * x.grad / hessian
    x.data=x_new
    x.grad.zero_()
    '''
print(f"优化后x的位置在{x.item()}，优化值为{func(x).item()}")    

