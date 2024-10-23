import torch
import numpy as np
'''
min f(x) = -3x[0] - 5x[1]
s.t. x[0] + x[2] = 4
    2x[1] + x[3] = 12
    3x[0] + 2x[1] + x[4] = 18
    x[0], x[1], x[2], x[3], x[4] >= 0
'''
eta=0.03
a=1
'''
min f(x)=c^T x
s.t Ax=b
x>=0
'''
c=torch.tensor([-3.,-5.,0.,0.,0.])
A=torch.tensor([[1., 0., 1., 0., 0.], [0., 2., 0., 1., 0.], [3., 2., 0., 0., 1.]])
b=torch.tensor([4.,12.,18.])

lambda_=torch.tensor([0.,0.,0.])
x=torch.tensor([2.,0.,2.,0.,0.],requires_grad=True)

def f(x):
    return c@x

def lagrangian_function(x,lamda_):
    return f(x)+lambda_@(A@x-b)+a/2*((A@x-b)**2).sum()#拉格朗日函数

def update_x(x):
    lagrangian_function(x,lambda_).backward()
    with torch.no_grad():
        x -= eta * x.grad
        x.clamp_(min=0)  #保证约束条件x>0
    x.grad.zero_()

def update_lambda(lambda_):
    lambda_new=lambda_+a*(A@x-b)
    lambda_.data=lambda_new

def pprint(i, x, lambda_):
    print(f'\n{i}th iter, L:{lagrangian_function(x, lambda_):.2f}, f: {f(x):.2f}' )
    print(f'x: {x}')
    print(f'lambda: {lambda_}')


for i in range(1001):
    if i%10==0:
        pprint(i,x,lambda_)
    update_x(x)
    update_lambda(lambda_)

'''
最优解
f(x)=-36
x=2,6,2,0,0
'''