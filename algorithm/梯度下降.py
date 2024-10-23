import torch
import numpy as np
import matplotlib.pyplot as plt

x=torch.tensor(3.0,requires_grad=True)
a=0.07
n=100
x_list=[]
y_list=[]

for i in range(n):
    y=x**2+2*x+5
    y.backward()
    x_new=x-a*x.grad
    x.data=x_new
    x_list.append(x_new.item())
    y_list.append(y.item())
    x.grad.zero_()
print(x)

Fig1=plt.figure()
x=np.arange(-5,5,0.05)
y=y=x**2+2*x+5
plt.plot(x,y)
plt.plot(x_list,y_list,'-o')
plt.show()