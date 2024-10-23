import numpy as np
import torch
import torch.nn as nn

x_values=[i for i in range(11)]

x_train=np.array(x_values,dtype=np.float32)
x_train=x_train.reshape(-1,1)
print(x_train.shape)

y_values=[i for i in x_values]
y_train=np.array(y_values,dtype=np.float32)
y_train=y_train.reshape(-1,1)
print(y_train.shape)

class LinearModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        out=self.linear(x)
        return out

input_dim=1
output_dim=1
model=LinearModel(input_dim,output_dim)
print(model)

epochs=1000
a=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=a)
criterion=nn.MSELoss()

for epoch in range(epochs):
    inputs=torch.from_numpy(x_train)
    labels=torch.from_numpy(y_train)

    optimizer.zero_grad()

    outputs=model(inputs)

    loss=criterion(outputs,labels)

    loss.backward()

    optimizer.step()
    if epoch%50==0:
        print(f'epoch{epoch},loss {loss.item()}')