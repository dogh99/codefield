import torch

x = torch.tensor([2.], requires_grad=True)
y = 3 * x**2

# 一阶导数
y.backward(create_graph=True)
print(x.grad)  # 这将打印一阶导数，应该是12

# 在计算二阶导数之前清零梯度
x.grad.zero_()

# 使用新的标量值来触发反向传播
grad_x = x.grad.clone()
grad_x.backward()
print(x.grad)  # 这将打印二阶导数，应该是6
