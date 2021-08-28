import torch
import numpy as np

data_x = np.ones((3, 3))
x = torch.tensor(data_x, requires_grad=True, dtype=torch.float)
w = torch.ones((3, 1), requires_grad=True)
b = torch.ones((3, 1), requires_grad=True)

wx = x @ w
y = wx + b
loss = torch.mean(y)
loss.backward()
print(w.grad)

# check leaf x,w and b,
# leaf：我们创造出来的变量，x，w，b。
# wx 和 y 和 loss是由这些变量计算出来的
print('is leaf: ', x.is_leaf, w.is_leaf, b.is_leaf, wx.is_leaf, y.is_leaf)

# check grad
print('grad: ', x.grad, w.grad, b.grad, wx.grad, y.grad)

# check grad fn
print('grad_fn: ', x.grad_fn, w.grad_fn, b.grad_fn, wx.grad_fn, y.grad_fn, loss.grad_fn)