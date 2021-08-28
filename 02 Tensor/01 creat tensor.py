import torch
import numpy as np

# from numpy or list
# 内存和numpy 数据共享，改一个=> 另一个改变
data = np.arange(9).reshape(3, 3)
tensor_1 = torch.tensor(data,
             dtype=None,
             device='cuda',
             requires_grad=False,
             pin_memory=False)
print(tensor_1)

# from_numpy
tensor_2 = torch.from_numpy(data)
print(tensor_2)

# zeros, ones, full
out_t = torch.tensor([1])
# out 相当于把值赋给 tensor out_t
tensor_3 = torch.zeros((3, 3), out=out_t)
print(out_t)
tensor_4 = torch.ones((3, 3))
# 5. => use float data type
tensor_5 = torch.full((3, 3), 5.)
print(tensor_5)

# arange, linspace, logspace, eyes
tensor_6 = torch.arange(2, 10, 2)
print(tensor_6)
# 5 equal parts
tensor_7 = torch.linspace(2, 10, 5)
print(tensor_7)
# 10: 以10为底的log
tensor_8 = torch.logspace(2, 10, 5, 10)
# 对角矩阵
tensor_9 = torch.eye(4, 3)

# from probability
# normal
# 这里的mean, std 可以是 标量 或 tensor

mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1, 5, dtype=torch.float)
tensor_10 = torch.normal(mean, std)
# mean 和 std 生成的 tensor
# 第一个元素 是 mean=1，std=1 的抽样
# 第二个元素 是 mean=2，std=2 的抽样
# 如过 mean=1 是标量，std 是tensor， 生成 mean=1,std=1, mean=1, std=2...
print('normal:', tensor_10)

# randn, randn_like, rand, rand_like
# 服从正太分布的随机数
tensor_11 = torch.randn(10)

tensor_like = torch.tensor((4, 4), dtype=torch.float32)
# 与 tensor_like 相同形状, 传入的tensor的数据类型是浮点数
tensor_12 = torch.randn_like(tensor_like)
print(tensor_11)
print(tensor_12)

# [0,1) uniform distribution, torch.rand_like
tensor_13 = torch.rand((3, 3))
print(tensor_13)

# [low, high) uniform distribution
tensor_14 = torch.randint(0, 10, (3, 3))
print(tensor_14)

# [0, n) 的随机排列，主要为了随机索引
tensor_idx = torch.randperm(10)
print(tensor_idx)

# bernoulli 输入值为一个tensor 0-1 之间表示概率， 输出为 0 或 1，服从伯努利分布
tensor_prob = torch.rand((3, 3))
tensor_15 = torch.bernoulli(tensor_prob)
print(tensor_15)