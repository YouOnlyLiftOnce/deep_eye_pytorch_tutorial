import torch
# 拼接
tensor_1 = torch.ones((3, 5))
tensor_2 = torch.zeros((3, 5))
# dim=0 方向拼接， 需要两个tensor dim=0 维度一致
tensor_cat_0 = torch.cat([tensor_1, tensor_2], dim=0)
# dim=1 方向拼接， 需要两个tensor dim=1 维度一致
tensor_cat_1 = torch.cat([tensor_1, tensor_2], dim=1)
print(tensor_cat_0.shape)
print(tensor_cat_1.shape)

# 在新创建的dim 上拼接, 需要两个tensor的 shape一样
tensor_stack = torch.stack([tensor_1, tensor_2], dim=2)
# 相当于在 dim=2 上stack 了
print(tensor_stack.shape)

# 切分
# 在dim=0 上 分3份，第一份为 2x3 第二份 2x3 第三份不够 1x3
tensor_3 = torch.ones((5, 3))
chunk_1, chunk_2, chunk_3 = torch.chunk(tensor_3, 3, dim=0)
print(chunk_1)
print(chunk_2)
print(chunk_3)

# split 这里的2表示每一份的长度
split_1, split_2, split_3 = torch.split(tensor_3, 2, dim=0)
print(split_1)
print(split_2)
print(split_3)
# 也可以传入一个list，指定每个split的长度
split_4, split_5 = torch.split(tensor_3, [3, 2], dim=0)
print(split_4.shape)
print(split_5.shape)

# 索引
tensor_4 = torch.arange(9).reshape(3, 3)
idx = torch.tensor([0, 2], dtype=torch.long)
# 这里的 0 指的是 dim=0，idx 需要时整形张量, 输出为索引的拼接
tensor_index_select = torch.index_select(tensor_4, 0, idx)
print(tensor_index_select)
# also can use the numpy way to index
print('index like numpy: ', tensor_4[:, 1])

# boolean mask index
tensor_4 = torch.arange(9).reshape(3, 3)
# ge: >= gt:> lt: < le: <= eq: =
mask = tensor_4.ge(5) # boolean tensor of same shape
# masked 之后的值重组为一维向量
tensor_masked = torch.masked_select(tensor_4, mask)
print('mask: ', mask)
print('masked selected tensor: ', tensor_masked)
# 用元素乘法来 mask，不改变tensor形状
print('masked tensor: ', tensor_4*mask)

# reshape
torch_reshape = torch.reshape(tensor_4, (1, 9))
print(torch_reshape.shape)

# transpose 指定维度
image = torch.randint(0, 255, (3, 28, 28))
image = torch.transpose(image, dim0=0, dim1=2)
print(image.shape)
# torch.t() transpose 二维矩阵

# 压缩 去除维度为1的dim
tensor_5 = torch.zeros((1, 3, 4, 1))
tensor_sq = torch.squeeze(tensor_5)
print('squeeze: ', tensor_sq.shape)

# 在dim=0 加轴
tensor_upsq = torch.unsqueeze(tensor_5, dim=0)
print(tensor_upsq.shape)