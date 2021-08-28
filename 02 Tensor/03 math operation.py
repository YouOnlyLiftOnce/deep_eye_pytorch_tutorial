import torch
# 参照notion笔记

# add
input = torch.ones((3, 3))
other = torch.ones((3, 3))
# input+other*alpha
tensor_1 = torch.add(input, other, alpha=5)
print(tensor_1)

# addcdiv : out = input + value x (tensor1/tensor2)
input = torch.ones((3, 3))
tensor1 = torch.full((3, 3), 4.)
tensor2 = torch.ones((3, 3))
tensor_addcdiv = torch.addcdiv(input, tensor1, tensor2, value=2)
print(tensor_addcdiv)

# addcmul(): out = input + value x tensor1 x tensor 2
tensor_addcmul = torch.addcmul(input, tensor1, tensor2, value=2)
print(tensor_addcmul)