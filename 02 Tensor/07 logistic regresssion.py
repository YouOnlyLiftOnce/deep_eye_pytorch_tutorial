import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============================ step 1/5 生成数据 ============================
sample_nums = 100
mean = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2) # 100x2
x0 = torch.normal(mean, 1, size=n_data.shape) + bias      # 类别0 数据 shape=(100, 2)
print(x0.shape)
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100)
x1 = torch.normal(-mean, 1, size=n_data.shape) + bias     # 类别1 数据 shape=(100, 2)
print(x1.shape)
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100)

train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

# ============================ step 2/5 选择模型 ============================

class LR(nn.Module):
    # constructor 传入方程
    def __init__(self):
        super(LR, self).__init__()
        # linear(input_dims, output_dim)
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()   # net initialization



# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()

# ============================ step 4/5 选择优化器   ============================
lr = 0.01
# 相当于是 学习方法的函数
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# ============================ step 5/5 模型训练 ============================
for iteration in range(2000):
    # forward
    y_pred = lr_net(train_x)
    # make sure size of target and input are the same
    loss = loss_fn(y_pred.squeeze(), train_y)
    # backward
    loss.backward()
    # update parameters
    # 也可以手动更新参数，但是这里的参数是封装在nn.Linear 访问不到
    optimizer.step()
    # 参数清零
    optimizer.zero_grad()


    # 绘图
    if iteration % 20 == 0:

        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break
