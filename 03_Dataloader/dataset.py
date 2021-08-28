from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

# 获取所在文件夹绝对路径名称
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RMBDataset(Dataset):

    def __init__(self, data_dir, label_dir, transform=None):
        # base： 项目文件夹的绝对地址
        # data_dir: 文件加下 data的文件名
        # label_dir: 文件加下 label的文件名
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.base = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.base, data_dir, label_dir)
        # 给定路径下所有文件名组成的列表
        self.img_path = os.listdir(self.path)
        # 把transform: torch.transform，数据预处理函数传给 self.transform
        self.transform = transform

    # 当 index时 自动调用这个函数
    # 输出 img，label
    def __getitem__(self, idx):
        # idx 对应文件名
        img_name = self.img_path[idx]
        # 每个文件的位置
        img_item_path = os.path.join(self.base, self.data_dir, self.label_dir, img_name)
        # read img, 不要用 opencv读，transform 不认
        img = Image.open(img_item_path)
        # label 所在文件夹名即是label
        label = self.label_dir

        # 对图片进行预处理
        if self.transform != None:
            img = self.transform(img)
        return img, int(label)


    def __len__(self):
        return len(self.img_path)



