import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None):
        """
        root_dir: 图像文件夹路径
        mask_dir: 对应mask文件夹路径
        transform: 图像和mask的预处理操作
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]  # 读取所有图像文件

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = os.path.join(self.root_dir, self.image_names[idx])

        # 获取对应的mask路径
        # 去除图像名称中的扩展名，添加"_mask"后再拼接扩展名
        image_name = os.path.splitext(self.image_names[idx])[0]  # 获取图像名称（不含扩展名）
        mask_name = f"mask_{image_name}.jpg"  # 正确的mask文件名格式
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像和mask
        image = Image.open(img_path).convert('RGB')  # 确保图像为RGB模式
        mask = Image.open(mask_path).convert('L')    # 假设mask是单通道（灰度图）

        # 应用预处理（transform）
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
