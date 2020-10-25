import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class ImageNetEval(Dataset):
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels = np.load(labels_path)
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.images_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), int(self.labels[item]) - 1

    def __len__(self):
        return len(self.labels)


