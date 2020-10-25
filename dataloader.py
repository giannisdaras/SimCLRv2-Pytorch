import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class ImageNetEval(Dataset):
    def __init__(self, images_path, labels_path, mappings_path=None):
        self.images_path = images_path
        self.labels = []
        with open(labels_path, 'r') as f:
            for line in f.readlines():
                self.labels.append(int(line.strip('\n')) - 1)
        self.labels = np.array(self.labels)
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])
        self.mappings = {}
        if mappings_path is not None:
            with open(mappings_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    number, desc = line.strip('\n').split(' ')[1:]
                    self.mappings[int(number) - 1] = desc

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.images_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]

    def __len__(self):
        return len(self.labels)
