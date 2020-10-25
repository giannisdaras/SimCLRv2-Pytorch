import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class ImageNetEval(Dataset):
    def __init__(self, images_path, labels_path, mappings_path):
        # two fold mappings: image_name => image_label, image_label => image_desc
        self.mapping = self.get_mappings(mappings_path)
        self.images_path = images_path
        self.labels = np.load(labels_path)
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.images_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]

    def __len__(self):
        return len(self.labels)

    def get_mappings(self, mappings_path):
        mapping = {}
        with open(mappings_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_name, image_label, image_desc = line.split(' ')
                mapping[image_name] = image_label
                mapping[image_label]  = image_desc.strip('\n')
        return mapping


if __name__ == '__main__':
    ds = ImageNetEval('/Users/giannis/datasets/ILSVRC2012_img_val',
                      '/Users/giannis/datasets/labels.npy',
                      '/Users/giannis/datasets/mappings.txt')
