import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class ImageNetEval(Dataset):
    def __init__(self, images_path, labels_path, mappings_path):
        # two fold mappings: image_name => image_label, image_label => image_desc
        self.mapping = self.get_mappings(mappings_path)
        self.images_path = images_path
        self.labels = self.get_labels(labels_path)
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.images_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]

    def __len__(self):
        return len(self.labels)

    def get_labels(self, labels_path):
        val_path = labels_path + 'val/'
        labels = []
        files = glob.glob(val_path + '/*')
        for file in sorted(files):
            with open(file, 'r') as xml_file:
                xml_file_lines = xml_file.readlines()
                for line in xml_file_lines:
                    if '<name>' in line:
                        labels.append(self.mapping[line.split('>')[1].split('<')[0]])
                        break
        return labels

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
                  '/Users/giannis/datasets/Annotations/CLS-LOC/',
                  '/Users/giannis/datasets/mappings.txt')
