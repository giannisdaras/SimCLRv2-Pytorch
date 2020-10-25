import numpy as np
import os
import argparse
from collections import Counter
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from resnet import get_resnet, name_to_params
from dataloader import ImageNetEval
import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum().item()
        res.append(correct_k)
    return res



def plot(dataset, index):
    image, label = dataset[index]
    print(dataset.mappings[label])
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

@torch.no_grad()
def run(args):
    dataset = ImageNetEval(args.images_path, args.labels_path,
                           args.mappings_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=8)
    model, _ = get_resnet(*name_to_params(args.pth_path))
    model.load_state_dict(torch.load(args.pth_path)['resnet'])
    model = model.to(args.device).eval()
    preds = []
    target = []
    for images, labels in tqdm(data_loader):
        _, pred = model(images.to(args.device), apply_fc=True).topk(1, dim=1)
        preds.append(pred.squeeze(1).cpu())
        target += labels
    p = torch.cat(preds).numpy()
    t = np.array(target, dtype=np.int32)
    all_counters = [Counter() for i in range(1000)]
    for i in range(len(dataset)):
        all_counters[t[i]][p[i]] += 1
    total_correct = 0
    for i in range(1000):
        total_correct += all_counters[i].most_common(1)[0][1]
    print(f'Accuracy: {total_correct / len(dataset) * 100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('pth_path', type=str,
                        help='path of the input checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--images_path', type=str,
                        default='~/datasets/ILSVRC2012_img_val')
    parser.add_argument('--labels_path', type=str,
                        default='~/datasets/ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument('--mappings_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    run(args)
