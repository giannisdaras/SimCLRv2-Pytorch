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


@torch.no_grad()
def run(args):
    dataset = ImageNetEval('/Users/giannis/datasets/ILSVRC2012_img_val',
                           '/Users/giannis/datasets/Annotations/CLS-LOC/',
                           '/Users/giannis/datasets/mappings.txt')

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    model, _ = get_resnet(*name_to_params(args.pth_path))
    model.load_state_dict(torch.load(args.pth_path)['resnet'])
    model = model.to(args.device).eval()
    preds = []
    target = []
    for images, labels in tqdm(data_loader):
        _, pred = model(images.to(args.device), apply_fc=True).topk(1, dim=1)
        preds.append(pred.squeeze(1).cpu())
        target.append(labels)
    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()

    all_counters = [Counter() for i in range(1000)]
    for i in range(len(dataloader)):
        all_counters[t[i]][p[i]] += 1
    total_correct = 0
    for i in range(1000):
        total_correct += all_counters[i].most_common(1)[0][1]
    print(f'Accuracy: {total_correct / len(dataloader) * 100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('pth_path', type=str, help='path of the input checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run(args)
