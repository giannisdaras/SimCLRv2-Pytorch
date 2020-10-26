import numpy as np
import os
from collections import Counter
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from resnet import get_resnet, name_to_params
from dataloader import ImageNetEval
import yaml
import torch.optim as optim

@torch.no_grad()
def evaluate(config):
    files = config['files']
    dataset = ImageNetEval(files['images_path'], files['labels_path'],
                           files['mappings_path'])
    data_loader = DataLoader(dataset, batch_size=config['evaluate']['batch_size'],
                             shuffle=False, pin_memory=True, num_workers=8)
    model, _ = get_resnet(*name_to_params(files['checkpoint']))
    model.load_state_dict(torch.load(files['checkpoint'])['resnet'])
    model = model.to(config['device']).eval()
    preds = []
    target = []
    for images, labels in tqdm(data_loader):
        _, pred = model(images.to(config['device']),
                        apply_fc=True).topk(config['evaluate']['top_k'], dim=1)
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


def train(config):
    t_config = config['train']
    files = config['files']
    dataset = ImageNetEval(files['images_path'], files['labels_path'],
                           files['mappings_path'])
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'],
                             shuffle=True, pin_memory=True, num_workers=8)
    model, _ = get_resnet(*name_to_params(files['checkpoint']))
    optimizer = optim.Adam(model.fc.parameters(), lr=t_config['lr'],
                           betas=(t_config['beta1'], t_config['beta2']))
    model.load_state_dict(torch.load(files['checkpoint'])['resnet'])
    model = model.to(config['device']).train()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = 'NA'
    for i in tqdm(range(t_config['epochs'])):
        for images, labels in tqdm(data_loader, pbar=f'Loss: {loss}'):
            _, pred = model(images.to(config['device']), apply_fc=True)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()



if __name__ == '__main__':
    with open('config.yml') as file:
        config = yaml.safe_load(file)
    # evaluate(config)
    train(config)
