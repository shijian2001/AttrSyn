import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from dataset import Cub2011, HfCub2011

def experiment(clip_model: str, domain: str, method: str, cuda: int=0):

    device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device)

    if domain == "photo":
        test = HfCub2011(split="test", transform=preprocess)
    elif domain == "painting":
        test = Cub2011("/linxindisk/AttrSyn/data/real/CUB-200-Painting", transform=preprocess)
    else:
        raise ValueError("Not Implemented Error")
    
    if method == "base":
        train = Cub2011(root=f"/linxindisk/AttrSyn/data/synthetic/cub-{domain}/sdxl-base-small", transform=preprocess)
    elif method == "attrsyn":
        train = Cub2011(root=f"/linxindisk/AttrSyn/data/synthetic/cub-{domain}/sdxl-diversity-small", transform=preprocess)
    else:
        raise ValueError("Not Implemented Error")

    def get_features(dataset):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(DataLoader(dataset, batch_size=256)):
                images, labels = batch['image'].to(device), batch['label']
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    LR = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    LR.fit(train_features, train_labels)
    lr_pred = LR.predict(test_features)
    lr_acc = np.mean((test_labels == lr_pred).astype(float)) * 100.
    print(f"{clip_model}-{method}-LR-{domain}-Accuracy:{lr_acc:.2f}%")

    MLP = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=1000, verbose=0)
    MLP.fit(train_features, train_labels)
    mlp_pred = MLP.predict(test_features)
    mlp_acc = np.mean((test_labels == mlp_pred).astype(float)) * 100.
    print(f"{clip_model}-{method}-MLP-{domain}-Accuracy:{mlp_acc:.2f}%")

# ["RN50", "RN101", "ViT-B/16", "ViT-L/14"]
for method in ["attrsyn", "base"]:
        for domain in ["photo", "painting"]:
            experiment(
                clip_model="ViT-L/14@336px",
                method=method,
                domain=domain,
                cuda=0
            )