import os
import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  
from dataset import Cub2011, HfCub2011

def experiment(clip_model: str, domain: str, cuda: int=0):
    device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device)

    if domain == "photo":
        test = HfCub2011(split="test", transform=preprocess)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to(device)
    elif domain == "painting":
        test = Cub2011("/linxindisk/AttrSyn/data/real/CUB-200-Painting", transform=preprocess)
        text_inputs = torch.cat([clip.tokenize(f"a painting of a {c}") for c in test.classes]).to(device)
    else:
        raise ValueError("Not Implemented Error")

    data_loader = DataLoader(test, batch_size=256, shuffle=False)

    correct_predictions = 0
    total_images = len(test)

    model.eval()

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, labels = batch['image'].to(device), batch['label']

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top1_predictions = similarity.argmax(dim=-1)

            correct_predictions += (top1_predictions == labels.to(device)).sum().item()

    accuracy = correct_predictions / total_images * 100
    print(f"{domain}-{clip_model}-Average accuracy: {accuracy:.2f}%")

for domain in ["photo", "painting"]:
    for clip_model in ["ViT-L/14@336px"]:
        experiment(clip_model, domain, cuda=1)