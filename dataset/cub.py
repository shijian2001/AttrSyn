import torchvision
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

classes_file = "/linxindisk/AttrSyn/data/real/CUB-200-Painting/cub_list_drawing.txt"
CLASSES = []

with open(classes_file, "r") as file:
    for line in file:
        line = line.strip()
        words = line.split()
        class_name = " ".join(words[:-1])
        CLASSES.append(class_name)

class HfCub2011(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('efekankavalci/CUB_200_2011', split=split)
        self.transform = transform

    @property
    def classes(self):
        return CLASSES

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }

class Cub2011(torchvision.datasets.ImageFolder):

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": target,
        }

if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    cub_2011 = HfCub2011(split="train", transform=train_transform)

    train_loader = DataLoader(cub_2011, batch_size=32, shuffle=True, num_workers=4)