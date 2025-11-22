# datasets.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["Very_Low", "Low", "Moderate", "High", "Very_High", "Non-burnable", "Water"]


# For NIR columns that are strings --> convert to numbers
ORDINAL_MAP = {
    "Very Low": 1,
    "Relatively Low": 2,
    "Relatively Moderate": 3,
    "Relatively High": 4,
    "Very High": 5
}

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def make_label_map():
    return {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

class FireRiskImageDataset(Dataset):
    """
    Image-only dataset (baseline)
    """
    def __init__(self, root_dir, split="train"):
        self.root = os.path.join(root_dir, split)
        self.label_map = make_label_map()
        self.samples = []

        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            if not os.path.isdir(class_path): continue
            for f in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, f), self.label_map[class_name]))

        if split == "train":
            self.transform = train_transform
        else:
            self.transform = eval_transform


    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class FireRiskMultiModalDataset(Dataset):
    """
    Image + Tabular dataset with ordinal encoding
    """
    def __init__(self, root_dir, csv_path, tab_cols, split="train", scaler=None):
        self.root = os.path.join(root_dir, split)
        df = pd.read_csv(csv_path)
        df = df[df["folder"] == split]
        df["filename"] = df["filename"].astype(str)

        # Convert categorical columns to ordinal numbers
        for col in tab_cols:
            if df[col].dtype == object:
                df[col] = df[col].map(ORDINAL_MAP)

        # Drop any rows with NaN after conversion
        df = df.dropna(subset=tab_cols).reset_index(drop=True)
        self.df = df
        self.tab_cols = tab_cols
        self.scaler = scaler if scaler else self._fit_scaler()

        self.samples = []
        self.label_map = make_label_map()
        for cls in os.listdir(self.root):
            class_path = os.path.join(self.root, cls)
            if not os.path.isdir(class_path): continue
            for f in os.listdir(class_path):
                match = self.df[self.df["filename"] == f]
                if len(match) == 0: continue
                row_idx = match.index[0]
                label = self.label_map[cls]
                self.samples.append((os.path.join(class_path, f), label, row_idx))

        labels_np = np.array([s[1] for s in self.samples])
        self.class_counts = np.bincount(labels_np, minlength=7)

        if split == "train":
            self.transform = train_transform
        else:
            self.transform = eval_transform


    def _fit_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.df[self.tab_cols])
        return scaler

    def __getitem__(self, index):
        path, label, row_idx = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        tab_df = self.df.loc[[row_idx], self.tab_cols]
        tab_scaled = self.scaler.transform(tab_df)
        tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32).squeeze(0)
        return img, tab_tensor, label

    def __len__(self):
        return len(self.samples)