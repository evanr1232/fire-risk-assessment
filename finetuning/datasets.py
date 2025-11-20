# datasets.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["Very_Low", "Low", "Moderate", "High", "Very_High", "Non-burnable", "Water"]

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

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class FireRiskMultiModalDataset(Dataset):
    """
    Image + Tabular dataset
    - Supports multiple tab columns
    - Drops rows that are missing tabular values
    - Ensures filename matching
    """

    def __init__(self, root_dir, csv_path, tab_cols, split="train", scaler=None):
        self.root = os.path.join(root_dir, split)
        df = pd.read_csv(csv_path)

        # Only rows for this split
        df = df[df["folder"] == split].copy()

        # Ensure filenames are strings (sometimes numbers appear)
        df["filename"] = df["filename"].astype(str).str.strip()

        # Drop rows where ANY tab column is NA
        df = df.dropna(subset=tab_cols)

        self.df = df
        self.tab_cols = tab_cols

        # Fit scaler only on training split
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.tab_cols])
        else:
            self.scaler = scaler

        self.samples = []
        self.label_map = make_label_map()

        # Match image files to their tab-row
        for cls in os.listdir(self.root):
            class_path = os.path.join(self.root, cls)
            if not os.path.isdir(class_path):
                continue

            for f in os.listdir(class_path):
                # ensure string match
                f_clean = str(f).strip()
                match = df[df["filename"] == f_clean]

                if len(match) == 0:
                    continue  # skip missing or invalid rows

                row_idx = match.index[0]
                self.samples.append((os.path.join(class_path, f), self.label_map[cls], row_idx))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __getitem__(self, index):
        path, label, row_idx = self.samples[index]

        # load image
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        # tabular vector
        tab_raw = self.df.loc[row_idx, self.tab_cols].values.astype(float)  # (tab_dim,)
        tab_scaled = self.scaler.transform([tab_raw])[0]
        tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32)

        return img, tab_tensor, label

    def __len__(self):
        return len(self.samples)