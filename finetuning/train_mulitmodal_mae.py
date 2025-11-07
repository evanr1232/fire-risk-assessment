import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from datasets import FireRiskMultiModalDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoder_path", type=str, required=True)
args = parser.parse_args()

class MultiModalViT(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        mae_state = torch.load(args.encoder_path, map_location="cpu")
        self.vit.load_state_dict(mae_state, strict=False)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.classifier = nn.Linear(self.vit.num_features + 128, 7)

    def forward(self, img, tab):
        vit_feat = self.vit.forward_features(img)
        tab_feat = self.tab_mlp(tab)
        return self.classifier(torch.cat([vit_feat, tab_feat], dim=-1))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TAB_COLS = ["Wildfire_Risk_Score_NRI"]

    train_ds = FireRiskMultiModalDataset("../data/FireRisk", "../data/NRI_Table_Counties/image_county_data_with_burnprob.csv", TAB_COLS, split="train")
    val_ds   = FireRiskMultiModalDataset("../data/FireRisk", "../data/NRI_Table_Counties/image_county_data_with_burnprob.csv", TAB_COLS, split="val", scaler=train_ds.scaler)

    model = MultiModalViT(tab_dim=len(TAB_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        for img, tab, label in DataLoader(train_ds, batch_size=32, shuffle=True):
            img, tab, label = img.to(device), tab.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img, tab), label)
            loss.backward()
            optimizer.step()

        # val loop
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, tab, label in DataLoader(val_ds, batch_size=32):
                img, tab, label = img.to(device), tab.to(device), label.to(device)
                preds = model(img, tab).argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Epoch {epoch}: Val Acc = {acc:.4f}, Macro-F1 = {macro_f1:.4f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(all_labels, all_preds, target_names=[
            "Very_Low","Low","Moderate","High","Very_High","Non-burnable","Water"
        ]))

    torch.save(model.state_dict(), "multimodal_mae_vit.pth")

if __name__ == "__main__":
    main()