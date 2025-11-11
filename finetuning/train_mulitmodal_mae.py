import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from datasets import FireRiskMultiModalDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")  # Headless for terminal with no display
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--encoder_path", type=str, required=True)
args = parser.parse_args()

CLASS_NAMES = ["Very_Low", "Low", "Moderate", "High", "Very_High", "Non-burnable", "Water"]

class MultiModalViT(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()

        # Load ViT encoder (no head)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        mae_state = torch.load(args.encoder_path, map_location="cpu")["model"]
        self.vit.load_state_dict(mae_state, strict=False)

        vit_dim = self.vit.num_features      # typically 768
        reduced_dim = 256                    # tunable

        # Normalize and reduce ViT features
        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, reduced_dim),
            nn.ReLU(),
        )

        # tabular pathway (unchanged & small)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # classifier takes viT + tab
        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7)
        )

    def forward(self, img, tab):
        vit_feat = self.vit.forward_features(img)     # (B, N, D)
        vit_feat = vit_feat.mean(dim=1)               # (B, D)
        vit_feat = self.vit_proj(vit_feat)            # (B, 256)

        tab_feat = self.tab_mlp(tab)                  # (B, 64)

        combined = torch.cat([vit_feat, tab_feat], dim=-1)
        return self.classifier(combined)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TAB_COLS = ["Wildfire_Risk_Score_NRI"]

    train_ds = FireRiskMultiModalDataset("../data/FireRisk", "../data/NRI_Table_Counties/image_county_data_with_burnprob.csv", TAB_COLS, split="train")
    val_ds   = FireRiskMultiModalDataset("../data/FireRisk", "../data/NRI_Table_Counties/image_county_data_with_burnprob.csv", TAB_COLS, split="val", scaler=train_ds.scaler)

    model = MultiModalViT(tab_dim=len(TAB_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    # =========================
    # Debug: Check MAE weight loading and feature outputs
    # =========================
    model.eval()

    # Load a small batch for testing
    sample_img, sample_tab, _ = next(iter(DataLoader(train_ds, batch_size=2)))
    sample_img, sample_tab = sample_img.to(device), sample_tab.to(device)

    # Check ViT features
    vit_feat = model.vit.forward_features(sample_img)
    print("=== ViT features ===")
    print("Mean:", vit_feat.mean().item())
    print("Std: ", vit_feat.std().item())
    print("Min: ", vit_feat.min().item())
    print("Max: ", vit_feat.max().item())

    # Check tabular features
    tab_feat = model.tab_mlp(sample_tab)
    print("\n=== Tabular features ===")
    print("Mean:", tab_feat.mean().item())
    print("Std: ", tab_feat.std().item())
    print("Min: ", tab_feat.min().item())
    print("Max: ", tab_feat.max().item())

    # Check classifier logits
    logits = model.classifier(torch.cat([vit_feat.mean(1), tab_feat], dim=-1))
    print("\n=== Classifier logits ===")
    print("Mean:", logits.mean().item())
    print("Std: ", logits.std().item())
    print("Min: ", logits.min().item())
    print("Max: ", logits.max().item())

    ###

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        for img, tab, label in tqdm(DataLoader(train_ds, batch_size=32, shuffle=True), desc=f"Train epoch {epoch}/{num_epochs}"):
            img, tab, label = img.to(device), tab.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img, tab), label)
            loss.backward()
            optimizer.step()

        # val loop
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, tab, label in tqdm(DataLoader(val_ds, batch_size=32), desc=f"Val epoch {epoch}/{num_epochs}"):
                img, tab, label = img.to(device), tab.to(device), label.to(device)
                preds = model(img, tab).argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASS_NAMES)

        plt.figure(figsize=(8,8))
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")

        # Save
        os.makedirs("results", exist_ok=True)  # make dir if not made
        plt.savefig("results/baseline_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Epoch {epoch}: Val Acc = {acc:.4f}, Macro-F1 = {macro_f1:.4f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(all_labels, all_preds, target_names=[
            "Very_Low","Low","Moderate","High","Very_High","Non-burnable","Water"
        ]))

    torch.save(model.state_dict(), "../models/multimodal_mae_vit.pth")

if __name__ == "__main__":
    main()
