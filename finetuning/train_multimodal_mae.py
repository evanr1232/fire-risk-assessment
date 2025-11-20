import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from datasets import FireRiskMultiModalDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--encoder_path", type=str, required=True, default="../models/mae_vit_encoder_imagenet1k_base.pth")
parser.add_argument("--num_epochs", type=int, required=True, default=50)
parser.add_argument("--lr", type=float, required=True, default=1e-4)
args = parser.parse_args()

CLASS_NAMES = ["Very_Low", "Low", "Moderate", "High", "Very_High", "Non-burnable", "Water"]

# ----------------------------------------------------------------------
# MULTIMODAL MODEL
# ----------------------------------------------------------------------
class MultiModalViT(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        mae_state = torch.load(args.encoder_path, map_location="cpu")
        if "model" in mae_state:
            mae_state = mae_state["model"]
        self.vit.load_state_dict(mae_state, strict=False)

        vit_dim = self.vit.num_features
        reduced_dim = 256

        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, reduced_dim),
        )

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(CLASS_NAMES))
        )

    def forward(self, img, tab):
        vit_feat = self.vit.forward_features(img)       
        vit_feat = vit_feat.mean(dim=1)                 
        vit_feat = self.vit_proj(vit_feat)              
        tab_feat = self.tab_mlp(tab)                    
        combined = torch.cat([vit_feat, tab_feat], dim=-1)
        return self.classifier(combined)

# ----------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Tabular columns to use
    # ------------------------
    TAB_COLS = [
        "WFIR_AFREQ","WFIR_EXP_AREA","WFIR_EXPB","WFIR_EXPP","WFIR_EXPPE",
        "WFIR_EXPA","WFIR_EXPT","WFIR_HLRB","WFIR_HLRP","WFIR_HLRA","WFIR_HLRR",
        "WFIR_EALB","WFIR_EALP","WFIR_EALPE","WFIR_EALA","WFIR_EALT","WFIR_EALS",
        "WFIR_EALR","WFIR_ALRB","WFIR_ALRP","WFIR_ALRA","WFIR_ALR_NPCTL",
        "WFIR_RISKV","WFIR_RISKS","WFIR_RISKR"
    ]

    # ------------------------
    # Datasets and loaders
    # ------------------------
    train_ds = FireRiskMultiModalDataset(
        "../data/FireRisk",
        "../data/NRI_Table_Counties/image_to_all_NRI_WFIR.csv",
        TAB_COLS,
        split="train"
    )

    val_ds = FireRiskMultiModalDataset(
        "../data/FireRisk",
        "../data/NRI_Table_Counties/image_to_all_NRI_WFIR.csv",
        TAB_COLS,
        split="val",
        scaler=train_ds.scaler
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ------------------------
    # Model
    # ------------------------
    model = MultiModalViT(tab_dim=len(TAB_COLS)).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    model.vit_proj.apply(init_weights)
    model.tab_mlp.apply(init_weights)
    model.classifier.apply(init_weights)

    lr = args.lr

    optimizer = torch.optim.AdamW([
        {"params": model.vit.parameters(),         "lr": 1e-5},
        {"params": model.vit_proj.parameters(),    "lr": lr},
        {"params": model.tab_mlp.parameters(),     "lr": lr},
        {"params": model.classifier.parameters(),  "lr": lr},
    ], weight_decay=0.05)

    criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs
    best_val_acc = 0.0
    history = []

    os.makedirs("results", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # ------------------------
    # Training
    # ------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_all_preds = []
        train_all_labels = []

        for images, tab_feats, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}"):
            images, tab_feats, labels = images.to(device), tab_feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, tab_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_all_preds.append(predicted.cpu())
            train_all_labels.append(labels.cpu())

        train_loss = running_loss / total
        train_acc = correct / total
        train_f1 = f1_score(
            torch.cat(train_all_labels).numpy(),
            torch.cat(train_all_preds).numpy(),
            average='macro'
        )

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, tab_feats, labels in tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}"):
                images, tab_feats, labels = images.to(device), tab_feats.to(device), labels.to(device)

                outputs = model(images, tab_feats)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * labels.size(0)

                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}]"
              f" | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
              f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # ------------------------
        # Save history to CSV
        # ------------------------
        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
        pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

        # ------------------------
        # Save best model
        # ------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/finetuned_best_model.pt")
            print(f"Best model updated at epoch {epoch+1} with val acc = {val_acc:.4f}")

        # ------------------------
        # Confusion matrix
        # ------------------------
        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        cm = confusion_matrix(all_labels_np, all_preds_np)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        plt.figure(figsize=(8,8))
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Multimodal Confusion Matrix (epoch {epoch+1})")
        plt.savefig(f"results/multimodal_cm_epoch{epoch+1}.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(classification_report(all_labels_np, all_preds_np, target_names=CLASS_NAMES))

    torch.save(model.state_dict(), "../models/multimodal_mae_vit_final.pth")

if __name__ == "__main__":
    main()