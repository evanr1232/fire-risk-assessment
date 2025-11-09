import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from datasets import FireRiskImageDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")  # Headless for terminal with no display
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--encoder_path", type=str, required=True)
args = parser.parse_args()

CLASS_NAMES = ["Very_Low", "Low", "Moderate", "High", "Very_High", "Non-burnable", "Water"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FireRiskImageDataset("../data/FireRisk", split="train")
    val_ds = FireRiskImageDataset("../data/FireRisk", split="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7)
    mae_state = torch.load(args.encoder_path, map_location="cpu")
    model.load_state_dict(mae_state, strict=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Train epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val epoch {epoch}/{num_epochs}"):
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

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

    torch.save(model.state_dict(), "baseline_mae_vit.pth")

if __name__ == "__main__":
    main()