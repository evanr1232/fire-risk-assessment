import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


EMB_PARQUET_TRAIN = "vit_preds_embedding_train.parquet"
EMB_PARQUET_VAL = "vit_preds_embedding_val.parquet"
CSV_PATH = "data/NRI_Table_Counties/image_county_data_with_burnprob.csv"

EPOCHS = 400
BATCH_SIZE = 256
LR = 3e-5
DECAY = 1e-4
EMB_DIM = 768
NUM_CLASSES = 7


def to_xy(df):
    emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]
    X_emb = df[emb_cols].to_numpy(dtype=np.float32)
    X_tab = df[["latitude","longitude","Wildfire_Risk_Score_NRI"]].to_numpy(dtype=np.float32)
    return X_emb, X_tab, df["label"].to_numpy(dtype=np.int64)

def main():
    # load data
    emb_cols = ["filename", "label"] + [f"emb_{i}" for i in range(EMB_DIM)]
    df_tr_e = pd.read_parquet(EMB_PARQUET_TRAIN)[emb_cols]
    df_va_e = pd.read_parquet(EMB_PARQUET_VAL)[emb_cols]
    df_c    = pd.read_csv(CSV_PATH)[["filename","latitude","longitude","Wildfire_Risk_Score_NRI"]]

    # merge with CSV
    df_tr = df_tr_e.merge(df_c, on="filename", how="inner")
    df_va = df_va_e.merge(df_c, on="filename", how="inner")

    df_tr = df_tr_e.merge(df_c, on="filename", how="inner")
    df_va = df_va_e.merge(df_c, on="filename", how="inner")

    # DROP NaNs
    need = ["latitude","longitude","Wildfire_Risk_Score_NRI"] + [f"emb_{i}" for i in range(EMB_DIM)]
    df_tr = df_tr.dropna(subset=need).reset_index(drop=True)
    df_va = df_va.dropna(subset=need).reset_index(drop=True)

    # build arrays
    Xtr_emb, Xtr_tab, ytr = to_xy(df_tr)
    Xva_emb, Xva_tab, yva = to_xy(df_va)

    # standardization
    m = Xtr_tab.mean(0, keepdims=True); s = Xtr_tab.std(0, keepdims=True) + 1e-6
    Xtr_tab = (Xtr_tab - m) / s
    Xva_tab = (Xva_tab - m) / s

    Xtr = np.concatenate([Xtr_emb, Xtr_tab], axis=1).astype(np.float32)
    Xva = np.concatenate([Xva_emb, Xva_tab], axis=1).astype(np.float32)

    Xtr = torch.from_numpy(Xtr); ytr = torch.from_numpy(ytr)
    Xva = torch.from_numpy(Xva); yva = torch.from_numpy(yva)

    # model
    in_dim = EMB_DIM + 3

    model = torch.nn.Linear(in_dim, NUM_CLASSES)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY) 
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"using device {device}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    def batches(Xt, yt, bs):
        for i in range(0, len(Xt), bs):
            yield Xt[i:i+bs], yt[i:i+bs]

    # train
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in batches(Xtr, ytr, BATCH_SIZE):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(yb)
        tr_acc = correct / max(1, tot)

        # validate
        model.eval()
        with torch.no_grad():
            xv, yv = Xva.to(device), yva.to(device)
            logits = model(xv)
            va_loss = crit(logits, yv).item()
            preds = logits.argmax(1)
            va_acc  = (logits.argmax(1) == yv).float().mean().item()

        # F1 (macro & weighted)
        y_true = yv.cpu().numpy()
        y_pred = preds.cpu().numpy()
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"epoch {ep:02d} | train_acc {tr_acc:.3f} | "
            f"val_acc {va_acc:.5f} | val_loss {va_loss:.4f} | "
            f"F1(macro) {f1_macro:.4f} | F1(weighted) {f1_weighted:.3f}")
        # print(f"epoch {ep:02d} | train_acc {tr_acc:.3f} | val_acc {va_acc:.3f} | val_loss {va_loss:.4f}")

if __name__ == "__main__":
    main()
