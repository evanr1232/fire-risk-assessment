import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchgeo.datasets import FireRisk
from torchgeo.datamodules import FireRiskDataModule  # optional, for data splits etc.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_vis", type=int, required=True, default="5")
args = parser.parse_args()

def fire_risk_transform(sample):
    # Apply transforms only to the image tensor
    image = sample["image"]

    image = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(image)
    sample["image"] = image
    return sample


def visualize_sample(sample, class_names=["high", "low", "moderate", "non-burnable", "very_high", "very_low", "water"], figsize=(8, 4)):
    """
    Visualize one sample from FireRisk.
    sample is a dict with keys like 'image', 'mask' / 'label', etc.
    """
    img = sample["image"]  # torch.Tensor, shape [C, H, W]
    label = sample["label"] if "label" in sample else None

    # Image shape (3, 320, 320) --> (320, 320, 3) (numpy standard)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    # Normalize values (originally outside of 0-255 normal rgb range)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 2 if label is not None else 1, figsize=figsize)
    if label is not None:
        ax_img, ax_lbl = axes
    else:
        ax_img = axes

    ax_img.imshow(img_np)
    ax_img.set_title("Image")
    ax_img.axis("off")

    if label is not None:
        # Overlay label number and name
        lbl = label.item() if isinstance(label, torch.Tensor) else label
        ax_lbl.text(
            0.5,
            0.5,
            f"Label: {lbl} \n{class_names[lbl]}",
            ha="center",
            va="center",
            fontsize=16,
        )
        ax_lbl.axis("off")
        ax_lbl.set_title("Fire Risk Data Snapshot")

    plt.tight_layout()
    plt.show()


def main(
    root: str = "data",
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
):

    # Init the dataset
    ds = FireRisk(root=root, split=split, transforms=fire_risk_transform, download=True)
    print("Dataset size:", len(ds))
    print("Keys for sample:", ds[0].keys())

    # Skip visualization if num_vis is set to 0
    if args["num_vis"] != 0:
        # Use a DataLoader -- can be used later to feed into model training loop
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


        # Visualize a few samples
        for i, sample in enumerate(loader):
            # sample is a batch (a dict of batched tensors)
            # Visualize one in the batch

            if i >= args["num_vis"]:
                break
            single = {k: v[i] for k, v in sample.items()}
            visualize_sample(single)


if __name__ == "__main__":
    main()