import torch
import re
import timm
from timm.models.vision_transformer import vit_base_patch16_224, resize_pos_embed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load checkpoint and setup
ckpt_path = "./MAE_ImageNet1k.pth"  # Path to the checkpoint file
NUM_CLASSES = 7  # Modify according to the number of classes you want (e.g., fire risk categories)

# Create the model (vision transformer, no pre-trained weights initially)
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)

# Load the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", ckpt))

# Filter out the decoder and mask token layers, keeping only the encoder
def filter_for_encoder(sd):
    out = {}
    for k, v in sd.items():
        k = re.sub(r"^(module\.)?", "", k)  # Strip DDP 'module.' if present
        k = re.sub(r"^(encoder\.)?", "", k)  # Strip 'encoder.' prefix if it exists
        if k.startswith("decoder") or "mask_token" in k:  # Skip decoder layers
            continue
        out[k] = v
    return out

# Load only the encoder weights (this avoids overwriting the head)
enc_state = filter_for_encoder(state)

# Load the model weights (ensure you're only loading the encoder part)
msg = model.load_state_dict(enc_state, strict=False)

# If the image size is different, you may need to resize positional embeddings
model.pos_embed = torch.nn.Parameter(resize_pos_embed(enc_state['pos_embed'], model.pos_embed))

# Switch model to evaluation mode (no dropout, batch norm)
model.eval()

# Ensure the device is set (GPU if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # "mps" for Apple Silicon
print(f"device: {device}")
model.to(device)

# Prepare validation data (assuming you have the images stored in directories 'FireRisk/val/high', 'FireRisk/val/low', etc.)

# Define data transformation pipeline (resize, normalization)
transform = transforms.Compose([
    transforms.Resize(224),  # Ensure all images are 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats (adjust if needed)
])

# Replace this with the path to your validation dataset
val_dataset_path = './data/FireRisk/FireRisk/val'  # The parent directory of 'high', 'low', 'moderate', etc.
val_dataset = datasets.ImageFolder(val_dataset_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Evaluate the model on the validation set
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        # Move inputs and labels to the device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)  # Get the raw model predictions (logits)
        
        # Get the class predictions (index of the max logit)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        print(f"Correct so far: {correct}")

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the model on the validation dataset: {accuracy:.2f}%')
