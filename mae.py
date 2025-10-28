import torch, re
import timm
from timm.models.vision_transformer import vit_base_patch16_224

ckpt_path = "./MAE_ImageNet1k.pth"

# load the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", ckpt)) 

model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7)

# keep only encoder weights; drop MAE-specific parts (decoder, mask token, etc.)
def filter_for_encoder(sd):
    out = {}
    for k, v in sd.items():
        k = re.sub(r"^(module\.)?", "", k)          # strip DDP 'module.' if present
        k = re.sub(r"^(encoder\.)?", "", k)         # MAE checkpoints often prefix 'encoder.'
        if k.startswith("decoder") or "mask_token" in k:
            continue
        out[k] = v
    return out

enc_state = filter_for_encoder(state)

msg = model.load_state_dict(enc_state, strict=False)

## If your input size ≠ 224, you may need to resize positional embeddings.
## Timm has a helper you can use BEFORE loading state dict:
# from timm.models.vision_transformer import resize_pos_embed
# model.pos_embed = torch.nn.Parameter(resize_pos_embed(enc_state['pos_embed'], model.pos_embed))

## Replace head if needed (timm already created it with YOUR_NUM_CLASSES)
# model.head = torch.nn.Linear(model.head.in_features, YOUR_NUM_CLASSES)

## Train: typically AdamW, lr ~ 5e-4 for head, lower for backbone (or use layer-wise decay)
