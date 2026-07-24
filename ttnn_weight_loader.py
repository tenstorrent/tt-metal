import torch
import ttnn
from pathlib import Path

def load_weight(device, weight_path):
    # Load torch tensor
    tensor = torch.load(weight_path)
    # Convert to ttnn tensor and send to device
    return ttnn.from_torch(tensor, device=device, layout=ttnn.TILE_LAYOUT)

def get_bark_parameters(device, weights_dir, model_type="semantic"):
    weights_dir = Path(weights_dir)
    params = {}
    
    # Filter for the specific model part (semantic, coarse, fine)
    # Bark weights are named like: semantic.transformer.h.0.attn.q_proj.weight
    for weight_file in weights_dir.glob(f"{model_type}_*.pt"):
        name = weight_file.stem.replace('_', '.')
        params[name] = load_weight(device, weight_file)
        
    return params
