import torch
import torch.nn as nn
from torchvision import models


class EfficientNetLiveness(nn.Module):
    """
    EfficientNet-B3 based Liveness Detection Model (matches your saved model)
    Input:  x float32 tensor of shape (B, 3, 300, 300)
    Output: logits of shape (B, 2)  (Real vs Fake)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load EfficientNet-B3 WITHOUT pretrained weights (will load from local file)
        self.model = models.efficientnet_b3(weights=None)
        
        # Freeze backbone (as in your training setup)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final classifier to 2 classes (as in your training setup)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        # Unfreeze classifier for inference
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def load_efficientnet_model(weights_path=None, device='cpu', pretrained=True):
    """Load EfficientNet model with pretrained weights"""
    from pathlib import Path
    import torch
    
    # Try to load full model first
    if weights_path is None:
        weights_path = Path(__file__).parent.parent / 'weights' / 'efficientnet_b3_liveness_full.pt'
    
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    print(f"\nEfficientNet Model:")
    
    # Check if weights exist before creating model
    if not weights_path.exists():
        raise FileNotFoundError(f"❌ EfficientNet weights not found at {weights_path}. Please ensure the file exists.")
    
    # Create base model first (without pretrained)
    model = EfficientNetLiveness(pretrained=False)
    
    # Load weights from local file
    print(f"  Attempting to load from {weights_path.name}...")
    try:
        loaded = torch.load(weights_path, map_location=device, weights_only=False)
        
        # If it's a full model, use it directly
        if isinstance(loaded, EfficientNetLiveness):
            loaded = loaded.to(device)
            loaded.eval()
            print(f"  ✓ EfficientNet loaded successfully (full model)")
            return loaded
        
        # If it's a state_dict, load it
        elif isinstance(loaded, dict):
            try:
                model.load_state_dict(loaded, strict=False)
                print(f"  ✓ EfficientNet weights loaded (state_dict)")
            except Exception as e:
                print(f"  ✗ Could not load state_dict: {e}")
                raise
    
    except TypeError:
        # Older PyTorch - try without weights_only
        try:
            loaded = torch.load(weights_path, map_location=device)
            if isinstance(loaded, EfficientNetLiveness):
                loaded = loaded.to(device)
                loaded.eval()
                print(f"  ✓ EfficientNet loaded successfully (full model)")
                return loaded
            elif isinstance(loaded, dict):
                try:
                    model.load_state_dict(loaded, strict=False)
                    print(f"  ✓ EfficientNet weights loaded (state_dict)")
                except Exception as e:
                    print(f"  ✗ Could not load: {e}")
                    raise
        except Exception as e:
            print(f"  ✗ torch.load failed: {e}")
            raise
    except Exception as e:
        print(f"  ✗ torch.load failed: {e}")
        raise
    
    # Return model with loaded weights
    model = model.to(device)
    model.eval()
    return model
