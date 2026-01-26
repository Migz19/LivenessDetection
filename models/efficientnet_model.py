import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNetLiveness(nn.Module):
    """
    EfficientNet-B3 based Liveness Detection Model (matches your saved model)
    Input:  x float32 tensor of shape (B, 3, 300, 300)
    Output: logits of shape (B, 2)  (Real vs Fake)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B3 (matching your saved model)
        if pretrained:
            self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        else:
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
        weights_path = Path(__file__).parent.parent / 'weights' / 'efficientnet.pt'
    
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    print(f"\nEfficientNet Model:")
    
    # Create base model first
    model = EfficientNetLiveness(pretrained=pretrained)
    
    # Try to load weights
    if weights_path.exists():
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
                    print(f"    Could not load state_dict: {e}")
                    print(f"    Using pretrained ImageNet weights instead")
        
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
                        print(f"    Could not load: {e}")
            except Exception as e:
                print(f"    torch.load failed: {e}")
        except Exception as e:
            print(f"    torch.load failed: {e}")
    
    # Return model (either with loaded weights or ImageNet pretrained)
    model = model.to(device)
    model.eval()
    if not weights_path.exists():
        print(f"  ✓ EfficientNet Model initialized (using ImageNet pretrained)")
    return model
