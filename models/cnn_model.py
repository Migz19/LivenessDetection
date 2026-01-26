import torch
import torch.nn as nn
import torch.nn.functional as F


class LivenessCNN(nn.Module):
    """
    Liveness Detection CNN Model
    Input:  x float32 tensor of shape (B, 3, H, W)  (recommended H=W=300)
    Output: logits of shape (B, 2)  (use CrossEntropyLoss + argmax)
    """
    def __init__(self, in_size=300):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 300 -> 150
            block(32, 64),   # 150 -> 75
            block(64, 128),  # 75 -> 37
            block(128, 256), # 37 -> 18
            block(256, 256), # 18 -> 9
        )
        
        # Adaptive average pooling to get fixed output size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification: Real vs Fake
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_cnn_model(weights_path=None, device='cpu'):
    """Load CNN model with pretrained weights"""
    from pathlib import Path
    import torch
    
    # Try to load full model first
    if weights_path is None:
        weights_path = Path(__file__).parent.parent / 'weights' / 'cnn_livness.pt'
    
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    # Create base model first
    model = LivenessCNN(in_size=300)
    model = model.to(device)
    
    print(f"\nCNN Model:")
    
    # Try to load weights
    if weights_path.exists():
        print(f"  Attempting to load from {weights_path.name}...")
        try:
            loaded = torch.load(weights_path, map_location=device, weights_only=False)
            
            # If it's a full model, use it directly
            if isinstance(loaded, LivenessCNN):
                loaded = loaded.to(device)
                loaded.eval()
                print(f"  ✓ CNN loaded successfully (full model)")
                return loaded
            
            # If it's a state_dict, load it
            elif isinstance(loaded, dict):
                try:
                    model.load_state_dict(loaded, strict=False)
                    print(f"  ✓ CNN weights loaded (state_dict)")
                except Exception as e:
                    print(f"    Could not load state_dict: {e}")
        
        except TypeError:
            # Older PyTorch - try without weights_only
            try:
                loaded = torch.load(weights_path, map_location=device)
                if isinstance(loaded, LivenessCNN):
                    loaded = loaded.to(device)
                    loaded.eval()
                    print(f"  ✓ CNN loaded successfully (full model)")
                    return loaded
                elif isinstance(loaded, dict):
                    try:
                        model.load_state_dict(loaded, strict=False)
                        print(f"  ✓ CNN weights loaded (state_dict)")
                    except Exception as e:
                        print(f"    Could not load: {e}")
            except Exception as e:
                print(f"    torch.load failed: {e}")
        except Exception as e:
            print(f"    torch.load failed: {e}")
    
    # Return model (either with loaded weights or random init)
    print(f"  ✓ CNN Model initialized")
    model.eval()
    return model
