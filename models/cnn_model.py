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
    import sys
    import importlib

    if weights_path is None:
        weights_path = Path(__file__).parent.parent / 'weights' / 'cnn_livness.pt'

    if isinstance(weights_path, str):
        weights_path = Path(weights_path)

    print(f"\nCNN Model:")

    if not weights_path.exists():
        print(f"  ✗ Weights not found at {weights_path}")
        model = LivenessCNN(in_size=300).to(device)
        model.eval()
        return model

    print(f"  Attempting to load from {weights_path.name}...")

    # ── Key fix: inject LivenessCNN into __main__ AND the module  ──
    # This is needed because the .pt file references __main__.LivenessCNN
    import __main__
    _had_attr = hasattr(__main__, 'LivenessCNN')
    _orig = getattr(__main__, 'LivenessCNN', None)
    __main__.LivenessCNN = LivenessCNN

    # Also register this module's path so pickle can find it either way
    current_module = sys.modules[__name__]
    if not hasattr(current_module, 'LivenessCNN'):
        current_module.LivenessCNN = LivenessCNN

    try:
        loaded = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,       # required for full-model pickles
        )

        if isinstance(loaded, nn.Module):
            # Check if loaded model is complete/has all required attributes
            if hasattr(loaded, 'avgpool') and hasattr(loaded, 'classifier'):
                loaded = loaded.to(device)
                loaded.eval()
                print(f"  ✓ CNN loaded successfully (full model)")
                return loaded
            else:
                # Model is incomplete, extract state_dict instead
                print(f"  ! Loaded model missing attributes, extracting state_dict...")
                state_dict = loaded.state_dict()
                model = LivenessCNN(in_size=300).to(device)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                print(f"  ✓ CNN weights loaded (state_dict from model)")
                return model

        elif isinstance(loaded, dict):
            # Could be a raw state_dict or a checkpoint dict
            state_dict = loaded.get('model_state_dict', loaded.get('state_dict', loaded))
            model = LivenessCNN(in_size=300).to(device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f"  ✓ CNN weights loaded (state_dict)")
            return model

        else:
            print(f"  ✗ Unexpected type in .pt file: {type(loaded)}")
            model = LivenessCNN(in_size=300).to(device)
            model.eval()
            return model

    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        model = LivenessCNN(in_size=300).to(device)
        model.eval()
        return model

    finally:
        # Restore __main__ to its original state
        if _had_attr:
            __main__.LivenessCNN = _orig
        elif hasattr(__main__, 'LivenessCNN'):
            del __main__.LivenessCNN