#!/usr/bin/env python3
"""
Script to extract weights from saved model files and save as state_dicts
Run this once to convert your saved models to state_dict format
"""

import torch
from pathlib import Path
from models.cnn_model import LivenessCNN
from models.efficientnet_model import EfficientNetLiveness

def convert_models():
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # Try to load and convert CNN
    cnn_path = weights_dir / 'cnn_livness.pt'
    if cnn_path.exists():
        print(f"Processing CNN model...")
        try:
            # Try to load as state_dict first
            cnn_state = torch.load(cnn_path, map_location='cpu', weights_only=False)
            
            if isinstance(cnn_state, dict):
                print(f"  ✓ CNN is a state_dict with {len(cnn_state)} keys")
                # Save it again for clarity
                torch.save(cnn_state, cnn_path)
            else:
                print(f"  Loaded as: {type(cnn_state)}")
                if hasattr(cnn_state, 'state_dict'):
                    print(f"  Has state_dict method, extracting...")
                    state = cnn_state.state_dict()
                    print(f"  ✓ Extracted {len(state)} state keys")
                    torch.save(state, cnn_path)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Try to load and convert EfficientNet
    eff_path = weights_dir / 'efficientnet.pt'
    if eff_path.exists():
        print(f"\nProcessing EfficientNet model...")
        try:
            # Try to load
            eff_data = torch.load(eff_path, map_location='cpu', weights_only=False)
            
            if isinstance(eff_data, dict):
                print(f"  ✓ EfficientNet is a state_dict with {len(eff_data)} keys")
                # Show some keys to understand structure
                keys = list(eff_data.keys())[:5]
                print(f"  Sample keys: {keys}")
                torch.save(eff_data, eff_path)
            else:
                print(f"  Loaded as: {type(eff_data)}")
                if hasattr(eff_data, 'state_dict'):
                    print(f"  Has state_dict method, extracting...")
                    state = eff_data.state_dict()
                    print(f"  ✓ Extracted {len(state)} state keys")
                    torch.save(state, eff_path)
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == '__main__':
    convert_models()
    print("\n✅ Conversion complete!")
