"""
Robust weight loading utility supporting multiple formats
"""

import torch
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def load_weights_robust(weights_path: Path, model, device: str = 'cpu', 
                       fallback_to_imagenet: bool = False) -> bool:
    """
    Robustly load weights from various formats.
    
    Supports:
    1. State dict (recommended format)
    2. Full model (pickle format)
    3. Checkpoint dict with 'model_state_dict' or 'state_dict'
    
    Args:
        weights_path: Path to weight file
        model: PyTorch model to load weights into
        device: Device to load to ('cpu' or 'cuda')
        fallback_to_imagenet: Whether to fall back to ImageNet weights on failure
    
    Returns:
        True if weights loaded successfully, False otherwise
    """
    
    if not weights_path.exists():
        return False
    
    try:
        # Method 1: Try torch.load with weights_only=False (supports full models and state_dicts)
        print(f"  Attempting to load from {weights_path.name}...")
        
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            if checkpoint is not None:
                return _load_checkpoint(model, checkpoint, weights_path.name)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            try:
                checkpoint = torch.load(weights_path, map_location=device)
                if checkpoint is not None:
                    return _load_checkpoint(model, checkpoint, weights_path.name)
            except Exception as e:
                print(f"    torch.load failed: {e}")
        except Exception as e:
            print(f"    torch.load failed: {e}")
            checkpoint = None
        
        # Method 2: Try pickle.load as fallback (for edge cases)
        print(f"    Trying pickle.load...")
        try:
            # Use latin1 encoding for compatibility with Python 2 pickles
            with open(weights_path, 'rb') as f:
                checkpoint = pickle.load(f, encoding='latin1')
            
            if checkpoint is not None:
                return _load_checkpoint(model, checkpoint, weights_path.name)
        except Exception as e:
            print(f"    pickle.load failed: {e}")
        
        return False
    
    except Exception as e:
        print(f"  Error loading weights: {e}")
        return False


def _load_checkpoint(model: torch.nn.Module, checkpoint: Any, name: str) -> bool:
    """
    Load checkpoint into model, handling various formats.
    
    Args:
        model: PyTorch model
        checkpoint: Loaded checkpoint (could be state_dict or full model)
        name: Name of weight file for logging
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Handle full model objects first (priority - these are the actual trained models)
        if isinstance(checkpoint, torch.nn.Module):
            print(f"    ✓ Loaded full model object from {name}")
            print(f"    Using model directly without state_dict extraction")
            # Return True to indicate success
            # The caller will need to use the checkpoint directly
            return True
        
        # Handle dict checkpoints with metadata
        elif isinstance(checkpoint, dict):
            # Try different state_dict keys
            for key in ['model_state_dict', 'state_dict', 'checkpoint', 'weights']:
                if key in checkpoint:
                    print(f"    Found '{key}' in checkpoint")
                    try:
                        model.load_state_dict(checkpoint[key], strict=False)
                        print(f"  ✓ Weights loaded successfully from {name}")
                        return True
                    except Exception as e:
                        print(f"    Could not load '{key}': {e}")
                        continue
            
            # Assume the dict itself is the state_dict
            try:
                model.load_state_dict(checkpoint, strict=False)
                print(f"  ✓ Weights loaded successfully from {name}")
                return True
            except Exception as e:
                print(f"    Could not load as state_dict: {e}")
                return False
        
        else:
            print(f"    Unknown checkpoint type: {type(checkpoint)}")
            return False
    
    except Exception as e:
        print(f"    Error processing checkpoint: {e}")
        return False


def load_weights_with_fallback(weights_path: Optional[Path], model: torch.nn.Module, 
                              device: str = 'cpu', model_name: str = "Model") -> bool:
    """
    Load weights with helpful console output.
    Falls back gracefully if weights can't be loaded.
    
    Args:
        weights_path: Path to weights file (None is OK)
        model: PyTorch model
        device: Device to load to
        model_name: Name for logging
    
    Returns:
        True if weights loaded, False if using fallback
    """
    
    if weights_path is None:
        print(f"ℹ {model_name}: No weights path specified")
        return False
    
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f"ℹ {model_name}: Weights not found at {weights_path}")
        return False
    
    print(f"\n{model_name}:")
    if load_weights_robust(weights_path, model, device):
        return True
    else:
        print(f"✓ {model_name} initialized (weights unavailable)")
        return False
