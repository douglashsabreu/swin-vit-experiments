"""Debug script to understand the model dimensions issue."""

import torch
import timm
from pathlib import Path
import sys

def debug_swin_model():
    """Debug the Swin Transformer model to understand dimension issues."""
    
    # Create the Swin model
    model_name = "swin_tiny_patch4_window7_224"
    model = timm.create_model(model_name, pretrained=True, features_only=False)
    
    print(f"Model: {model_name}")
    print(f"Model type: {type(model)}")
    
    # Check model attributes
    if hasattr(model, 'num_features'):
        print(f"num_features: {model.num_features}")
    
    if hasattr(model, 'head') or hasattr(model, 'fc'):
        head = getattr(model, 'head', None) or getattr(model, 'fc', None)
        print(f"Head type: {type(head)}")
        if hasattr(head, 'in_features'):
            print(f"Head in_features: {head.in_features}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        # Test different forward methods
        try:
            if hasattr(model, 'forward_features'):
                features = model.forward_features(dummy_input)
                print(f"forward_features output shape: {features.shape if hasattr(features, 'shape') else type(features)}")
                if isinstance(features, (list, tuple)):
                    print(f"Features is list/tuple with {len(features)} elements")
                    for i, f in enumerate(features):
                        if hasattr(f, 'shape'):
                            print(f"  Element {i} shape: {f.shape}")
            else:
                print("No forward_features method")
        except Exception as e:
            print(f"Error with forward_features: {e}")
        
        # Test regular forward
        try:
            output = model(dummy_input)
            print(f"Regular forward output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
            if isinstance(output, (list, tuple)):
                print(f"Output is list/tuple with {len(output)} elements")
                for i, o in enumerate(output):
                    if hasattr(o, 'shape'):
                        print(f"  Element {i} shape: {o.shape}")
        except Exception as e:
            print(f"Error with regular forward: {e}")

def debug_batch_processing():
    """Debug with batch size similar to error."""
    print("\n=== Testing with batch size 8 ===")
    
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=False)
    dummy_input = torch.randn(8, 3, 224, 224)
    print(f"Batch input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward_features'):
            features = model.forward_features(dummy_input)
            print(f"Batch forward_features output shape: {features.shape if hasattr(features, 'shape') else type(features)}")
            
            # Test pooling if spatial
            if hasattr(features, 'ndim') and features.ndim == 4:
                pooled = features.mean(dim=(-2, -1))
                print(f"After pooling shape: {pooled.shape}")
            elif hasattr(features, 'shape') and len(features.shape) > 2:
                print(f"Features has more than 2 dimensions: {features.shape}")

if __name__ == "__main__":
    debug_swin_model()
    debug_batch_processing()
