"""Test model with the fix applied."""

import torch
import sys
sys.path.append('src')

from factories.model_factory import _SwinWithHead
import timm

def test_model_with_fix():
    """Test the model with the pooling fix."""
    model_name = "swinv2_cr_tiny_384"
    backbone = timm.create_model(model_name, pretrained=False, features_only=False)
    
    # Create a simple head for testing
    head = torch.nn.Linear(768, 4)  # 4 classes
    
    # Create the wrapper model
    model = _SwinWithHead(backbone, head)
    
    # Test with batch size 8
    dummy_input = torch.randn(8, 3, 384, 384)
    print(f"Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        # Test forward features first
        features = backbone.forward_features(dummy_input)
        print(f"Backbone features shape: {features.shape}")
        
        # Test the full model
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Expected output shape: [8, 4]")
        
        if output.shape == (8, 4):
            print("✅ Model works correctly!")
        else:
            print("❌ Model output shape is incorrect")

if __name__ == "__main__":
    test_model_with_fix()
