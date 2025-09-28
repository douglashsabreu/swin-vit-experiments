"""Test the final model configuration with 384x384 images."""

import torch
import timm

def test_swinv2_cr_tiny_384():
    """Test the SwinV2 CR Tiny 384 model."""
    model_name = "swinv2_cr_tiny_384"
    print(f"Testing {model_name} with 384x384 images")
    
    model = timm.create_model(model_name, pretrained=False, features_only=False)
    
    print(f"Model type: {type(model)}")
    if hasattr(model, 'num_features'):
        print(f"num_features: {model.num_features}")
    
    # Test with batch size 8 (same as config)
    dummy_input = torch.randn(8, 3, 384, 384)
    print(f"Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward_features'):
            features = model.forward_features(dummy_input)
            print(f"forward_features output shape: {features.shape}")
            
            if features.ndim == 4:
                pooled = features.mean(dim=(1, 2))
                print(f"After pooling shape: {pooled.shape}")
        
        output = model(dummy_input)
        print(f"Regular forward output shape: {output.shape}")
    
    print("✅ Model works correctly with 384x384 images!")

if __name__ == "__main__":
    test_swinv2_cr_tiny_384()
