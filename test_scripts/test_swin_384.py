"""Test Swin models with 384x384 images."""

import torch
import timm

def test_swin_models_384():
    """Test various Swin models with 384x384 input."""
    
    models_to_test = [
        "swin_tiny_patch4_window7_224",  # Original
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "swin_tiny_patch4_window12_384",  # Native 384 support
        "swin_small_patch4_window12_384", 
        "swin_base_patch4_window12_384"
    ]
    
    dummy_input = torch.randn(2, 3, 384, 384)
    
    for model_name in models_to_test:
        try:
            print(f"\n=== Testing {model_name} ===")
            
            model = timm.create_model(model_name, pretrained=False, features_only=False)
            
            if hasattr(model, 'num_features'):
                print(f"num_features: {model.num_features}")
            
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(dummy_input)
                    print(f"forward_features output shape: {features.shape}")
                    
                    # Test pooling
                    if features.ndim == 4:
                        pooled = features.mean(dim=(1, 2))
                        print(f"After pooling shape: {pooled.shape}")
                
                output = model(dummy_input)
                print(f"Regular forward output shape: {output.shape}")
                
        except Exception as e:
            print(f"Error with {model_name}: {e}")

if __name__ == "__main__":
    test_swin_models_384()
