"""Check which 384 models have pretrained weights."""

import timm

def check_pretrained_384_models():
    """Check which 384x384 Swin models have pretrained weights."""
    
    models_384 = [
        "swin_base_patch4_window12_384",
        "swin_large_patch4_window12_384", 
        "swinv2_cr_base_384",
        "swinv2_cr_giant_384",
        "swinv2_cr_huge_384",
        "swinv2_cr_large_384",
        "swinv2_cr_small_384",
        "swinv2_cr_tiny_384"
    ]
    
    for model_name in models_384:
        try:
            print(f"\n=== Testing {model_name} ===")
            
            # Try with pretrained=True
            try:
                model = timm.create_model(model_name, pretrained=True)
                print(f"✅ Has pretrained weights")
                
                if hasattr(model, 'num_features'):
                    print(f"   num_features: {model.num_features}")
                    
            except Exception as e:
                if "No pretrained weights" in str(e):
                    print(f"❌ No pretrained weights available")
                else:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Model not found: {e}")

if __name__ == "__main__":
    check_pretrained_384_models()
