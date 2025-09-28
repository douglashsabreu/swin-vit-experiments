"""Check available Swin models in timm."""

import timm

def list_swin_models():
    """List all available Swin models."""
    all_models = timm.list_models('swin*')
    
    print("Available Swin models:")
    for model in all_models:
        print(f"  {model}")
    
    # Filter for 384 models
    models_384 = [m for m in all_models if '384' in m]
    print(f"\nModels with 384 support:")
    for model in models_384:
        print(f"  {model}")

if __name__ == "__main__":
    list_swin_models()
