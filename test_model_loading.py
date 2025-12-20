"""
Script to test loading your trained models
Usage: python test_model_loading.py
"""

import torch
import os
from pathlib import Path

def test_vit_model(model_path: str):
    """Test loading ViT model from .pth file"""
    print(f"\n🧪 Testing ViT model: {model_path}")

    Uses unified load_model_from_models() which handles:
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        # Use unified model loader from common.py
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"   Checkpoint type: {type(checkpoint)}")
        print(f"   ✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Model device: {next(model.parameters()).device}")

        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {checkpoint.keys()}")

            if 'model' in checkpoint:
                print("   ✅ Found 'model' key in checkpoint")
                print(f"   Model type: {type(checkpoint['model'])}")
            elif 'state_dict' in checkpoint:
                print("   ✅ Found 'state_dict' key in checkpoint")
            else:
                print("   ℹ️  Checkpoint is a state_dict directly")
        else:
            print("   ℹ️  Checkpoint is the model itself")
            output = model(dummy_input)
        # Try to load with DINO architecture
        try:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

            if isinstance(checkpoint, dict):
        except Exception as e:
            print(f"   ❌ Error during model inference: {e}")
            return False

                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
def test_yolo_model(model_path: str):
    """Test loading YOLO model"""
    print(f"\n🧪 Testing YOLO model: {model_path}")

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_yolo_model(model_name: str):
    """
    Test loading YOLO model from models/ directory.

    Args:
        model_name: Model filename or pattern in models/ directory
    """
    print(f"\n🧪 Testing YOLO model: {model_name}")

    try:
        from ultralytics import YOLO
        from common import find_model_file

        model_path = find_model_file(model_name)
        print(f"   Found model at: {model_path}")

def inspect_pth_file(model_path: str):
    """Inspect .pth file structure"""
    print(f"\n🔍 Inspecting .pth file: {model_path}")
        # Check if 'person' class exists
        if 0 in model.names and model.names[0] == 'person':
            print(f"   ✅ 'person' class found at index 0")
        else:
            print(f"   ⚠️  Warning: 'person' class not at index 0")
    if not Path(model_path).exists():
        print(f"❌ File not found: {model_path}")
        return
    except Exception as e:
        print(f"   ❌ Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
def inspect_pth_file(model_name: str):
    """
    Inspect .pth file structure from models/ directory.

    Args:
        model_name: Model filename or pattern in models/ directory
    """
    print(f"\n🔍 Inspecting .pth file: {model_name}")

    try:
        from common import find_model_file
        model_path = find_model_file(model_name)
        print(f"   Found at: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"   Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            
            for key in checkpoint.keys():
                value = checkpoint[key]
                elif isinstance(value, dict):
                    print(f"     - {key}: Dict with {len(value)} items")
                    if len(value) <= 5:
                        print(f"       {list(value.keys())}")
                else:
                    print(f"     - {key}: {type(value)}")
        print("   ✅ Inspection complete")
        
    except Exception as e:
        print(f"   ❌ Error inspecting file: {e}")
        import traceback
        traceback.print_exc()
    # Model paths
    vit_model_path = os.getenv('VIT_MODEL_PATH', 'models/dino_vits16_epoch100.pth')
    yolo_model_path = os.getenv('YOLO_MODEL_PATH', 'models/yolov8_person_detection.pt')
    print("=" * 60)
    print("🔬 Model Loading Test Suite")
    print("=" * 60)
    vit_success = test_vit_model(vit_model_path)
    # Model names (files must exist in models/ directory)
    vit_model_name = os.getenv('VIT_MODEL_PATH', 'vit_final_model.pth')
    if Path(vit_model_path).exists():
        inspect_pth_file(vit_model_path)

    # Inspect ViT .pth file
    yolo_success = test_yolo_model(yolo_model_path)
        inspect_pth_file(vit_model_name)
    except Exception as e:
        print(f"   ⚠️  Could not inspect VIT model: {e}")

    # Test YOLO model
    yolo_success = test_yolo_model(yolo_model_name)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"ViT Model:  {'✅ PASSED' if vit_success else '❌ FAILED'}")
    print(f"YOLO Model: {'✅ PASSED' if yolo_success else '❌ FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()