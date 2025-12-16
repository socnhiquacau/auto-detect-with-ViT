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
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"   Checkpoint type: {type(checkpoint)}")
        
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
        
        # Try to load with DINO architecture
        try:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model.to(device)
            model.eval()
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   ✅ Model loaded successfully!")
            print(f"   Output shape: {output.shape}")
            print(f"   Feature dimension: {output.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error during model inference: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_model(model_path: str):
    """Test loading YOLO model"""
    print(f"\n🧪 Testing YOLO model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        print(f"   ✅ YOLO model loaded successfully!")
        print(f"   Model type: {model.task}")
        print(f"   Model names: {model.names}")
        
        # Check if 'person' class exists
        if 0 in model.names and model.names[0] == 'person':
            print(f"   ✅ 'person' class found at index 0")
        else:
            print(f"   ⚠️  Warning: 'person' class not at index 0")
            print(f"   Available classes: {model.names}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_pth_file(model_path: str):
    """Inspect .pth file structure"""
    print(f"\n🔍 Inspecting .pth file: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ File not found: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"   Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"     - {key}: Tensor {value.shape}")
                elif isinstance(value, dict):
                    print(f"     - {key}: Dict with {len(value)} items")
                    if len(value) <= 5:
                        print(f"       {list(value.keys())}")
                else:
                    print(f"     - {key}: {type(value)}")
        
        print("   ✅ Inspection complete")
        
    except Exception as e:
        print(f"   ❌ Error inspecting file: {e}")

def main():
    print("=" * 60)
    print("🔬 Model Loading Test Suite")
    print("=" * 60)
    
    # Model paths
    vit_model_path = os.getenv('VIT_MODEL_PATH', 'models/dino_vits16_epoch100.pth')
    yolo_model_path = os.getenv('YOLO_MODEL_PATH', 'models/yolov8_person_detection.pt')
    
    # Test ViT model
    vit_success = test_vit_model(vit_model_path)
    
    # Inspect ViT .pth file
    if Path(vit_model_path).exists():
        inspect_pth_file(vit_model_path)
    
    # Test YOLO model
    yolo_success = test_yolo_model(yolo_model_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"ViT Model:  {'✅ PASSED' if vit_success else '❌ FAILED'}")
    print(f"YOLO Model: {'✅ PASSED' if yolo_success else '❌ FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()