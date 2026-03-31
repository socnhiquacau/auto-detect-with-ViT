#!/usr/bin/env python3
"""Script to check what's in the model file."""
import ssl
from pathlib import Path

import torch


def main() -> None:
    # Disable SSL verification (for development only)
    ssl._create_default_https_context = ssl._create_unverified_context

    print("Loading model file...")
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "best_model_state_dict.pth"
    obj = torch.load(model_path, map_location="cpu")

    print(f"\nType: {type(obj)}")

    if isinstance(obj, dict):
        print(f"\nDict keys: {list(obj.keys())}")

        # Check if it's a full model or just state_dict
        for key in ["model", "state_dict", "net", "module"]:
            if key in obj:
                print(f"\nFound '{key}' in dict")
                sub_obj = obj[key]
                print(f"Type of '{key}': {type(sub_obj)}")
                if isinstance(sub_obj, dict):
                    print(f"First 10 keys: {list(sub_obj.keys())[:10]}")

        # If dict doesn't have common checkpoint keys, it might be state_dict itself
        if not any(k in obj for k in ["model", "state_dict", "net", "module"]):
            print("\nLooks like a direct state_dict")
            print(f"First 10 keys: {list(obj.keys())[:10]}")

    elif isinstance(obj, torch.nn.Module):
        print("\nThis is a full model (nn.Module)")
        print(f"Model: {obj}")

    print("\n[OK] Model file loaded successfully")


if __name__ == "__main__":
    main()
