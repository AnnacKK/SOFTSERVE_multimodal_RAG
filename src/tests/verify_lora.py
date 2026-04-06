import torch
import sys


def verify_stack():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")

    # Check for the specific missing attribute
    try:
        # Some versions use int1 for boolean/bitmask operations in Triton
        has_int1 = hasattr(torch, 'int1')
        print(f"Does torch have 'int1' attribute? {has_int1}")
    except Exception as e:
        print(f"Error accessing torch.int1: {e}")

    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("❌ ERROR: CUDA is not available. Unsloth will not work.")


if __name__ == "__main__":
    verify_stack()
    try:

        print("✅ Unsloth imported successfully!")
    except Exception as e:
        print(f"❌ Unsloth import failed: {e}")