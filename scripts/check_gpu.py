#!/usr/bin/env python3
"""Check GPU/CUDA availability and system info."""

import sys

def check_gpu():
    """Check if GPU is available and print system info."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed!")
        print("Install with: pip install torch")
        return False

    print("=" * 60)
    print("GPU / CUDA Status Check")
    print("=" * 60)

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {'YES' if cuda_available else ' NO'}")

    if cuda_available:
        # CUDA version
        print(f"CUDA version: {torch.version.cuda}")

        # GPU count
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")

        # GPU details
        print("\nGPU Devices:")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")

        # Current device
        current_device = torch.cuda.current_device()
        print(f"\nCurrent device: {current_device} ({torch.cuda.get_device_name(current_device)})")

        # Memory info
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")

        print("\nGPU is ready for training!")
        print("   Training will automatically use CUDA")

    else:
        print("\nGPU not available - will train on CPU")
        print("\nPossible reasons:")
        print("  1. No GPU hardware installed")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA:")
        print("  Visit: https://pytorch.org/get-started/locally/")
        print("  Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print("=" * 60)

    return cuda_available


if __name__ == "__main__":
    has_gpu = check_gpu()
    sys.exit(0 if has_gpu else 1)
