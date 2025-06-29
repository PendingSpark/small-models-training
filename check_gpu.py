#!/usr/bin/env python3
"""
GPU availability and capabilities checker script.
Checks CUDA availability, GPU properties, bf16 support, and memory information.
"""

import torch
import sys


def check_gpu_capabilities():
    """Check and display GPU capabilities."""
    print("=" * 60)
    print("GPU Availability and Capabilities Check")
    print("=" * 60)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"\n1. CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\nNo CUDA-capable GPU detected.")
        print("Running on CPU only.")
        return
    
    # Get number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs detected: {gpu_count}")
    
    # Check each GPU
    for i in range(gpu_count):
        print(f"\n{'-' * 40}")
        print(f"GPU {i} Information:")
        print(f"{'-' * 40}")
        
        # Set current device
        torch.cuda.set_device(i)
        
        # 2. Show GPU name and properties
        gpu_name = torch.cuda.get_device_name(i)
        print(f"\n2. GPU Name: {gpu_name}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice Properties:")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Multi-processor Count: {props.multi_processor_count}")
        print(f"  - CUDA Capability: {props.major}.{props.minor}")
        
        # 3. Check bf16 support
        print(f"\n3. BF16 Support:")
        # BF16 is supported on compute capability 8.0 and above (Ampere and newer)
        bf16_supported = props.major >= 8
        print(f"  - Hardware BF16 Support: {bf16_supported}")
        
        # Also check torch support
        try:
            # Try to create a small bf16 tensor
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=f'cuda:{i}')
            torch_bf16_support = True
        except:
            torch_bf16_support = False
        
        print(f"  - PyTorch BF16 Support: {torch_bf16_support}")
        
        # 4. Show available memory
        print(f"\n4. Memory Information:")
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free_memory = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3
        
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Allocated Memory: {allocated:.2f} GB")
        print(f"  - Reserved Memory: {reserved:.2f} GB")
        print(f"  - Free Memory: {free_memory:.2f} GB")
        
        # Additional useful information
        print(f"\nAdditional Information:")
        print(f"  - PyTorch Version: {torch.__version__}")
        print(f"  - CUDA Version (PyTorch built with): {torch.version.cuda}")
        
        # Check if mixed precision is available
        try:
            from torch.cuda.amp import autocast
            amp_available = True
        except ImportError:
            amp_available = False
        
        print(f"  - Automatic Mixed Precision Available: {amp_available}")


def main():
    """Main function to run GPU checks."""
    try:
        check_gpu_capabilities()
    except Exception as e:
        print(f"\nError occurred while checking GPU: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("GPU check completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
  
