"""Quick test script to verify the project setup."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from src.utils import DEVICE, NUM_EPOCHS
        print("  ✓ src.utils imported successfully")
        
        from src.data import load_dataset_paths, get_eval_transform
        print("  ✓ src.data imported successfully")
        
        from src.model import create_model
        print("  ✓ src.model imported successfully")
        
        from src.training import create_trainer
        print("  ✓ src.training imported successfully")
        
        from src.evaluation import evaluate_model
        print("  ✓ src.evaluation imported successfully")
        
        print("\n✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False


def test_device():
    """Test device availability."""
    print("\nTesting device setup...")
    
    import torch
    from src.utils import DEVICE
    
    print(f"  Device: {DEVICE}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print("✓ Device test passed!")


def test_model_creation():
    """Test if model can be created."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.model import create_model
        from src.utils import DEVICE
        
        model = create_model()
        model = model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {total_params:,}")
        print("✓ Model creation test passed!")
        return True
    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ARRIVE-ALIVE PROJECT SETUP TEST")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Device", test_device() is not None))
    results.append(("Model Creation", test_model_creation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Ensure dataset is in: Driver Drowsiness Dataset (DDD)/")
        print("  2. Run training: python train.py")
        print("  3. Start API: python -m uvicorn api.main:app --reload")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
