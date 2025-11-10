#!/usr/bin/env python
"""
Setup and verification script for PACS pruning project.
Run this to check if all components are properly configured.
"""

import os
import sys


def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'scikit-learn',
        'tqdm',
        'PIL',
        'tensorboard',
        'torchmetrics',
        'timm',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed!")
    return True


def check_project_structure():
    """Verify project directory structure is correct."""
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)
    
    expected_dirs = [
        'artifacts',
        'artifacts/baselines',
        'artifacts/results',
        'configs',
        'data_handling',
        'models',
        'analysis',
        'testing',
        'utils'
    ]
    
    all_good = True
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            all_good = False
    
    if all_good:
        print("\nProject structure is correct!")
    else:
        print("\nSome directories are missing. Creating them...")
        for dir_path in expected_dirs:
            os.makedirs(dir_path, exist_ok=True)
        print("Directories created!")
    
    return True


def check_config():
    """Check if configuration is properly set."""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    try:
        from configs import pacs_config as config
        
        print(f"✓ Configuration loaded")
        print(f"  - Domains: {config.DOMAINS}")
        print(f"  - Hidden sizes: {config.HIDDEN_SIZES}")
        print(f"  - Pruning target: {config.PRUNING_TARGET_PROPORTION:.1%}")
        print(f"  - Selection method: {config.SELECTION_METHOD}")
        
        # Check data path
        if config.DATA_ROOT == "/path/to/pacs/dataset":
            print(f"\n⚠ WARNING: DATA_ROOT is not configured!")
            print(f"  Please update DATA_ROOT in configs/pacs_config.py")
            print(f"  Current value: {config.DATA_ROOT}")
            return False
        elif not os.path.exists(config.DATA_ROOT):
            print(f"\n⚠ WARNING: DATA_ROOT does not exist!")
            print(f"  Path: {config.DATA_ROOT}")
            print(f"  Please check the path or download the PACS dataset")
            return False
        else:
            print(f"  - Data root: {config.DATA_ROOT}")
            
            # Check for PACS domains
            domains_found = []
            for domain in config.DOMAINS:
                domain_path = os.path.join(config.DATA_ROOT, domain)
                if os.path.exists(domain_path):
                    domains_found.append(domain)
            
            if len(domains_found) == len(config.DOMAINS):
                print(f"  - All domains found: {domains_found}")
            else:
                missing = set(config.DOMAINS) - set(domains_found)
                print(f"\n⚠ WARNING: Missing domains: {missing}")
                print(f"  Found: {domains_found}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False


def run_simple_test():
    """Run a simple test to verify core functionality."""
    print("\n" + "="*60)
    print("RUNNING SIMPLE FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        # Test imports
        print("Testing imports...")
        from models.pacs_net import PACSNet
        from models.pruning import rebuild_pruned_model
        import torch
        import numpy as np
        
        print("✓ Imports successful")
        
        # Test model creation
        print("\nTesting model creation...")
        model = PACSNet(num_classes=7, hidden_sizes=[512, 256])
        print(f"✓ Model created with {model.get_total_mlp_neurons()} neurons")
        
        # Test pruning
        print("\nTesting pruning logic...")
        total_neurons = model.get_total_mlp_neurons()
        selected = list(range(0, total_neurons, 2))  # Select every other neuron
        
        class DummyConfig:
            NUM_CLASSES = 7
            HIDDEN_SIZES = [512, 256]
        
        pruned_model = rebuild_pruned_model(model, selected, DummyConfig())
        print(f"✓ Pruned model created with {pruned_model.get_total_mlp_neurons()} neurons")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = pruned_model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        print("\n✓ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup verification."""
    print("\n" + "="*80)
    print("PACS PRUNING PROJECT - SETUP VERIFICATION")
    print("="*80)
    
    results = []
    
    # Check dependencies
    deps_ok = check_dependencies()
    results.append(("Dependencies", deps_ok))
    
    if not deps_ok:
        print("\n" + "="*80)
        print("Please install missing dependencies before continuing.")
        print("Run: pip install -r requirements.txt")
        print("="*80)
        sys.exit(1)
    
    # Check project structure
    structure_ok = check_project_structure()
    results.append(("Project Structure", structure_ok))
    
    # Check configuration
    config_ok = check_config()
    results.append(("Configuration", config_ok))
    
    # Run simple test
    test_ok = run_simple_test()
    results.append(("Functionality Test", test_ok))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, status in results:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:20} {status_str}")
    
    if all(status for _, status in results):
        print("\n✨ All checks passed! The project is ready to run.")
        print("\nNext steps:")
        print("1. Run a single experiment: python run_experiment.py sketch 42")
        print("2. Run full experiments: python main.py")
        print("3. Run unit tests: python testing/test_pruning.py")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        if not config_ok:
            print("\nMost likely you need to:")
            print("1. Download the PACS dataset")
            print("2. Update DATA_ROOT in configs/pacs_config.py")
    
    print("="*80)


if __name__ == "__main__":
    main()
