#!/usr/bin/env python3
"""
Setup script to check dependencies and prepare for Enhanced AI training
"""

import sys
import subprocess
import importlib

def check_dependency(package_name, install_name=None):
    """Check if a package is installed"""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is missing")
        print(f"  Install with: pip install {install_name}")
        return False

def install_dependencies():
    """Install missing dependencies"""
    dependencies = [
        ('numpy', 'numpy'),
        ('numba', 'numba'), 
        ('matplotlib', 'matplotlib')
    ]
    
    missing = []
    for package, install_name in dependencies:
        if not check_dependency(package, install_name):
            missing.append(install_name)
    
    if missing:
        print(f"\nInstalling missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✓ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            print("Try manually: pip install numpy numba matplotlib")
            return False
    else:
        print("✓ All dependencies are installed!")
        return True

def check_files():
    """Check if training files exist"""
    import os
    
    required_files = [
        'yaniv_neural_network_enhanced.py',
        'genetic_algorithm_enhanced.py', 
        'train_enhanced_quick.py',
        'convert_enhanced_model.py'
    ]
    
    print("\nChecking training files:")
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            all_present = False
    
    return all_present

def show_training_options():
    """Show available training options"""
    print("\n" + "="*50)
    print("ENHANCED AI TRAINING OPTIONS")
    print("="*50)
    
    print("\n1. QUICK START (Recommended for first time):")
    print("   python3 train_enhanced_quick.py")
    print("   - 15-30 minutes")
    print("   - 20 generations") 
    print("   - Creates quick_enhanced_best.json")
    
    print("\n2. FULL TRAINING (Better performance):")
    print("   python3 genetic_algorithm_enhanced.py")
    print("   - 1-2 hours")
    print("   - 50+ generations")
    print("   - Creates enhanced_yaniv_best.json")
    
    print("\n3. CUSTOM TRAINING:")
    print("   python3 custom_training.py")
    print("   - Configure your own parameters")
    print("   - See TRAINING_GUIDE.md for details")
    
    print("\n4. TESTING:")
    print("   python3 test_enhanced_ai.py")
    print("   - Compare AI versions")
    print("   - Performance analysis")
    
    print("\n5. CONVERT TO WEB:")
    print("   python3 convert_enhanced_model.py") 
    print("   - Updates browser model")
    print("   - Use after training")
    
    print("\n" + "="*50)
    print("After training, refresh your browser to use the new model!")
    print("="*50)

def main():
    print("Enhanced AI Training Setup")
    print("="*40)
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("✗ Python 3.6+ required")
        return False
    else:
        print(f"✓ Python {sys.version.split()[0]}")
    
    # Check and install dependencies
    if not install_dependencies():
        return False
    
    # Check files
    if not check_files():
        print("\n✗ Some training files are missing")
        print("Make sure you have all the enhanced AI files")
        return False
    
    print("\n✅ Setup complete! Ready to train Enhanced AI")
    show_training_options()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)