#!/usr/bin/env python3
"""
Dependencies Installation Script
===============================
This script installs all required dependencies for:
1. excel_processor_combined.py - Excel/CSV processing with opcodes extraction and formatting
2. create_integrated_visualization.py - Operations analysis visualization

Usage:
    python install_dependencies.py [--upgrade] [--user]

Options:
    --upgrade    Upgrade packages to latest versions
    --user       Install packages for current user only (no admin required)
"""

import subprocess
import sys
import os
from pathlib import Path

# Required packages for both scripts
REQUIRED_PACKAGES = [
    # Core data processing (excel_processor_combined.py)
    "pandas>=1.5.0",  # DataFrame operations and Excel/CSV reading
    "openpyxl>=3.0.0",  # Excel file manipulation and formatting
    "xlsxwriter>=3.0.0",  # Alternative Excel writer (backup)
    # Visualization (create_integrated_visualization.py)
    "matplotlib>=3.5.0",  # Plotting and visualization
    "seaborn>=0.11.0",  # Statistical visualization
    "numpy>=1.21.0",  # Numerical operations
    # Optional but recommended
    "xlrd>=2.0.0",  # Reading older Excel formats (.xls)
    "Pillow>=8.0.0",  # Image processing for plot exports
]

# System packages (for Linux systems)
SYSTEM_PACKAGES = {
    "ubuntu": [
        "python3-dev",
        "python3-pip",
        "python3-venv",
        "build-essential",
        "libfreetype6-dev",
        "libpng-dev",
        "pkg-config",
    ],
    "centos": ["python3-devel", "python3-pip", "gcc", "gcc-c++", "freetype-devel", "libpng-devel", "pkgconfig"],
}


def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=capture_output, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e.stderr if e.stderr else str(e)}")
        return None


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        print("   Please upgrade Python before continuing.")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def detect_system():
    """Detect the operating system"""
    try:
        with open("/etc/os-release", "r") as f:
            content = f.read().lower()
            if "ubuntu" in content or "debian" in content:
                return "ubuntu"
            elif "centos" in content or "rhel" in content or "fedora" in content:
                return "centos"
    except FileNotFoundError:
        pass

    return None


def install_system_dependencies():
    """Install system-level dependencies"""
    print("\n🔧 Installing system dependencies...")

    system = detect_system()
    if not system:
        print("⚠️  Could not detect system type. Skipping system package installation.")
        print("   You may need to manually install: python3-dev, build tools, and graphics libraries")
        return True

    packages = SYSTEM_PACKAGES.get(system, [])
    if not packages:
        print(f"⚠️  No system packages defined for {system}")
        return True

    if system == "ubuntu":
        # Update package list first
        print("   Updating package list...")
        result = run_command("sudo apt update", check=False)
        if result is None:
            print("⚠️  Could not update package list. Continuing anyway...")

        # Install packages
        cmd = f"sudo apt install -y {' '.join(packages)}"
    elif system == "centos":
        cmd = f"sudo yum install -y {' '.join(packages)}"
    else:
        print(f"⚠️  Unknown system: {system}")
        return True

    print(f"   Installing: {' '.join(packages)}")
    result = run_command(cmd, check=False)

    if result and result.returncode == 0:
        print("✅ System dependencies installed successfully")
        return True
    else:
        print("⚠️  Some system packages may not have installed correctly.")
        print("   This might affect matplotlib's ability to save images.")
        return True  # Continue anyway


def check_pip():
    """Ensure pip is available and up to date"""
    print("\n🔍 Checking pip...")

    # Check if pip is available
    result = run_command("python -m pip --version", check=False)
    if result is None or result.returncode != 0:
        print("❌ pip is not available. Please install pip first.")
        return False

    print("✅ pip is available")

    # Upgrade pip
    print("   Upgrading pip...")
    result = run_command("python -m pip install --upgrade pip", check=False)
    if result and result.returncode == 0:
        print("✅ pip upgraded successfully")
    else:
        print("⚠️  Could not upgrade pip, but continuing...")

    return True


def install_python_packages(upgrade=False, user=False):
    """Install required Python packages"""
    print(f"\n📦 Installing Python packages...")

    # Build pip command
    cmd_parts = ["python", "-m", "pip", "install"]

    if upgrade:
        cmd_parts.append("--upgrade")
        print("   (Upgrading to latest versions)")

    if user:
        cmd_parts.append("--user")
        print("   (Installing for current user only)")

    # Install packages one by one for better error handling
    failed_packages = []

    for package in REQUIRED_PACKAGES:
        package_name = package.split(">=")[0]  # Get package name without version
        print(f"   Installing {package_name}...")

        cmd = " ".join(cmd_parts + [package])
        result = run_command(cmd, check=False)

        if result and result.returncode == 0:
            print(f"   ✅ {package_name} installed successfully")
        else:
            print(f"   ❌ Failed to install {package_name}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"     - {pkg}")
        print("\n   You may need to install these manually or check for system-specific issues.")
        return False

    print(f"\n✅ All {len(REQUIRED_PACKAGES)} Python packages installed successfully!")
    return True


def verify_installation():
    """Verify that all packages can be imported"""
    print("\n🧪 Verifying installation...")

    test_imports = [
        ("pandas", "pd"),
        ("openpyxl", None),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("numpy", "np"),
    ]

    failed_imports = []

    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module} - {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n❌ {len(failed_imports)} packages failed to import:")
        for pkg in failed_imports:
            print(f"     - {pkg}")
        return False

    print("\n✅ All packages imported successfully!")
    return True


def create_test_script():
    """Create a simple test script to verify everything works"""
    test_script = """#!/usr/bin/env python3
'''
Quick test script to verify dependencies are working
'''

import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test_basic_functionality():
    print("🧪 Testing basic functionality...")

    # Test pandas
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"   ✅ pandas: Created DataFrame with shape {df.shape}")

    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    print(f"   ✅ numpy: Created array with {len(arr)} elements")

    # Test matplotlib (without showing plot)
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title("Test Plot")
    plt.close()
    print("   ✅ matplotlib: Created and closed test plot")

    # Test seaborn
    sns.set_style("whitegrid")
    print("   ✅ seaborn: Set plot style")

    print("\\n🎉 All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
"""

    test_file = Path("test_dependencies.py")
    with open(test_file, "w") as f:
        f.write(test_script)

    print(f"\n📝 Created test script: {test_file}")
    print("   Run 'python test_dependencies.py' to test basic functionality")


def print_usage_info():
    """Print information about using the installed packages"""
    print("\n" + "=" * 80)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 80)

    print("\n📋 INSTALLED PACKAGES:")
    for package in REQUIRED_PACKAGES:
        print(f"   ✅ {package}")

    print("\n🚀 READY TO USE:")
    print("   📊 excel_processor_combined.py - Process CSV/Excel files with opcodes")
    print("   📈 create_integrated_visualization.py - Create operations analysis charts")

    print("\n💡 USAGE EXAMPLES:")
    print("   # Process CSV files and create Excel with formatting")
    print("   python excel_processor_combined.py data.csv")
    print("   python excel_processor_combined.py /path/to/csv/folder/")
    print("")
    print("   # Create visualization from processed Excel file")
    print("   python create_integrated_visualization.py processed_data.xlsx")

    print("\n🔧 TROUBLESHOOTING:")
    print("   • If matplotlib plots don't display: install 'python3-tk' (Ubuntu) or 'tkinter' package")
    print("   • If Excel files won't open: ensure openpyxl version is compatible")
    print("   • For permission issues: try running with --user flag")

    print("\n📁 FILES CREATED:")
    print("   • test_dependencies.py - Run this to verify installation")


def main():
    """Main installation function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Install dependencies for Excel processor and visualization scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--upgrade", action="store_true", help="Upgrade packages to latest versions")
    parser.add_argument("--user", action="store_true", help="Install packages for current user only")
    parser.add_argument("--skip-system", action="store_true", help="Skip system package installation")

    args = parser.parse_args()

    print("🚀 Starting Dependencies Installation")
    print("=" * 80)
    print("Installing packages for:")
    print("  • excel_processor_combined.py")
    print("  • create_integrated_visualization.py")
    print("=" * 80)

    # Step 1: Check Python version
    if not check_python_version():
        return 1

    # Step 2: Install system dependencies
    if not args.skip_system:
        if not install_system_dependencies():
            print("⚠️  System dependency installation had issues, but continuing...")
    else:
        print("\n⏭️  Skipping system package installation")

    # Step 3: Check pip
    if not check_pip():
        return 1

    # Step 4: Install Python packages
    if not install_python_packages(upgrade=args.upgrade, user=args.user):
        print("\n❌ Some Python packages failed to install.")
        print("   Try running with --user flag or check your Python environment.")
        return 1

    # Step 5: Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed.")
        print("   Some packages may not be working correctly.")
        return 1

    # Step 6: Create test script
    create_test_script()

    # Step 7: Print success info
    print_usage_info()

    return 0


if __name__ == "__main__":
    sys.exit(main())
