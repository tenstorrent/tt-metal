#!/bin/bash

# Dependencies Installation Script (Bash Version)
# ===============================================
# This script installs all required dependencies for:
# 1. excel_processor_combined.py - Excel/CSV processing
# 2. create_integrated_visualization.py - Visualization
#
# Usage:
#   ./install_dependencies.sh [--user] [--upgrade]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}🔍 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse command line arguments
USER_INSTALL=""
UPGRADE_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER_INSTALL="--user"
            shift
            ;;
        --upgrade)
            UPGRADE_FLAG="--upgrade"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--user] [--upgrade]"
            echo "  --user    Install packages for current user only"
            echo "  --upgrade Upgrade packages to latest versions"
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Dependencies Installation"
echo "======================================"
echo "Installing packages for:"
echo "  • excel_processor_combined.py"
echo "  • create_integrated_visualization.py"
echo "======================================"

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    print_error "Python $PYTHON_VERSION detected. Python 3.8+ is required."
    exit 1
fi

print_success "Python $PYTHON_VERSION - Compatible"

# Detect OS and install system dependencies
print_status "Installing system dependencies..."

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
else
    print_warning "Cannot detect OS. Skipping system package installation."
    OS="unknown"
fi

case $OS in
    ubuntu|debian)
        print_status "Detected Ubuntu/Debian system"
        if command -v sudo >/dev/null 2>&1; then
            sudo apt update
            sudo apt install -y python3-dev python3-pip python3-venv build-essential \
                               libfreetype6-dev libpng-dev pkg-config python3-tk
            print_success "System dependencies installed"
        else
            print_warning "sudo not available. Please install system packages manually:"
            echo "  apt install python3-dev python3-pip python3-venv build-essential libfreetype6-dev libpng-dev pkg-config python3-tk"
        fi
        ;;
    centos|rhel|fedora)
        print_status "Detected CentOS/RHEL/Fedora system"
        if command -v sudo >/dev/null 2>&1; then
            if command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y python3-devel python3-pip gcc gcc-c++ freetype-devel libpng-devel pkgconfig python3-tkinter
            else
                sudo yum install -y python3-devel python3-pip gcc gcc-c++ freetype-devel libpng-devel pkgconfig tkinter
            fi
            print_success "System dependencies installed"
        else
            print_warning "sudo not available. Please install system packages manually."
        fi
        ;;
    *)
        print_warning "Unknown OS: $OS. Skipping system package installation."
        print_warning "You may need to manually install: python3-dev, build tools, and graphics libraries"
        ;;
esac

# Check and upgrade pip
print_status "Checking pip..."
if ! python3 -m pip --version >/dev/null 2>&1; then
    print_error "pip is not available. Please install pip first."
    exit 1
fi

print_success "pip is available"
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip $USER_INSTALL

# Install Python packages
print_status "Installing Python packages..."

if [[ -n "$USER_INSTALL" ]]; then
    print_status "(Installing for current user only)"
fi

if [[ -n "$UPGRADE_FLAG" ]]; then
    print_status "(Upgrading to latest versions)"
fi

# Check if requirements.txt exists, otherwise install packages individually
if [[ -f "requirements.txt" ]]; then
    print_status "Installing from requirements.txt..."
    python3 -m pip install -r requirements.txt $USER_INSTALL $UPGRADE_FLAG
else
    print_status "Installing packages individually..."

    # Core packages
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "pandas>=1.5.0"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "openpyxl>=3.0.0"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "xlsxwriter>=3.0.0"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "matplotlib>=3.5.0"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "seaborn>=0.11.0"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "numpy>=1.21.0"

    # Optional packages
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "xlrd>=2.0.0" || print_warning "xlrd installation failed (optional)"
    python3 -m pip install $USER_INSTALL $UPGRADE_FLAG "Pillow>=8.0.0" || print_warning "Pillow installation failed (optional)"
fi

print_success "Python packages installed"

# Verify installation
print_status "Verifying installation..."

python3 -c "
import sys
failed = []

try:
    import pandas as pd
    print('   ✅ pandas')
except ImportError as e:
    print('   ❌ pandas -', e)
    failed.append('pandas')

try:
    import openpyxl
    print('   ✅ openpyxl')
except ImportError as e:
    print('   ❌ openpyxl -', e)
    failed.append('openpyxl')

try:
    import matplotlib.pyplot as plt
    print('   ✅ matplotlib')
except ImportError as e:
    print('   ❌ matplotlib -', e)
    failed.append('matplotlib')

try:
    import seaborn as sns
    print('   ✅ seaborn')
except ImportError as e:
    print('   ❌ seaborn -', e)
    failed.append('seaborn')

try:
    import numpy as np
    print('   ✅ numpy')
except ImportError as e:
    print('   ❌ numpy -', e)
    failed.append('numpy')

if failed:
    print(f'\\n❌ {len(failed)} packages failed to import')
    sys.exit(1)
else:
    print('\\n✅ All packages imported successfully!')
"

if [[ $? -ne 0 ]]; then
    print_error "Installation verification failed"
    exit 1
fi

# Create test script
print_status "Creating test script..."
cat > test_dependencies.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script to verify dependencies are working
"""

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

    print("\n🎉 All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
EOF

chmod +x test_dependencies.py
print_success "Created test script: test_dependencies.py"

# Print completion message
echo ""
echo "======================================"
echo "🎉 INSTALLATION COMPLETE!"
echo "======================================"
echo ""
echo "📋 INSTALLED PACKAGES:"
echo "   ✅ pandas (DataFrame operations)"
echo "   ✅ openpyxl (Excel manipulation)"
echo "   ✅ matplotlib (Plotting)"
echo "   ✅ seaborn (Statistical plots)"
echo "   ✅ numpy (Numerical operations)"
echo "   ✅ xlsxwriter (Excel writing)"
echo ""
echo "🚀 READY TO USE:"
echo "   📊 excel_processor_combined.py - Process CSV/Excel files"
echo "   📈 create_integrated_visualization.py - Create visualizations"
echo ""
echo "💡 USAGE EXAMPLES:"
echo "   python3 excel_processor_combined.py data.csv"
echo "   python3 create_integrated_visualization.py processed_data.xlsx"
echo ""
echo "🧪 TEST INSTALLATION:"
echo "   python3 test_dependencies.py"
echo ""
print_success "Installation completed successfully!"
