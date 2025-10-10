# TT-Metal Operations Visualizer

A comprehensive toolkit for profiling, analyzing, and visualizing TT-Metal model performance data. This toolkit provides three levels of data generation and analysis, from raw profiling data to beautiful visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Three-Level Generation Process](#three-level-generation-process)
- [Detailed Usage](#detailed-usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

## 🎯 Overview

The TT-Metal Operations Visualizer consists of three main components that work together to provide comprehensive performance analysis:

1. **🔬 Profile Generation**: Run performance tests and generate profiling data
2. **📊 Excel Processing**: Combine and format CSV data into Excel workbooks
3. **📈 Visualization**: Create beautiful charts and graphs from the processed data

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- TT-Metal environment set up with `$TT_METAL_HOME` defined
- Bash shell (for profile generation)

### Install Dependencies

Choose one of three methods:

#### Method 1: Python Script (Recommended - Cross-platform)
```bash
cd /path/to/tt-metal/ttnn/ttnn/ops_visualizer/
python install_dependencies.py --user
```

#### Method 2: Bash Script (Linux/macOS only)
```bash
cd /path/to/tt-metal/ttnn/ttnn/ops_visualizer/
chmod +x install_dependencies.sh
./install_dependencies.sh --user
```

#### Method 3: Direct pip (Simple)
```bash
cd /path/to/tt-metal/ttnn/ttnn/ops_visualizer/
pip install -r requirements.txt --user
```

### Installation Options
- `--user`: Install for current user only (no admin required)
- `--upgrade`: Upgrade packages to latest versions

## ⚡ Quick Start

```bash
# 1. Build Metal and Setup Envoirment
git clone -b sdawle/model_ops_visualizer https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh -p # (Build with profiler flag on, to generate perf sheets)
./create_env.sh
source python_env/bin/activate
export TT_METAL_HOME=/path/to/your/tt-metal
export PYTHONPATH=/path/to/your/tt-metal
# BELOW TWO ENV VARIABLE ARE OPTIONAL FOR N300
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml;

# 2. Generate profiling data (Level 1)
cd $TT_METAL_HOME/ttnn/ttnn/ops_profiler/
./generate_model_perf_sheets.sh

# 3. Create combined Excel (Level 2)
cd $TT_METAL_HOME/ttnn/ttnn/ops_visualizer/
python create_combined_excel.py $TT_METAL_HOME/generated/profiler/reports

# 4. Generate visualizations (Level 3)
python model_ops_visualizer.py reports_combined_complete.xlsx
```

## 🔄 Three-Level Generation Process

### Level 1: Profile Generation 🔬
**Script**: `generate_model_perf_sheets.sh`
**Location**: `$TT_METAL_HOME/ttnn/ttnn/ops_profiler/`
**Purpose**: Run performance tests and generate raw profiling data

```bash
cd $TT_METAL_HOME/ttnn/ttnn/ops_profiler/
./generate_model_perf_sheets.sh
```

**What it does:**
- Runs pytest with tracy profiling for all supported models
- Generates CSV files with operation performance data
- Creates timestamped folders in `$TT_METAL_HOME/generated/profiler/reports/`
- Handles test failures gracefully and continues with remaining models
- Can be interrupted safely with Ctrl+C

**Output Structure:**
```
$TT_METAL_HOME/generated/profiler/reports/
├── UNET_256x256_oct10_1430/
│   ├── UNET_256x256.csv
│   ├── profile_log_device.csv
│   └── tracy_profile_log_host.tracy
├── VGG19_256x256_oct10_1432/
│   ├── VGG19_256x256.csv
│   ├── profile_log_device.csv
│   └── tracy_profile_log_host.tracy
└── ... (more models)
```

**Supported Models (27 total):**
- UNET, VGG19, ViT, Mobilenet, Sentence-BERT, ResNet
- YOLOv4/v5/v6/v7/v8/v9/v10/v11/v12 variants
- Segformer, Ultrafast Lane Detection
- Experimental: Swin, VoVNet, Swin_v2

### Level 2: Excel Processing 📊
**Script**: `create_combined_excel.py`
**Location**: `$TT_METAL_HOME/ttnn/ttnn/ops_visualizer/`
**Purpose**: Combine CSV files into formatted Excel workbooks

```bash
cd $TT_METAL_HOME/ttnn/ttnn/ops_visualizer/
python create_combined_excel.py $TT_METAL_HOME/generated/profiler/reports
```

**What it does:**
- Recursively searches for CSV files in the reports directory
- Processes and cleans operation data (opcodes, durations, etc.)
- Creates a combined Excel file with separate sheets for each model
- Adds formatted columns with data bars for visual comparison
- Calculates FPS (Frames Per Second) for each model
- Filters out system files like `profile_log_device.csv`

**Features:**
- ✅ **Smart Column Detection**: Handles different CSV formats automatically
- ✅ **Data Cleaning**: Simplifies operation names (e.g., `DeviceOperation` → removed)
- ✅ **Visual Formatting**: Blue gradient data bars for duration comparison
- ✅ **Performance Metrics**: Automatic FPS calculations
- ✅ **Error Handling**: Skips invalid files gracefully

**Output**: `reports_combined_complete.xlsx`

### Level 3: Visualization 📈
**Script**: `model_ops_visualizer.py`
**Location**: `$TT_METAL_HOME/ttnn/ttnn/ops_visualizer/`
**Purpose**: Generate beautiful visualizations from Excel data

```bash
cd $TT_METAL_HOME/ttnn/ttnn/ops_visualizer/
python model_ops_visualizer.py reports_combined_complete.xlsx
```

**What it generates:**
- 📊 **Performance comparison charts** (FPS rankings)
- 🥧 **Operation type distribution** (pie charts)
- 📈 **Duration analysis** (stacked bar charts)
- 📋 **Summary tables** with key statistics
- 🎨 **Professional styling** with consistent color schemes

**Output**: `model_performance_analysis.png`

## 📖 Detailed Usage

### Advanced Options

#### Profile Generation Options
```bash
# Run specific models only (edit the script)
./generate_model_perf_sheets.sh

# Check logs for detailed results
tail -f model_perf_test_results_*.log
```

#### Excel Processing Options
```bash
# Process single CSV file
python create_combined_excel.py path/to/single_file.csv

# Process with custom output directory
python create_combined_excel.py input_path output_directory

# Process existing Excel file (add formatting only)
python create_combined_excel.py existing_file.xlsx
```

#### Visualization Options
```bash
# Basic usage
python model_ops_visualizer.py input_file.xlsx

# With custom output directory
python model_ops_visualizer.py input_file.xlsx output_directory

# Help and options
python model_ops_visualizer.py --help
```

### Environment Variables

Make sure these are set:
```bash
export TT_METAL_HOME=/path/to/your/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

## 📁 Output Files

### Level 1 Outputs
- **CSV files**: Raw performance data for each model
- **Tracy files**: Detailed profiling traces
- **Log files**: Test execution logs with pass/fail status

### Level 2 Outputs
- **Excel workbook**: `reports_combined_complete.xlsx`
  - Separate sheet for each model
  - Formatted duration columns with data bars
  - FPS calculations and performance rankings

### Level 3 Outputs
- **PNG visualization**: `model_performance_analysis.png`
  - High-resolution charts and graphs
  - Professional styling suitable for presentations
  - Comprehensive performance analysis

## 🔧 Troubleshooting

### Common Issues

#### "No CSV files found"
```bash
# Check if TT_METAL_HOME is set
echo $TT_METAL_HOME

# Verify reports directory exists
ls -la $TT_METAL_HOME/generated/profiler/reports/
```

#### "At least one sheet must be visible" Excel error
- This is now fixed in the latest version
- The script handles incompatible CSV files gracefully
- Check the console output for which files were skipped

#### Import errors
```bash
# Reinstall dependencies
python install_dependencies.py --upgrade --user

# Or manually install missing packages
pip install pandas openpyxl matplotlib seaborn --user
```

#### Permission errors
```bash
# Use --user flag for installations
python install_dependencies.py --user

# Or create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python install_dependencies.py
```

### Performance Tips

- **Large datasets**: The tools can handle hundreds of models efficiently
- **Memory usage**: Excel processing is memory-efficient with chunked operations
- **Parallel processing**: Level 1 can be run in parallel for different model subsets

## 📦 Requirements

### Python Packages
- `pandas>=1.5.0` - Data processing and Excel operations
- `openpyxl>=3.0.0` - Excel file manipulation and formatting
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical visualization
- `numpy>=1.21.0` - Numerical operations
- `xlsxwriter>=3.0.0` - Alternative Excel writer
- `Pillow>=8.0.0` - Image processing for exports

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.7 or higher
- **Memory**: 4GB+ recommended for large datasets
- **Storage**: 1GB+ for generated reports and visualizations

## 🎯 Example Workflow

Complete workflow from start to finish:

```bash
# 1. Navigate to TT-Metal directory
cd /path/to/tt-metal

# 2. Set environment
export TT_METAL_HOME=$(pwd)

# 3. Install dependencies (one-time setup)
cd ttnn/ttnn/ops_visualizer/
python install_dependencies.py --user

# 4. Generate profiling data (Level 1)
cd ../ops_profiler/
./generate_model_perf_sheets.sh

# 5. Wait for completion, then process data (Level 2)
cd ../ops_visualizer/
python create_combined_excel.py $TT_METAL_HOME/generated/profiler/reports

# 6. Generate visualizations (Level 3)
python model_ops_visualizer.py reports_combined_complete.xlsx

# 7. View results
ls -la *.xlsx *.png
```

## 🤝 Contributing

To contribute to this toolkit:
1. Follow the existing code style
2. Add tests for new features
3. Update this README for any new functionality
4. Ensure compatibility across all three levels

## 📄 License

This toolkit is part of the TT-Metal project. Please refer to the main TT-Metal license for usage terms.

---

**Happy profiling and visualizing! 🚀📊**
