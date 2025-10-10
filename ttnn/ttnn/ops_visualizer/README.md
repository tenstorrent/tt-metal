# TT-Metal Operations Visualizer

A toolkit for profiling, analyzing, and visualizing TT-Metal model performance data through three levels of processing.

## 🚀 Quick Start

```bash
# 1. Setup Environment
git clone -b sdawle/model_ops_visualizer https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh -p  # Build with profiler flag
./create_env.sh
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Optional for N300
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# 2. Install Dependencies
cd ttnn/ttnn/ops_visualizer/
pip install -r requirements.txt --user

# 3. Generate profiling data (Level 1)
cd ../ops_profiler/
./generate_model_perf_sheets.sh

# 4. Create combined Excel (Level 2)
cd ../ops_visualizer/
python create_combined_excel.py $TT_METAL_HOME/generated/profiler/reports

# 5. Generate visualizations (Level 3)
python model_ops_visualizer.py reports_combined_complete.xlsx
```

## 🔄 Three-Level Process

### Level 1: Profile Generation 🔬
**Script**: `generate_model_perf_sheets.sh`
- Runs pytest with tracy profiling for 20+ models
- Generates timestamped CSV files in `$TT_METAL_HOME/generated/profiler/reports/`
- Handles failures gracefully, continues with remaining models
- Can be safely interrupted with Ctrl+C

### Level 2: Excel Processing 📊
**Script**: `create_combined_excel.py`
- Combines CSV files into formatted Excel workbook
- Adds data bars, FPS calculations, and performance rankings
- Filters out system files automatically
- Handles different CSV formats gracefully

### Level 3: Visualization 📈
**Script**: `model_ops_visualizer.py`
- Creates professional charts and graphs
- Generates `model_performance_analysis.png`
- Includes performance comparisons, operation distributions, and summary tables

## 📁 Output Structure

```
$TT_METAL_HOME/generated/profiler/reports/
├── UNET_256x256_oct10_1430/
│   ├── UNET_256x256.csv
│   └── ... (tracy files)
├── VGG19_256x256_oct10_1432/
│   ├── VGG19_256x256.csv
│   └── ... (tracy files)
└── reports_combined_complete.xlsx  # Level 2 output
└── model_performance_analysis.png  # Level 3 output
```

## 🔧 Advanced Usage

### Individual Processing
```bash
# Process single CSV file
python create_combined_excel.py path/to/file.csv

# Custom output directory
python create_combined_excel.py input_path output_dir

# Visualization with custom output
python model_ops_visualizer.py input.xlsx output_dir
```

### Supported Models (20+ total)
UNET, VGG19, ViT, Mobilenet, Sentence-BERT, ResNet, YOLOv4-v12 variants, Segformer, Ultrafast Lane Detection, and experimental models (Swin, VoVNet, Swin_v2).

## 🔧 Troubleshooting

**No CSV files found**: Check `$TT_METAL_HOME` is set and reports directory exists
**Import errors**: Run `pip install -r requirements.txt --user --upgrade`
**Permission errors**: Use `--user` flag or create virtual environment

## 📦 Requirements

**Python Packages**: pandas, openpyxl, matplotlib, seaborn, numpy, xlsxwriter, Pillow
**System**: Ubuntu 22.04, Python 3.7+, 4GB+ RAM, 1GB+ storage

---

**Happy Visualizing! 🚀📊**
