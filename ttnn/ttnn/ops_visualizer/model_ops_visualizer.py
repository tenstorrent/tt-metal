#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import argparse
import sys
from pathlib import Path
import matplotlib.colors as mcolors


def clean_model_name(model_name):
    """Remove dates and clean up model names"""
    cleaned = re.sub(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", "", model_name)
    cleaned = re.sub(r"_\d{4}_\d{2}_\d{2}", "", cleaned)
    cleaned = cleaned.replace("_2025_10_04_", "_")
    cleaned = cleaned.replace("_2025_10_04", "")

    # Handle specific model name cleanups
    if "ViT" in cleaned:
        cleaned = "ViT_base_224x224"
    elif "Yolov12x" in cleaned:
        cleaned = "Yolov12x_640x640"
    elif "Yolov4_base_640x640" in cleaned:
        cleaned = "Yolov4_base_640x640"
    elif "YOLOV10X" in cleaned or "Yolov10" in cleaned:
        cleaned = "Yolov10_x_640x640"
    elif "SENTENCE_BERT" in cleaned:
        cleaned = "Sentence-BERT_base_seql384_b8"
    elif "VANILLA_UNET" in cleaned:
        cleaned = "UNet_Vanilla_480x640"

    return cleaned


def get_operation_name(op_code):
    """Extract the actual operation name from OP CODE"""
    op_str = str(op_code).strip()
    if op_str and not op_str.startswith("nan"):
        return op_str
    else:
        return "Unknown"


def categorize_operation(op_name):
    """Categorize operations into groups"""
    op_lower = op_name.lower()

    # Core Operations - Split Conv and Matmul
    if op_name in ["Conv2d"]:
        return "Conv Operations"
    elif op_name in ["Matmul"]:
        return "Matmult Operations"
    elif op_name in [
        "Embeddings",
        "NLPConcatHeads",
        "NlpCreateHeadsSegformer",
        "SplitFusedQKVAndSplitHeads",
        "ConcatenateHeads",
        "CreateQKVHeadsDeviceOperation",
    ]:
        return "NLP Operations"
    # Split Softmax from Normalization
    elif op_name in ["Softmax", "SoftmaxDeviceOperation"]:
        return "Softmax"
    elif op_name in ["LayerNorm", "BatchNormOperation"]:
        return "Normalization Operations"
    # Split Unary from Binary
    elif op_name in ["Unary"]:
        return "Unary Operations"
    elif op_name in ["Binary", "BinaryNg", "BinaryDeviceOperation", "BinaryNgDeviceOperation"]:
        return "Binary Operations"
    elif op_name in ["Pool2D", "UpSample"]:
        return "Pooling & Sampling Operations"
    elif op_name in [
        "Move",
        "Copy",
        "Reshard",
        "ReshardDeviceOperation",
        "I2S",
        "S2I",
        "ShardedToInterleavedDeviceOperation",
        "InterleavedToShardedDeviceOperation",
    ]:
        return "Data Movement & Memory Operations"
    # Move Pad to Tensor Manipulation
    elif op_name in ["Concat", "ConcatDeviceOperation", "Slice", "Reshape", "Transpose", "Permute", "Repeat", "Pad"]:
        return "Tensor Manipulation Operations"
    elif op_name in ["Tilize", "TilizeWithValPadding", "Untilize", "UntilizeWithUnpadding", "FillPad", "Fold"]:
        return "Data Processing Operations"
    elif op_name in ["Halo", "Reduce", "Typecast", "CloneOperation"]:
        return "Utility Operations"
    else:
        return "Utility Operations"  # Fallback for any unlisted operations


def sort_models_logically(model_data):
    """Sort models in logical order: Transformers first, then YOLO models (v12 to v4), then all other models"""

    def get_model_sort_key(model_name):
        name_lower = model_name.lower()

        # Transformers first
        if any(x in name_lower for x in ["vit", "swin", "bert", "segformer"]):
            return (0, 0, model_name)

        # YOLO models (v12 to v4 - reverse order)
        elif "yolov12" in name_lower:
            return (1, 12, model_name)
        elif "yolov11" in name_lower:
            return (1, 11, model_name)
        elif "yolov10" in name_lower or "yolov1" in name_lower:
            return (1, 10, model_name)
        elif "yolov9" in name_lower:
            return (1, 9, model_name)
        elif "yolov8" in name_lower:
            return (1, 8, model_name)
        elif "yolov7" in name_lower:
            return (1, 7, model_name)
        elif "yolov6" in name_lower:
            return (1, 6, model_name)
        elif "yolov5" in name_lower:
            return (1, 5, model_name)
        elif "yolov4" in name_lower:
            return (1, 4, model_name)

        # All other models
        else:
            return (2, 0, model_name)

    return sorted(model_data, key=lambda x: get_model_sort_key(x["model_name"]))


def read_data_file(file_path):
    """Read Excel or CSV file and return data sheets"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        # For CSV files, read as a single sheet
        df = pd.read_csv(file_path)
        return {"data": df}
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        # For Excel files, read all sheets
        return pd.read_excel(file_path, sheet_name=None)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .csv, .xlsx, .xls")


def create_integrated_visualization(input_file, output_file=None):
    """Create integrated visualization with stacked bar chart, pie chart, and operation grouping table"""

    print("🚀 Starting Integrated Operations Analysis Visualization...")

    try:
        print(f"🔍 Processing {input_file}...")

        # Read the data file
        all_sheets = read_data_file(input_file)
        print(f"📊 Found {len(all_sheets)} sheets: {list(all_sheets.keys())}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Process each model
    model_data = []
    all_op_names = set()

    for sheet_name, df in all_sheets.items():
        if df.empty:
            continue

        # Clean model name
        clean_name = clean_model_name(sheet_name)

        # Clean the dataframe
        df_clean = df.dropna(subset=["OP CODE"])
        df_clean = df_clean[df_clean["DEVICE KERNEL DURATION [ns]"] > 0]

        if df_clean.empty:
            continue

        # Get operation names and categorize them
        df_clean["OP_NAME"] = df_clean["OP CODE"].apply(get_operation_name)
        df_clean["CATEGORY"] = df_clean["OP_NAME"].apply(categorize_operation)

        # Collect all unique operation names
        all_op_names.update(df_clean["OP_NAME"].unique())

        # Calculate total duration
        total_duration_ns = df_clean["DEVICE KERNEL DURATION [ns]"].sum()
        total_duration_ms = total_duration_ns / 1_000_000

        # Calculate FPS
        fps = 1000 / total_duration_ms if total_duration_ms > 0 else 0

        # Calculate category breakdown
        category_breakdown = {}
        for category in df_clean["CATEGORY"].unique():
            category_duration = df_clean[df_clean["CATEGORY"] == category]["DEVICE KERNEL DURATION [ns]"].sum()
            category_breakdown[category] = category_duration

        model_info = {
            "model_name": clean_name,
            "total_duration_ms": total_duration_ms,
            "fps": fps,
            "operation_count": len(df_clean),
            "category_breakdown": category_breakdown,
        }

        model_data.append(model_info)
        print(f"   ✅ {clean_name}: {len(df_clean)} ops, {total_duration_ms:.1f}ms, {fps:.2f} FPS")

    # Sort models logically
    model_data = sort_models_logically(model_data)

    # Invert the model order completely
    model_data = model_data[::-1]

    # Calculate category percentages across all models
    category_percentages = {}
    total_all_duration_ns = sum(model["total_duration_ms"] * 1_000_000 for model in model_data)

    for model in model_data:
        for category, duration_ns in model["category_breakdown"].items():
            if category not in category_percentages:
                category_percentages[category] = 0
            category_percentages[category] += duration_ns

    # Convert to percentages
    for category in category_percentages:
        category_percentages[category] = (
            (category_percentages[category] / total_all_duration_ns * 100) if total_all_duration_ns > 0 else 0
        )

    # Sort categories by percentage (descending order - highest to lowest)
    sorted_categories = sorted(category_percentages.items(), key=lambda x: x[1], reverse=True)
    all_categories = [category for category, _ in sorted_categories]  # Get all categories in sorted order

    # Define fresh colors for operation categories (12-category system)
    fresh_colors = {
        "Conv Operations": "#FF6347",  # Tomato
        "Matmult Operations": "#FF4500",  # OrangeRed
        "NLP Operations": "#9370DB",  # MediumPurple
        "Softmax": "#FFD700",  # Gold
        "Normalization Operations": "#FFA500",  # Orange
        "Unary Operations": "#3CB371",  # MediumSeaGreen
        "Binary Operations": "#32CD32",  # LimeGreen
        "Pooling & Sampling Operations": "#1E90FF",  # DodgerBlue
        "Data Movement & Memory Operations": "#BA55D3",  # MediumOrchid
        "Tensor Manipulation Operations": "#20B2AA",  # LightSeaGreen
        "Data Processing Operations": "#FF8C00",  # DarkOrange
        "Utility Operations": "#708090",  # SlateGray
    }

    print(f"\n📊 Creating integrated visualizations for {len(model_data)} models...")

    # Create figure with a grid for 3 subplots: bar chart (top, spans 2 cols), pie chart (bottom-left), table (bottom-right)
    fig = plt.figure(figsize=(40, 20))  # Increased width for longer horizontal graph
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.5], width_ratios=[1.5, 1.2])  # Increased width ratio for top chart
    ax1 = fig.add_subplot(gs[0, :])  # Top chart spans both columns
    ax2 = fig.add_subplot(gs[1, 0])  # Pie chart in bottom-left
    ax3 = fig.add_subplot(gs[1, 1])  # Table in bottom-right

    # --- Plot 1: Stacked Horizontal Bar Chart for Operation Categories ---
    # Create stacked data for plotting
    stacked_data = []
    for model_info in model_data:
        row_data = {}
        total_model_duration_ns = model_info["total_duration_ms"] * 1_000_000
        for category in sorted(all_categories):
            duration_ns = model_info["category_breakdown"].get(category, 0)
            percentage = (duration_ns / total_model_duration_ns * 100) if total_model_duration_ns > 0 else 0
            row_data[category] = percentage
        stacked_data.append(row_data)

    # Create DataFrame for stacked bar chart
    stacked_df = pd.DataFrame(stacked_data, index=[model["model_name"] for model in model_data])

    # Create the stacked horizontal bar chart
    colors = [fresh_colors.get(cat, "#CCCCCC") for cat in stacked_df.columns]
    stacked_df.plot(kind="barh", stacked=True, ax=ax1, color=colors, width=0.8)

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(stacked_df.iterrows()):
        cumulative_percentage = 0
        for category in stacked_df.columns:
            percentage = row[category]
            if percentage > 1.0:  # Only label segments larger than 1% for readability
                ax1.text(
                    cumulative_percentage + percentage / 2,
                    i,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            cumulative_percentage += percentage

        # Add Non Conv + Matmult percentage column
        non_conv_matmul_percentage = 100 - row.get("Conv Operations", 0) - row.get("Matmult Operations", 0)
        ax1.text(
            102,
            i,
            f"{non_conv_matmul_percentage:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.5),
        )

    # Add title for Non Conv + Matmult column
    ax1.text(
        102,
        len(stacked_df) + 0.3,
        "Non Conv +\nMatmult %",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1),
    )

    ax1.set_xlabel("Percentage of Total Device Time (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Models", fontsize=12, fontweight="bold")
    ax1.set_title("Operation Category Breakdown by Model", fontsize=16, fontweight="bold", pad=20)
    ax1.legend(title="Operation Categories", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax1.grid(False)  # Remove grid lines
    ax1.set_xlim(0, 100)  # Set cutoff at 100% since stacked bars sum to 100%

    # --- Plot 2: Pie Chart for Category Summary ---
    # Prepare data for pie chart
    pie_data = []
    pie_labels = []
    pie_colors = []

    for category, percentage in sorted_categories:
        if percentage > 0:  # Only include categories with data
            pie_data.append(percentage)
            pie_labels.append(category)
            pie_colors.append(fresh_colors.get(category, "#CCCCCC"))

    # Create donut chart with labels directly on segments
    wedges, texts, autotexts = ax2.pie(
        pie_data,
        labels=pie_labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=90,
        pctdistance=0.85,
        labeldistance=1.1,
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )  # Donut chart style

    # Style the pie chart
    for i, (autotext, label_text) in enumerate(zip(autotexts, pie_labels)):
        autotext.set_color("black")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

        # Custom positioning for specific percentage labels
        if "NLP Operations" in label_text:
            # Move NLP Operations percentage up (keep original left position)
            current_pos = autotext.get_position()
            autotext.set_position((current_pos[0] - 0.02, current_pos[1] + 0.16))
        elif "Normalization Operations" in label_text:
            # Move Normalization Operations percentage up (keep original left position)
            current_pos = autotext.get_position()
            autotext.set_position((current_pos[0] + 0.03, current_pos[1] + 0.10))
        elif "Unary Operations" in label_text:
            # Move Unary Operations percentage up and left (less than Normalization)
            current_pos = autotext.get_position()
            autotext.set_position((current_pos[0] + 0.02, current_pos[1] + 0.06))
        elif "Pooling & Sampling Operations" in label_text:
            # Move Pooling & Sampling Operations percentage up (less than Unary)
            current_pos = autotext.get_position()
            autotext.set_position((current_pos[0] + 0.02, current_pos[1] + 0.01))

    # Style the labels directly on the pie chart with custom positioning and colors
    for i, (text, color) in enumerate(zip(texts, pie_colors)):
        text.set_fontsize(8)
        text.set_fontweight("bold")
        text.set_color(color)  # Use the same color as the pie segment
        # Add background to text for better readability
        text.set_bbox(dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1.5))

        # Custom positioning for specific labels
        label_text = text.get_text()
        if "NLP Operations" in label_text:
            # Move NLP Operations up and further left
            current_pos = text.get_position()
            text.set_position((current_pos[0] - 0.15, current_pos[1] + 0.25))
        elif "Normalization Operations" in label_text:
            # Move Normalization Operations up and further left (less than NLP)
            current_pos = text.get_position()
            text.set_position((current_pos[0] - 0.1, current_pos[1] + 0.18))
        elif "Unary Operations" in label_text:
            # Move Unary Operations up and left (less than Normalization)
            current_pos = text.get_position()
            text.set_position((current_pos[0] - 0.02, current_pos[1] + 0.12))
        elif "Pooling & Sampling Operations" in label_text:
            # Move Pooling & Sampling Operations up (less than Unary)
            current_pos = text.get_position()
            text.set_position((current_pos[0], current_pos[1] + 0.05))

    # Add title in the center of the pie chart
    ax2.text(
        0.5,
        0.5,
        "Operation Category\nSummary - Total\nContribution Across\nAll Models",
        fontsize=14,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment="center",
        horizontalalignment="center",
    )

    # --- Plot 3: Operation Grouping Table ---
    ax3.axis("off")  # Turn off axis for table
    ax3.set_title("Operation Grouping", fontsize=16, fontweight="bold", pad=10)

    # Define operation groups with their OP codes based on 11-category system
    operation_groups = {
        "Conv Operations": ["Conv2d"],
        "Matmult Operations": ["Matmul"],
        "NLP Operations": [
            "Embeddings",
            "NLPConcatHeads",
            "NlpCreateHeadsSegformer",
            "SplitFusedQKVAndSplitHeads",
            "ConcatenateHeads",
            "CreateQKVHeadsDeviceOperation",
        ],
        "Softmax": ["Softmax", "SoftmaxDeviceOperation"],
        "Normalization Operations": ["LayerNorm", "BatchNormOperation"],
        "Unary Operations": ["Unary"],
        "Binary Operations": ["Binary", "BinaryNg", "BinaryDeviceOperation", "BinaryNgDeviceOperation"],
        "Pooling & Sampling Operations": ["Pool2D", "UpSample"],
        "Data Movement & Memory Operations": [
            "Move",
            "Copy",
            "Reshard",
            "ReshardDeviceOperation",
            "I2S",
            "S2I",
            "ShardedToInterleavedDeviceOperation",
            "InterleavedToShardedDeviceOperation",
        ],
        "Tensor Manipulation Operations": [
            "Concat",
            "ConcatDeviceOperation",
            "Slice",
            "Reshape",
            "Transpose",
            "Permute",
            "Repeat",
            "Pad",
        ],
        "Data Processing Operations": [
            "Tilize",
            "TilizeWithValPadding",
            "Untilize",
            "UntilizeWithUnpadding",
            "FillPad",
            "Fold",
        ],
        "Utility Operations": ["Halo", "Reduce", "Typecast", "CloneOperation"],
    }

    table_data = []
    row_colors = []
    for group_name, op_codes in operation_groups.items():
        # Add group name row
        table_data.append([group_name, ""])
        row_colors.append(fresh_colors.get(group_name, "#CCCCCC"))
        # Add op codes under the group
        for op_code in op_codes:
            table_data.append(["", op_code])
            row_colors.append("#F8F9FA")  # Light background for op codes

    # Create table
    table = ax3.table(
        cellText=table_data, colLabels=["Operation Group", "OP Code"], cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
    )  # Fill the subplot area

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)  # Increased cell height to prevent text overlap

    # Style the table
    for i in range(len(table_data) + 1):  # +1 for header
        for j in range(2):
            cell = table[(i, j)]
            cell.set_edgecolor("none")  # Remove cell borders

    # Color the group name cells to span across both columns
    row_idx = 1  # Start after header row
    for group_name, op_codes in operation_groups.items():
        if group_name in fresh_colors:
            # Color both columns for the group name row
            for col in range(2):
                cell = table[(row_idx, col)]  # Group name cell
                cell.set_facecolor(fresh_colors[group_name])
                cell.set_text_props(weight="bold", color="white")
        row_idx += 1  # Move to next group name row
        row_idx += len(op_codes)  # Skip the OP code rows for this group

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()

    # Determine output filename
    if output_file is None:
        input_path = Path(input_file)
        output_file = f"model_performance_analysis.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🎉 Successfully created model performance visualization: {output_file}")
    print(f"📊 Processed {len(model_data)} models with {len(all_categories)} operation categories")


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Create integrated operations analysis visualization from Excel or CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_integrated_visualization.py data.xlsx
  python create_integrated_visualization.py data.csv -o output.png
  python create_integrated_visualization.py combined_models_analysis.xlsx
        """,
    )

    parser.add_argument("input_file", help="Path to input Excel (.xlsx, .xls) or CSV file")
    parser.add_argument("-o", "--output", help="Output PNG file path (default: auto-generated)")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        sys.exit(1)

    if input_path.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
        print(f"❌ Error: Unsupported file format '{input_path.suffix}'. Supported formats: .csv, .xlsx, .xls")
        sys.exit(1)

    # Create visualization
    create_integrated_visualization(args.input_file, args.output)


if __name__ == "__main__":
    main()
