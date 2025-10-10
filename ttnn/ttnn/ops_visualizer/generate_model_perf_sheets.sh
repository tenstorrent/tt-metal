#!/bin/bash

# Performance test script for all models
# This script runs pytest with tracy profiling for each model and continues even if individual tests fail
#
# IMPLEMENTED MODELS:
# ✓ UNET 256x256 (VGG-UNet)
# ✓ VGG19 256x256
# ✓ ViT base 224x224 (b=8)
# ✓ Mobilenet v2 224x224 (b=10)
# ✓ Sentence-BERT base (SeqL=384) (b=8)
# ✓ ResNet 50 224x224 (b=16)
# ✓ Yolov8 x 640x640
# ✓ Yolov8 s 640x640
# ✓ Yolov6 l 640x480
# ✓ Ultrafast Lane Detection v2 320x800
# ✓ Yolov7 base/l 640x640
# ✓ Yolov4 base 320x320
# ✓ Yolov4 base 640x640
# ✓ Yolov11 n 640x640
# ✓ Yolov5 x 640x640
# ✓ UNet Vanilla 480x640
# ✓ Yolov8-world s 640x640
# ✓ Yolov10 x 640x640
# ✓ Segformer b0 - decoder 512x512
# ✓ Yolov9 c - obj det 640x640
# ✓ Yolov9 c - segm 640x640
# ✓ Yolov12 x 640x640
# ✓ Swin s 512x512 (experimental)
# ✓ VoVNet 19b 224x224 (experimental)
# ✓ Swin_v2 s 512x512 (experimental)
#
# ALL 24 MODELS FOUND AND IMPLEMENTED!

set +e  # Don't exit on error - continue with next model if one fails

# Cleanup function to kill all child processes
cleanup() {
    echo ""
    echo -e "\033[1;33m⚠️  Script interrupted! Cleaning up...\033[0m"

    # Kill all child processes of this script
    local script_pid=$$
    echo "Killing all child processes of PID $script_pid..."

    # Kill all tracy processes
    pkill -f "tracy" 2>/dev/null && echo "✓ Killed tracy processes"

    # Kill all pytest processes
    pkill -f "pytest" 2>/dev/null && echo "✓ Killed pytest processes"

    # Kill all python processes that are children of this script
    pkill -P $script_pid 2>/dev/null && echo "✓ Killed child python processes"

    # Force kill any remaining child processes
    sleep 1
    pkill -9 -P $script_pid 2>/dev/null

    echo -e "\033[0;32m✓ Cleanup completed\033[0m"

    # Log the interruption
    if [ -n "$LOG_FILE" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "Script interrupted at $(date)" | tee -a "$LOG_FILE"
        echo "Cleanup completed" | tee -a "$LOG_FILE"
    fi

    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM
echo "Signal handlers set up - Press Ctrl+C to stop the script safely"

# Activate Python environment
echo "Activating Python environment..."
if [ -d "python_env" ]; then
    source python_env/bin/activate
    echo "✓ Python environment activated"
else
    echo "⚠️  Python environment not found, using system python3"
    # Create alias for python if it doesn't exist
    alias python=python3
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log file for results
LOG_FILE="model_perf_test_results_$(date +%Y%m%d_%H%M%S).log"
echo "Starting performance tests at $(date)" | tee "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Function to run a test with error handling
run_model_test() {
    local model_name="$1"
    local model_size="$2"
    local test_command="$3"

    echo -e "${YELLOW}Testing: $model_name $model_size${NC}" | tee -a "$LOG_FILE"
    echo "Command: $test_command" | tee -a "$LOG_FILE"

    # Set TTNN_CONFIG_OVERRIDES with model-specific report name
    export TTNN_CONFIG_OVERRIDES='{
      "enable_fast_runtime_mode": false,
      "enable_logging": true,
      "report_name": "'$model_name'_'$model_size'",
      "enable_graph_report": false,
      "enable_detailed_buffer_report": true,
      "enable_detailed_tensor_report": false,
      "enable_comparison_mode": false
    }'

    # Run the test
    local test_result=0
    echo "Starting test (Press Ctrl+C to stop)..." | tee -a "$LOG_FILE"

    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED: $model_name $model_size${NC}" | tee -a "$LOG_FILE"
        test_result=0
    else
        echo -e "${RED}✗ FAILED: $model_name $model_size${NC}" | tee -a "$LOG_FILE"
        test_result=1
    fi

    # Rename the generated reports folder and CSV file
    if [ -n "$TT_METAL_HOME" ] && [ -d "$TT_METAL_HOME/generated/profiler/reports" ]; then
        echo "Renaming reports folder and CSV file for $model_name $model_size..." | tee -a "$LOG_FILE"

        # Find the most recently created folder in reports directory
        local reports_dir="$TT_METAL_HOME/generated/profiler/reports"
        local latest_folder=$(find "$reports_dir" -maxdepth 1 -type d -not -path "$reports_dir" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

        if [ -n "$latest_folder" ] && [ -d "$latest_folder" ]; then
            # Create folder name with timestamp: model_name_model_size_date_time
            local current_date=$(date +%b%d_%H%M | tr '[:upper:]' '[:lower:]')  # e.g., oct10_0421
            local target_folder_name="${model_name}_${model_size}_${current_date}"
            local target_folder_path="$reports_dir/$target_folder_name"

            # Rename the folder if it's not already named correctly
            if [ "$(basename "$latest_folder")" != "$target_folder_name" ]; then
                # If target folder already exists, remove it first
                if [ -d "$target_folder_path" ]; then
                    echo "Removing existing folder: $target_folder_path" | tee -a "$LOG_FILE"
                    rm -rf "$target_folder_path"
                fi

                echo "Renaming folder: $(basename "$latest_folder") -> $target_folder_name" | tee -a "$LOG_FILE"
                mv "$latest_folder" "$target_folder_path"
                latest_folder="$target_folder_path"
            fi

            # Rename CSV file if it exists (handle multiple CSV patterns)
            local target_csv_name="${model_name}_${model_size}.csv"
            local target_csv_path="$latest_folder/$target_csv_name"

            echo "Looking for CSV files in: $latest_folder" | tee -a "$LOG_FILE"
            echo "Available files:" | tee -a "$LOG_FILE"
            ls -la "$latest_folder" | tee -a "$LOG_FILE"

            local csv_found=false

            # Look for ops_profiling.csv first
            local ops_csv_file="$latest_folder/ops_profiling.csv"
            if [ -f "$ops_csv_file" ] && [ "$(basename "$ops_csv_file")" != "$target_csv_name" ]; then
                echo "Renaming CSV file: ops_profiling.csv -> $target_csv_name" | tee -a "$LOG_FILE"
                mv "$ops_csv_file" "$target_csv_path"
                csv_found=true
            else
                # Look for ops_perf_results_*.csv pattern
                local ops_perf_file=$(find "$latest_folder" -name "ops_perf_results_*.csv" -type f | head -1)
                if [ -n "$ops_perf_file" ] && [ -f "$ops_perf_file" ] && [ "$(basename "$ops_perf_file")" != "$target_csv_name" ]; then
                    echo "Renaming CSV file: $(basename "$ops_perf_file") -> $target_csv_name" | tee -a "$LOG_FILE"
                    mv "$ops_perf_file" "$target_csv_path"
                    csv_found=true
                else
                    # Look for any other CSV files (excluding profile_log_device.csv)
                    local other_csv=$(find "$latest_folder" -name "*.csv" -not -name "profile_log_device.csv" -not -name "$target_csv_name" -type f | head -1)
                    if [ -n "$other_csv" ] && [ -f "$other_csv" ]; then
                        echo "Renaming CSV file: $(basename "$other_csv") -> $target_csv_name" | tee -a "$LOG_FILE"
                        mv "$other_csv" "$target_csv_path"
                        csv_found=true
                    fi
                fi
            fi

            if [ "$csv_found" = false ]; then
                echo "⚠️  No performance CSV file found for $model_name $model_size" | tee -a "$LOG_FILE"
                echo "   This might indicate the test was interrupted or failed to generate profiling data" | tee -a "$LOG_FILE"
            fi

            echo "✓ Reports organized for $model_name $model_size" | tee -a "$LOG_FILE"
        else
            echo "⚠️  No reports folder found for $model_name $model_size" | tee -a "$LOG_FILE"
        fi
    else
        echo "⚠️  TT_METAL_HOME not set or reports directory not found" | tee -a "$LOG_FILE"
    fi

    return $test_result
}

# Counter for results
PASSED=0
FAILED=0
TOTAL=0

echo "Starting model performance tests..." | tee -a "$LOG_FILE"

# UNET 256x256
TOTAL=$((TOTAL + 1))
if run_model_test "UNET" "256x256" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# VGG19 256x256
TOTAL=$((TOTAL + 1))
if run_model_test "VGG19" "256x256" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# ViT base 224x224 (b=8)
TOTAL=$((TOTAL + 1))
if run_model_test "ViT" "base_224x224_b8" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/vit/tests/pcc/test_ttnn_optimized_sharded_vit_wh.py::test_vit -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Mobilenet v2 224x224 (b=10)
TOTAL=$((TOTAL + 1))
if run_model_test "Mobilenet" "v2_224x224_b10" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/mobilenetv2/tests/pcc/test_mobilenetv2.py::test_mobilenetv2 -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Sentence-BERT base (SeqL=384) (b=8)
TOTAL=$((TOTAL + 1))
if run_model_test "Sentence-BERT" "base_seql384_b8" "python -m tracy -p -r -v -m pytest --disable-warnings $TT_METAL_HOME/models/demos/sentence_bert/tests/pcc/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# ResNet 50 224x224 (b=16)
TOTAL=$((TOTAL + 1))
if run_model_test "ResNet" "50_224x224_b16" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/wormhole/resnet50/tests/test_resnet50_functional.py -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov8 x 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov8" "x_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov8x/tests/pcc/test_yolov8x.py::test_yolov8x_640 -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov8 s 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov8" "s_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640 -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov6 l 640x480
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov6" "l_640x480" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov6l/tests/pcc/test_ttnn_yolov6l.py::test_yolov6l -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov7 base/l 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov7" "base_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov7/tests/pcc/test_ttnn_yolov7.py::test_yolov7 -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov4 base 320x320
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov4" "base_320x320" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0] -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov4 base 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov4" "base_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_true-0] -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov11 n 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov11" "n_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov11/tests/pcc/test_ttnn_yolov11.py::test_yolov11 -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov5 x 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov5" "x_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov5x/tests/pcc/test_ttnn_yolov5x.py::test_yolov5x -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# UNet Vanilla 480x640
TOTAL=$((TOTAL + 1))
if run_model_test "UNet" "Vanilla_480x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/vanilla_unet/tests/pcc/test_ttnn_unet.py::test_unet -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov8-world s 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov8-world" "s_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOMEmodels/demos/yolov8s_world/tests/pcc/test_ttnn_yolov8s_world.py::test_yolo_model -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov10 x 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov10" "x_640x640" "python -m tracy -p -r -v -m pytest --disable-warnings $TT_METAL_HOME/models/demos/yolov10x/tests/pcc/test_ttnn_yolov10x.py::test_yolov10x -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Segformer b0 - decoder 512x512
TOTAL=$((TOTAL + 1))
if run_model_test "Segformer" "b0_decoder_512x512" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py::test_segformer_for_semantic_segmentation -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Ultrafast Lane Detection v2 320x800
TOTAL=$((TOTAL + 1))
if run_model_test "Ultrafast_Lane_Detection" "v2_320x800" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov9 c - obj det 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov9" "c_objdet_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov9c/tests/pcc/test_ttnn_yolov9c.py::test_yolov9c -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov9 c - segm 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov9" "c_segm_640x640" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/demos/yolov9c/tests/pcc/test_ttnn_yolov9c.py::test_yolov9c -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Yolov12 x 640x640
TOTAL=$((TOTAL + 1))
if run_model_test "Yolov12" "x_640x640" "python -m tracy -p -r -v -m pytest --disable-warnings $TT_METAL_HOME/models/demos/yolov12x/tests/pcc/test_ttnn_yolov12x.py::test_yolov12x[pretrained_weight_true-0] -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo -e "${YELLOW}Starting experimental models...${NC}" | tee -a "$LOG_FILE"

# Swin s 512x512 (experimental)
TOTAL=$((TOTAL + 1))
if run_model_test "Swin" "s_512x512" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/pytest models/experimental/swin_s/tests/pcc/test_ttnn_swin_transformer.py::test_swin_s_transformer -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi


# VoVNet 19b 224x224 (experimental)
TOTAL=$((TOTAL + 1))
if run_model_test "VoVNet" "19b_224x224" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/experimental/vovnet/tests/pcc/test_tt_vovnet.py::test_vovnet_model_inference -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Swin_v2 s 512x512 (experimental)
TOTAL=$((TOTAL + 1))
if run_model_test "Swin_v2" "s_512x512" "python -m tracy -p -r -v -m pytest $TT_METAL_HOME/models/experimental/swin_v2/tests/pcc/test_ttnn_swin_v2_s.py::test_swin_s_transformer -v"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PERFORMANCE TEST SUMMARY" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total tests: $TOTAL" | tee -a "$LOG_FILE"
echo -e "${GREEN}Passed: $PASSED${NC}" | tee -a "$LOG_FILE"
echo -e "${RED}Failed: $FAILED${NC}" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Check the log for details.${NC}" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# Note: If script was interrupted, the cleanup() function will handle the exit
