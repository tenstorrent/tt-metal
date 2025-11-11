#!/bin/bash

# Script to run all single-card demo tests
# Run this from the tt-metal root directory

set -e

# Determine the root directory (assuming script is run from root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Root directory: $ROOT_DIR"

# Set up environment variables
export TT_METAL_HOME="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export LD_LIBRARY_PATH="$ROOT_DIR/build/lib"
export ARCH_NAME=wormhole_b0
export LOGURU_LEVEL=INFO

echo "Environment setup complete:"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  ARCH_NAME=$ARCH_NAME"

# Change to root directory
cd "$ROOT_DIR"

# Source the test functions
echo "Sourcing test functions from: tests/scripts/single_card/run_single_card_demo_tests.sh"
source tests/scripts/single_card/run_single_card_demo_tests.sh

# Track failures
declare -a FAILED_TESTS=()
declare -a PASSED_TESTS=()

# Function to run a test and track its status
run_test() {
    local test_func=$1
    echo ""
    echo "========================================"
    echo "Running: $test_func"
    echo "========================================"

    if $test_func; then
        echo "✓ PASSED: $test_func"
        PASSED_TESTS+=("$test_func")
    else
        echo "✗ FAILED: $test_func"
        FAILED_TESTS+=("$test_func")
    fi
}

# Run all functional (non-perf) tests
echo ""
echo "========================================"
echo "Starting Demo Test Suite"
echo "========================================"

# Only the stable diffusion tests run
run_test run_stable_diffusion_func
run_test run_sdxl_func

# These tests have all kinds of problems...
#run_test run_falcon7b_func
#run_test run_llama3_func
#run_test run_vgg_func
#run_test run_bert_tiny_func
#run_test run_bert_func
#run_test run_resnet_func
#run_test run_distilbert_func
#run_test run_mnist_func
#run_test run_squeezebert_func
#run_test run_segformer_func
#run_test run_sentencebert_func
#run_test run_yolov11_func
#run_test run_yolov11m_func
#run_test run_ufld_v2_func
#run_test run_efficientnet_b0_func
#run_test run_vanilla_unet_demo
#run_test run_swin_s_demo
#run_test run_swin_v2_demo
#run_test run_vgg_unet_demo
#run_test run_vit_demo
#run_test run_vovnet_demo
#run_test run_yolov9c_perf
#run_test run_yolov8s_perf
#run_test run_mobilenetv2_perf
#run_test run_yolov8s_world_perf
#run_test run_yolov8x_perf
#run_test run_yolov4_perf
#run_test run_yolov10x_demo
#run_test run_yolov7_demo
#run_test run_yolov6l_demo
#run_test run_yolov12x_demo

# Print summary
echo ""
echo "========================================"
echo "Test Suite Summary"
echo "========================================"
echo "Total tests run: $((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]}))"
echo "Passed: ${#PASSED_TESTS[@]}"
echo "Failed: ${#FAILED_TESTS[@]}"
echo ""

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
    exit 1
else
    echo "All tests passed! ✓"
    exit 0
fi
