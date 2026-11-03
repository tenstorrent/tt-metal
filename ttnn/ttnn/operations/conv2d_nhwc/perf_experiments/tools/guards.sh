#!/bin/bash
# guards.sh : one profiled run of each guard-set probe on the real op; prints TRISC_1 max per case.
cd "$(dirname "$0")/../../../../../.." || exit 1
T=tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_perf_probe.py
P=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/prof.sh
for c in layer3_ds layer3_3x3 layer4_3x3; do
  echo "[$c]"; R9_CASE=$c bash $P $T test_perf_probe_resnet_block 2>&1 | grep -E "==|TRISC_1" | tail -2
done
for k in test_perf_probe_block_sdxl_l0_ds_bf16_bf8b test_guard_block_bf16_nobias test_perf_probe_block_28x28x256 test_guard_width_strided test_guard_interleaved_bf16_relu test_perf_probe_resnet_layer2_ds test_perf_probe_resnet_layer1_3x3; do
  bash $P $T $k 2>&1 | grep -E "==|TRISC_1" | tail -3
done
