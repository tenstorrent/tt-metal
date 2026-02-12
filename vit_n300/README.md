# ViT N300 — ND Failure Investigation & Stress Tests

This folder contains scripts, tests, and documentation for investigating the **non-deterministic (ND) device hang** in the `vit-N300-func` CI test. The hang manifests as a fetch-queue timeout, caused by a Tensix matmul deadlock in `bmm_large_block_zm_fused_bias_activation`.

## Folder Structure

```
vit_n300/
├── README.md                    # This file
├── ND_failure.log               # CI failure log with triage data
├── scripts/                     # Shell scripts
│   ├── run_vit_n300.sh          # Run the original ViT N300 test
│   ├── stress_test_vit_n300.sh  # Stress test: repeat original test
│   ├── stress_test_copy_stress.sh  # Stress test: copy/stall amplification
│   └── stress_test_matmul.sh    # Stress test: matmul deadlock (RECOMMENDED)
├── tests/                       # Python test files
│   ├── test_matmul_deadlock_stress.py  # Matmul deadlock reproducer (RECOMMENDED)
│   └── test_vit_2cq_copy_stress.py    # Copy-path stall amplifier
├── explanations/                # Analysis documentation
│   ├── 01_layer1_vit_n300_test_overview.md
│   ├── 02_layer2_host_to_device_copy_prefetcher_dispatch.md
│   ├── 03_layer3_semaphore_protocol_and_dispatch_hang.md
│   └── STRESS_STRATEGY.md
└── logs/                        # Stress test output logs (gitignored)
```

## Quick Start — Reproduce the ND Failure

The **matmul deadlock stress test** targets the actual root cause. Start here:

```bash
# Run the 2CQ variant (closest to CI failure scenario)
./vit_n300/scripts/stress_test_matmul.sh --2cq-only

# Run all variants (traced, direct, 2CQ)
./vit_n300/scripts/stress_test_matmul.sh

# Run just the traced variant (fastest per iteration)
./vit_n300/scripts/stress_test_matmul.sh --traced-only
```

Monitor output in real time:

```bash
tail -f vit_n300/logs/stress_matmul_*.log
```

### What the matmul stress test does

It spams the **exact block-sharded matmul configs** from the ViT model that deadlocked in CI:

| Matmul       | Grid | M×K×N          | Notes                    |
|-------------|------|----------------|--------------------------|
| QKV         | 8×8  | 1792×768×2304  | QKV projection           |
| self_output | 8×8  | 1792×768×768   | Attention output proj    |
| FF1         | 8×8  | 1792×768×3072  | Feedforward + GELU       |
| FF2         | 8×8  | 1792×3072×768  | Feedforward output       |

Each run: 5000 iterations × 4 matmuls = **20,000 block-sharded matmul ops**.

Three variants:
- **traced** — trace replay on CQ0 (fast, tests trace-related timing)
- **direct** — no trace, different memory allocation patterns
- **2cq** — trace on CQ0 + concurrent CQ1 copies (matches CI failure scenario exactly)

## Other Stress Tests

### Original ViT test stress loop

```bash
./vit_n300/scripts/stress_test_vit_n300.sh
```

Repeats the full ViT inference test for 30 minutes. Low reproduction rate (~0% in 20 runs).

### Copy/stall amplification

```bash
./vit_n300/scripts/stress_test_copy_stress.sh
```

Amplifies the CQ1 copy stall path (10 copies/iteration × 2000 iterations). See `explanations/STRESS_STRATEGY.md`.

## Root Cause Analysis

From CI triage (`ND_failure.log`):

1. **Primary**: Tensix cores deadlock in `bmm_large_block_zm_fused_bias_activation`
   - `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded` → `noc_semaphore_wait`
   - `reader_bmm_tile_layout_in1_receiver_writer_padding` → `noc_semaphore_wait`
   - `bmm_large_block_zm_fused_bias_activation` (trisc0) → `cb_wait_front`
   - `bmm_large_block_zm_fused_bias_activation` (trisc1) → `matmul_block` (running)
   - `bmm_large_block_zm_fused_bias_activation` (trisc2) → `pack_main`

2. **Secondary**: CQ0 dispatch stuck in `process_go_signal_mcast_cmd` (waiting for deadlocked workers)

3. **Symptom**: CQ1 prefetcher stuck in `process_stall` → host times out on fetch queue

See `explanations/` for layered analysis.

## Environment Variables

| Variable        | Default      | Description                                    |
|----------------|--------------|------------------------------------------------|
| `TT_METAL_HOME`| Auto-detected| Path to tt-metal repo root                     |
| `ARCH_NAME`    | `wormhole_b0`| Target architecture (N300 uses Wormhole)       |
| `LOGURU_LEVEL` | `INFO`       | Logging verbosity                              |
