# ViT N300 — ND Failure Investigation & Stress Tests

This folder contains scripts, tests, and documentation for investigating the **non-deterministic (ND) device hang** in the `vit-N300-func` CI test. The hang manifests as a fetch-queue timeout, caused by a Tensix matmul deadlock in `bmm_large_block_zm_fused_bias_activation`.

## Folder Structure

```
vit_n300/
├── README.md                    # This file
├── ND_failure.log               # CI failure log #1 (Feb 9) with triage data
├── ND_failure2.log              # CI failure log #2 (Jan 24) with triage data
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

# Run all variants (traced, direct, 2CQ, wide)
./vit_n300/scripts/stress_test_matmul.sh

# Run just the traced variant (fastest per iteration)
./vit_n300/scripts/stress_test_matmul.sh --traced-only

# Run wide matmuls (extra back-pressure from larger output tiles)
./vit_n300/scripts/stress_test_matmul.sh --wide-only
```

Monitor output in real time:

```bash
tail -f vit_n300/logs/stress_matmul_*.log
```

### With DPRINT CB monitoring

Enable device-side circular buffer monitoring to see fill levels on Tensix cores.
DPRINT output goes to a separate log file in `vit_n300/logs/`.

```bash
# Enable DPRINT (monitors 4 corner cores, brisc + ncrisc only)
./vit_n300/scripts/stress_test_matmul.sh --2cq-only --dprint

# Monitor DPRINT output
tail -f vit_n300/logs/dprint_*.log
```

Note: DPRINT adds some timing perturbation. Run without `--dprint` first to reproduce the deadlock at full speed, then enable it for diagnostics.

### What the matmul stress test does

It spams the **exact block-sharded matmul configs** from the ViT model that deadlocked in CI:

| Matmul       | Grid | M x K x N       | Notes                    |
|-------------|------|-----------------|--------------------------|
| QKV         | 8x8  | 1792 x 768 x 2304  | QKV projection           |
| self_output | 8x8  | 1792 x 768 x 768   | Attention output proj    |
| FF1         | 8x8  | 1792 x 768 x 3072  | Feedforward + GELU       |
| FF2         | 8x8  | 1792 x 3072 x 768  | Feedforward output       |

Each run: 10,000 iterations x 4 matmuls = **40,000 block-sharded matmul ops**.

Four test variants:
- **traced** — trace replay on CQ0 (fast, tests trace-related timing)
- **direct** — no trace, different memory allocation patterns
- **2cq** — trace on CQ0 + 5 concurrent CQ1 copies per iter (matches CI failure exactly)
- **wide** — adds extra-wide matmul configs (6144, 4096 output width) for more back-pressure

### Deadlock amplification

Two changes increase the probability of hitting the deadlock:

1. **Removed double-buffering**: `MCAST_INPUT_BUFFERING_DEPTH` set to 1 in `matmul_utilities.hpp` (was 2). This halves input CB sizes, eliminating slack that prevents back-pressure cycles. Requires rebuild.

2. **DPRINT instrumentation**: All 3 matmul kernels (compute, in0 reader, in1 reader/writer) have DPRINT statements at CB sync points. These compile to nothing unless `TT_METAL_DPRINT_CORES` is set, so there is zero overhead in normal runs.

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

Amplifies the CQ1 copy stall path (10 copies/iteration x 2000 iterations). See `explanations/STRESS_STRATEGY.md`.

## Root Cause Analysis

Confirmed by two independent CI failures (`ND_failure.log`, `ND_failure2.log`):

1. **Primary**: Tensix cores deadlock in `bmm_large_block_zm_fused_bias_activation`
   - Circular wait among 5 RISCs: brisc (output writer) blocked on CB, trisc2 (pack) blocked pushing to full CB, trisc0 (unpack) waiting for context, ncrisc (in0 reader) waiting for brisc
   - Can also involve cross-core multicast deadlock (multiple cores stuck on `noc_semaphore_wait`)
   - Occurs at different kernel stages: matmul loop (Failure 1) or bias+activation path (Failure 2)
   - Different cores each time (race condition, not hardware defect)

2. **Secondary**: CQ0 dispatch stuck in `process_go_signal_mcast_cmd` (waiting for deadlocked workers)

3. **Symptom**: CQ1 prefetcher stuck in `process_stall` -> host times out on fetch queue

See `explanations/` for layered analysis.

## Environment Variables

| Variable        | Default      | Description                                    |
|----------------|--------------|------------------------------------------------|
| `TT_METAL_HOME`| Auto-detected| Path to tt-metal repo root                     |
| `ARCH_NAME`    | `wormhole_b0`| Target architecture (N300 uses Wormhole)       |
| `LOGURU_LEVEL` | `INFO`       | Logging verbosity                              |
| `TT_METAL_DPRINT_CORES` | Not set | Set to enable DPRINT (e.g. `(0,0),(7,7)`) |
| `TT_METAL_DPRINT_RISCVS` | Not set | Which RISCs to monitor (e.g. `BR+NC`) |
| `TT_METAL_DPRINT_FILE` | stdout | File path for DPRINT output |
