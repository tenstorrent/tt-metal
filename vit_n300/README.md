# ViT N300 Local Demo Test

This folder provides scripts to run the **VIT single-card demo test** locally on N300 hardware. It reproduces the `vit-N300-func` test from the single-card-demo-tests CI pipeline (see `.github/workflows/single-card-demo-tests-impl.yaml`).

## Prerequisites

- **Hardware**: Single N300 card (Tenstorrent Wormhole architecture)
- **tt-metal repo**: Cloned and built (`./build_metal.sh` or equivalent)
- **Hugging Face**: Log in with `huggingface-cli login` or `export HF_TOKEN=<token>` (for `google/vit-base-patch16-224` model download)
- **Python env**: TT-Metal/TT-NN Python environment activated with required dependencies

## How to Run

From the tt-metal repo root:

```bash
./vit_n300/run_vit_n300.sh
```

Or from within the `vit_n300` folder:

```bash
cd vit_n300
./run_vit_n300.sh
```

Or from anywhere (with `TT_METAL_HOME` set):

```bash
TT_METAL_HOME=/path/to/tt-metal ./vit_n300/run_vit_n300.sh
```

## What the Test Does

The script runs:

```
pytest models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
```

This executes the ViT Base patch16-224 inference performance test with:

- Batch size: 8
- 2 command queues (2CQ) trace execution
- Warmup: 100 iterations
- Measurement: 1000 iterations
- Expected N300 throughput: ~1323 samples/sec (within 4% margin)

## Stress Test (Reproduce ND Failure)

To stress test for the non-deterministic fetch-queue timeout:

```bash
./vit_n300/stress_test_vit_n300.sh
```

Output is saved to `vit_n300/logs/stress_YYYYMMDD_HHMMSS.log` and also printed to the terminal. Monitor in another window with:

```bash
tail -f vit_n300/logs/stress_*.log
```

Runs the same test repeatedly for ~30 minutes, stopping on first failure.

### Copy-stress variant (amplify ND failure)

To stress the copy/stall path with ~20× more stall points:

```bash
./vit_n300/stress_test_copy_stress.sh
```

Uses `vit_n300/test_vit_2cq_copy_stress.py`: 10 copies per iteration × 2000 iterations ≈ 21,000 stall points per run (vs ~1,100 in original). See `vit_n300/STRESS_STRATEGY.md` for more strategies.

## Environment Variables

| Variable        | Default      | Description                                    |
|----------------|--------------|------------------------------------------------|
| `TT_METAL_HOME`| Auto-detected| Path to tt-metal repo root                     |
| `ARCH_NAME`    | `wormhole_b0`| Target architecture (N300 uses Wormhole)       |
| `LOGURU_LEVEL` | `INFO`       | Logging verbosity                              |

## Troubleshooting

- **Device not found**: Ensure exactly one N300 PCIe device is visible (`tt-smi` or equivalent).
- **Hugging Face errors**: Run `huggingface-cli login` or set `HF_TOKEN`.
- **Build errors**: Ensure `build/lib` exists and contains TT-Metal shared libraries.
- **Import errors**: Activate the correct Python environment; `PYTHONPATH` is set by the script.
