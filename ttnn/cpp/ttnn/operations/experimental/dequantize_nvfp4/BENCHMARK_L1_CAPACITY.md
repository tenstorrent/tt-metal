# L1-capacity benchmark workload

The default performance workload now queries free/contiguous L1 capacity and
uses a balanced number of packed input tiles on every compute core. It leaves
64 KiB per bank for CBs, alignment, and runtime headroom, and prefers a multiple
of sixteen packed tiles/core for regular geometry. Explicit smaller counts and
the historical workload remain selectable through environment variables.

On the measured Blackhole:

| Property | Value |
| --- | ---: |
| Compute cores / L1 banks | 130 / 130 |
| Initial allocatable L1 per bank | 1,460,864 B |
| Packed tiles per core | **128 on every core** |
| BF16 output tiles per core | **512 on every core** |
| Output shape | 16640 x 4096 |
| Reconstructed elements | 68,157,440 |
| Packed input | 32.5 MiB |
| Prepared BF16 scales | 8.125 MiB |
| BF16 output | 130 MiB |
| Total tensor footprint / traffic | 170.625 MiB |
| Measured tensor allocation per bank | 1,376,256 B, **94.2%** of allocatable L1 |
| Free per bank with output live | 84,608 B |

One packed input tile requires 2,048 B packed input, 512 B scales, and four
2,048 B output tiles: 10,752 B total. The current SFPU-scale program uses
25,664 B of CBs per core, within the 64 KiB allowance. CB allocations are not
included in the tensor allocator's peak above. The filled workload passed
watcher/assert-enabled execution. Its measured tensor allocation exactly
matches the estimate.

The previous 2048x4096 workload gave 98 cores sixteen packed tiles and 32 cores
fifteen. The new workload is 8.125 times larger globally, with eight times as
much work on the busiest cores and no load imbalance. Processing still uses
one packed tile / four output tiles per DEST acquire/init batch; the increased
dataset amortizes kernel startup/drain, not every per-batch init or transpose.

## Measurements

Same committed converter, interleaved L1, BF16 DEST/output, SFPU tensor scaling,
global scale 0.8125, all 130 cores, reported clock 1350 MHz. Each workload ran
four eight-call groups. Calls 1–2 of each group were discarded, retaining 24
observations. CPU generation, upload, output sampling/download, allocation,
and synchronization are outside the converter's device-kernel interval.

| Workload | Packed tiles/core | Outputs/core | Median cycles/output tile | MAD | Relative MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 2048x4096 | 15–16 | 60–64 | 526.33 | 7.05 | 1.34% |
| Balanced half-capacity 8320x4096 | 64 | 256 | 481.42 | 6.67 | 1.39% |
| Capacity-filled 16640x4096 | 128 | 512 | **480.23** | 6.60 | 1.37% |

The filled operation took 182.13 us, delivering 374.2 GElem/s across the actual
130 cores. Normalized time fell about 8.8% relative to the historical workload;
half/full sizes are close, indicating that this sweep is near its observed
steady-state rate. This is a workload-size/balance change, not another kernel
optimization. Relative dispersion did not materially improve in this sample.
The increased size therefore reduces normalized overhead without establishing
a reduction in relative per-call noise. Additional repetitions remain useful.

An independent full-capacity repeat measured 479.06 cycles/tile, MAD 4.60
(0.96%). The two filled medians agree within 0.25%, while relative dispersion
varies between runs. Both are archived rather than selecting only the lower
variance result. All
performance cases also check sampled values spanning the output after timing;
the existing reconstruction gate remains separate and unchanged.

## Reproduce

```bash
source source-env.sh
# Default: size against available L1.
scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/test_nvfp4_perf.py --count=4
# Smaller balanced sweep point:
NVFP4_BENCH_PACKED_TILES_PER_CORE=64 scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/test_nvfp4_perf.py --count=4
# Historical, unbalanced workload:
NVFP4_BENCH_LEGACY=1 scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/test_nvfp4_perf.py --count=4
```

Metadata records mode, shape, core assignments, initial allocator capacity,
reserved bytes, predicted footprint, and measured peak/free bytes. Invalid
explicit sizes fail before tensor allocation. The evaluator hashes this test
and compares workload metadata, so old and new workloads cannot be compared
as if they were identical. Fresh baselines are required for future kernel
optimization loops.

Evidence: `generated/nvfp4_l1_capacity/{legacy,half,full,full-repeat}/`, containing
per-case timing samples, JUnit, commands, source hashes/disassembly, and raw
Tracy reports. `filled-dev.json` and `filled-dev.xml` record the watcher run.
