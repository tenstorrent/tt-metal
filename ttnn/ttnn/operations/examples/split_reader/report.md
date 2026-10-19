# split_reader — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-08-special-astancov-for-reservation-44672` |
| arch | Wormhole B0 (8×8 compute grid) |
| base commit | `89089ffd11e` |
| date | 2026-07-16 |
| metrics | `DEVICE KERNEL DURATION`, `DEVICE NCRISC KERNEL DURATION`, and `DEVICE BRISC KERNEL DURATION` in ns, read in-process with `ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data` |
| method | 5 warmup + 20 timed launches per variant and transaction size, flush-bracketed, on-device duration averaged; all points in one device session |

> Numbers are illustrative of the effect, not a CI bound. Re-run the CLI on the target architecture
> and workload. Add results from another architecture as a new block rather than replacing this one.

## Config A — 8 source cores, 8 tiles/source, 8 tiles/block, transaction-size sweep

The input is height-sharded in L1 across logical source row `(0,0)..(7,0)`. Consumer `(0,1)`
gathers all 64 tiles and BRISC copies them to DRAM in their original order. Split-off puts all
reads on NCRISC. Split-on gives the first four tiles of every block to NCRISC and the second four
to BRISC. Blocks affect only read scheduling and bounded buffering; the DRAM tensor equals the
input exactly and no compute kernel runs.

Depending on transaction size, the consumer issues 64–4,096 reads to transfer the same 128 KiB
from row L1 into its two gather CBs, then writes the same 128 KiB to DRAM in both modes.

```text
 transaction  NoC reads   off NCRISC   on NCRISC   on BRISC  off device  on device   speedup
       32 B       4096     146303.2     74854.8    84739.8    147427.2    84767.1     1.74x
       64 B       2048      82415.9     43319.5    50895.2     83488.8    50925.7     1.64x
      128 B       1024      50434.9     29264.6    34809.9     51507.7    34837.6     1.48x
      256 B        512      34802.6     21928.8    26396.8     35836.3    26434.4     1.36x
      512 B        256      27276.9     18553.7    22523.0     28317.8    22563.5     1.26x
     1024 B        128      24350.0     17314.9    21121.6     25385.4    21163.8     1.20x
     2048 B         64      24051.2     17008.3    20740.8     25085.1    20787.0     1.21x
```

All duration columns are ns/op. The per-RISC columns are profiler kernel lifetimes and include NoC
barriers and CB waits; they are not instruction-level utilization counters. `on BRISC` also
includes the unchanged DRAM writer work.

## Findings

- With split off, NCRISC time rises from **24.1 us** at 64 reads to **146.3 us** at 4,096 reads.
  This is the main signal: more commands add RISC-V issue work while transferred bytes stay fixed.
- At 4,096 reads the single NCRISC is the device bottleneck. Split mode moves half of the read
  workload to BRISC: NCRISC finishes in **74.9 us**, BRISC in **84.7 us**, and device time improves
  from **147.4 us** to **84.8 us** (**1.74×**).
- The gain falls to about **1.20×** at 1–2 KiB transactions, where RISC-V issue overhead is no
  longer the dominant fraction of runtime.
- Both modes pass exact equality against the bf16 input tensor.
- Source placement, consumer placement, CB capacity, input reads, DRAM writes, and output order
  are identical between modes; only the RISC-V assignment of second-half reads changes.
- BRISC always writes the output. Split-on additionally uses it to gather half of each block while
  NCRISC gathers the other half.

## Reproduce

```bash
python -m ttnn.operations.examples.split_reader \
  --cores 8 --tiles-per-core 8 --block-tiles 8 \
  --transaction-bytes 32 64 128 256 512 1024 2048 --iters 20
```

Performance remains reported evidence, not a test assertion.

## 2026-10-12 — helps / gain-shrinks calibration (Wormhole B0 n150)

| stamp | value |
|---|---|
| board | Wormhole B0 n150 (8×8 Tensix grid) |
| branch | `adamjanovic/perf-example-improvements` (base `adce58258c3`) |
| test | `test_split_reader_regimes` (8 source Tensix cores × 8 tiles, 8-tile blocks, bf16) |
| method | 5 warmup launches per variant; then 12 trials per variant, interleaved off/on; a trial = mean per-launch `DEVICE KERNEL DURATION`, `DEVICE NCRISC KERNEL DURATION`, `DEVICE BRISC KERNEL DURATION` over 5 launches; medians compared |

Run 1 (min–max of device time over the 12 trials; per-RISC-V values are median kernel lifetimes):

| regime | bytes per NoC read | reads/launch | off device ns | off NCRISC ns | on device ns | on NCRISC ns | on BRISC ns | speedup |
|---|--:|--:|---|--:|---|--:|--:|--:|
| helps_issue_bound_32B | 32 | 4096 | 147273.5 (147235.6–147332.0) | 146273.2 | 79967.5 (79838.0–81450.6) | 73651.6 | 79948.9 | 1.84× |
| diminished_2048B | 2048 | 64 | 25041.4 (25025.6–25077.4) | 24050.9 | 18182.3 (18175.4–18252.4) | 14830.8 | 18143.8 | 1.38× |

Speedups over four runs: 32 B 1.84 / 1.84 / 1.84 / 1.84×; 2,048 B 1.38 / 1.38 / 1.38 / 1.38×.

Search for a flat regime (`test_split_reader_device_perf` with `SR_*` overrides, mean of 20 launches,
device-time speedup on/off):

| sources × tiles/source, block | 16 B | 32 B | 256 B | 512 B | 1024 B | 2048 B |
|---|--:|--:|--:|--:|--:|--:|
| 8 × 8, block 8 | — | 1.83× | 1.50× | — | 1.36× | 1.38× |
| 8 × 8, block 4 | 1.89× | — | — | 1.32× | — | 1.29× |
| 8 × 8, block 32 | — | 1.84× | 1.59× | — | 1.50× | 1.48× |
| 8 × 32, block 8 | — | 1.81× | 1.52× | — | 1.36× | 1.36× |
| 2 × 4, block 4 | — | 1.83× | 1.50× | — | 1.37× | 1.33× |
| 8 × 1, block 4 | — | 1.80× | 1.47× | — | 1.35× | 1.34× |

No reachable configuration is flat: the reader barriers after every tile, so even with one 2 KiB read
per tile a second issuing RISC-V doubles the tiles in flight. The issue-bound component of the gain
(the part that grows as reads get smaller at fixed bytes) disappears from ~1 KiB per read up; the
~1.3–1.5× floor is per-tile barrier latency, not issue rate.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores (this test uses 8 source + 1 consumer), 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the 2026-10-12 WH block: `test_split_reader_regimes` (5 warmup launches; 12 trials per variant interleaved off/on, a trial = mean per-launch `DEVICE KERNEL DURATION`, `DEVICE NCRISC KERNEL DURATION`, `DEVICE BRISC KERNEL DURATION` over 5 launches; medians) and `test_split_reader_device_perf` |

### Sweep — 8 source Tensix cores × 8 tiles, 8-tile blocks, transaction size

| bytes per NoC read | NoC reads/launch | off NCRISC µs | on device speedup |
|--:|--:|--:|--:|
| 32 | 4096 | 111.4 | **1.90×** |
| 64 | 2048 | 64.0 | **1.85×** |
| 128 | 1024 | 40.2 | **1.78×** |
| 256 | 512 | 28.4 | **1.70×** |
| 512 | 256 | 22.8 | **1.64×** |
| 1024 | 128 | 20.1 | **1.60×** |
| 2048 | 64 | 19.0 | **1.61×** |

Full row from `test_split_reader_device_perf` (64 B): off NCRISC 63971.3, on NCRISC 32015.3, on BRISC 34751.5,
off device 64438.7, on device 34796.2 ns → 1.85×. The other rows are the regimes test (32 B, 2048 B) and probes
of the same A/B.

### Regimes (`test_split_reader_regimes` on BH p150b)

| regime | bytes per NoC read | reads/launch | off device ns | off NCRISC ns | on device ns | on NCRISC ns | on BRISC ns | speedup | asserted |
|---|--:|--:|---|--:|---|--:|--:|--:|---|
| helps_issue_bound_32B | 32 | 4096 | 111869.9 (111812.4–111889.6) | 111402.5 | 58811.2 (58758.2–59560.6) | 55871.8 | 58766.9 | **1.90×** | ≥ 1.4× |
| diminished_2048B | 2048 | 64 | 19432.3 (19412.2–19488.4) | 18970.9 | 12069.7 (12050.6–12084.2) | 10030.6 | 12006.0 | **1.61×** | ≤ 1.75× |

### Reading it
- **Same shape as WH B0 n150, higher floor.** The baseline NCRISC lifetime falls 111.4 → 19.0 µs from 32 B to
  2,048 B per read (WH B0: 146.3 → 24.1 µs) and is flat from ~1 KiB up, so the issue-bound component disappears
  at the same read size on both boards. What is left is 1.60–1.61× on BH p150b vs 1.36–1.38× on WH B0.
- **Why the floor is higher:** at 2,048 B the split's critical RISC-V (BRISC: half the reads + all DRAM writes)
  takes 12.0 µs vs the baseline NCRISC's 19.0 µs (ratio 1.58); on WH B0 it is 18.1 vs 24.1 µs (1.33). BRISC's
  DRAM-write share costs relatively less on BH p150b; this is an observation from the lifetimes, not isolated.
- **No flat regime on either board.** The regimes test therefore asserts the 2,048 B gain stays below 1.75× on
  every board (the issue-bound gain is 1.90× here), not ≤ 1.05×.
