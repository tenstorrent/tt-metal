# Read coalescing — plain-copy measurements

Measured 2026-10-13 on `bh-50-special-dstoiljkovic-for-reservation-133375`,
Blackhole, 11×10 grid, eight DRAM banks. All arms use the same 104 cores
(13 per bank with affinity enabled); the other six cores are unused.

The operation copies bf16 ROW_MAJOR width-sharded DRAM to ROW_MAJOR
interleaved DRAM. It has **two data-movement kernels and one double-buffered
CB, with no compute kernel**. The reader produces unchanged bytes; the writer
consumes them directly. CB capacity is 8/16/64 KiB per core for 128/256/1024 B
shard rows, identical across switches. Each arm copies the same bytes and
issues the same 32 output row writes per block. Coalescing changes 32 reader
calls into one wide call; hardware burst splitting remains the API's job.

Seven trials × ten launches after three warmups per arm, rotating arm order;
medians below. Each arm is checked bit-exact before timing. `kernel_iters=1`.
Default shapes give equal block counts per core under both assignments.
These copy results replace the earlier tilize measurements.

## Storage and equal-work control for the headline case

For **3328×512 bf16**, bank `b` stores columns `64*b` through `64*b+63`
for all 3328 rows, in row-major order. Each shard row is **128 B**; its 32
consecutive rows form a contiguous **4 KiB** block with no alignment gaps.

- **Affinity off:** each core copies one 32-row band from all eight banks
  (8 banks × 1 block × 4 KiB = **32 KiB/core**).
- **Affinity on:** 13 cores serve each bank; each core copies eight 32-row
  bands within that bank (1 bank × 8 blocks × 4 KiB = **32 KiB/core**).
- Independently, **coalescing off/on** issues 32 × 128 B calls or one 4 KiB
  call for each block. Both issue one read barrier per block.

No data is re-sharded between arms. Coalescing is a transfer-size choice;
affinity is an assignment across successive transfers. On this case the table
below shows both paying off: **88.308 → 27.256 µs** for coalescing, then
**27.256 → 22.657 µs** with affinity added. The affinity-only control is
**60.260 µs**. See the [README](README.md#concrete-storage-and-work-assignment)
for the storage diagram and the other shapes below for the limits of the gain.

## Timing qualification

The board has mismatched core timer bases, so the stock cross-core timestamp
subtraction is unusable. The benchmark reads raw kernel zones, groups starts
separated by more than 1e8 cycles into separate timer clusters, and takes the
**maximum within-cluster span** per program. Every core participates. The raw
header supplies clock frequency. The benchmark verifies the expected program
count before averaging launches.

With one timer base this is ordinary first-start to last-end kernel duration;
with multiple bases it cannot recover the unknown relative start offset
between clusters. These numbers are within-cluster kernel spans, not a
reconstruction of globally synchronized wall time. All arms use the same
placement and reducer. Host setup and input creation are excluded.

## Four-way result

Times in microseconds. Speedups are baseline / variant.

| H × W | Shard-row bytes | Baseline | Coalesce | Affinity | Both | Coalesce speedup | Combined speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3328 × 512 | 128 | 88.308 | 27.256 | 60.260 | 22.657 | 3.24× | 3.90× |
| 6656 × 512 | 128 | 172.961 | 47.271 | 126.046 | 52.177 | 3.66× | 3.31× |
| 3328 × 1024 | 256 | 99.215 | 48.695 | 67.213 | 41.101 | 2.04× | 2.41× |
| 6656 × 1024 | 256 | 188.791 | 86.271 | 139.421 | 91.054 | 2.19× | 2.07× |
| 3328 × 4096 | 1024 | 181.359 | 163.831 | 140.187 | 128.335 | 1.11× | 1.41× |
| 6656 × 4096 | 1024 | 343.166 | 294.634 | 280.249 | 257.800 | 1.16× | 1.33× |

Coalescing alone wins **1.11–3.66×**, strongest for the smallest original reads.
Affinity alone wins **1.22–1.48×**. Added after coalescing, affinity ranges from
**0.91× to 1.28×**: it helps four cases but loses on **6656×512** and
**6656×1024** (52.177 vs 47.271 µs, and 91.054 vs 86.271 µs). Both beat the
unoptimized baseline in every measured case, but **both is not always best**.
The switches stay independent and coalescing alone remains the default.

The output writes can limit the total copy even when read time improves.
These measurements do not isolate read versus write stage cost, so they do
not establish the cause of the affinity regressions. They also do not prove
one full-block call is the best transfer size for every workload.

## Reproduce

```bash
python -m ttnn.operations.examples.read_coalescing --trials 7 --launches 10
```

## Raw summary

```text
read_coalescing   box=bh-50-special-dstoiljkovic-for-reservation-133375  arch=BLACKHOLE  grid=11x10  dram_banks=8  N=7 trials x 10 launches (median)  kernel-iters=1
  timing = max within-timer-cluster kernel span; multi-base programs observed=1796
  op = copy bf16, ROW_MAJOR DRAM (ws = width-sharded, one shard per bank) -> ROW_MAJOR interleaved DRAM; placement = row-major grid; blk = blocks per core

         shape src chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
   3328x512     ws    64    128  104    8   104   8  none        88308    0.5%    base
   3328x512     ws    64    128  104    8   104   8  coalesce     27256    1.1%  3.240x
   3328x512     ws    64    128  104    8   104   8  affinity     60260    1.4%  1.465x
   3328x512     ws    64    128  104    8   104   8  both        22657    1.0%  3.898x
    coalescing added to affinity: 2.660x
    affinity added to coalescing: 1.203x

   6656x512     ws    64    128  208    8   104  16  none       172961    0.8%    base
   6656x512     ws    64    128  208    8   104  16  coalesce     47271    0.9%  3.659x
   6656x512     ws    64    128  208    8   104  16  affinity    126046    0.5%  1.372x
   6656x512     ws    64    128  208    8   104  16  both        52177    2.4%  3.315x
    coalescing added to affinity: 2.416x
    affinity added to coalescing: 0.906x

   3328x1024    ws   128    256  104    8   104   8  none        99215    1.3%    base
   3328x1024    ws   128    256  104    8   104   8  coalesce     48695    0.4%  2.037x
   3328x1024    ws   128    256  104    8   104   8  affinity     67213    1.0%  1.476x
   3328x1024    ws   128    256  104    8   104   8  both        41101    0.7%  2.414x
    coalescing added to affinity: 1.635x
    affinity added to coalescing: 1.185x

   6656x1024    ws   128    256  208    8   104  16  none       188791    2.0%    base
   6656x1024    ws   128    256  208    8   104  16  coalesce     86271    0.3%  2.188x
   6656x1024    ws   128    256  208    8   104  16  affinity    139421    0.9%  1.354x
   6656x1024    ws   128    256  208    8   104  16  both        91054    2.3%  2.073x
    coalescing added to affinity: 1.531x
    affinity added to coalescing: 0.947x

   3328x4096    ws   512   1024  104    8   104   8  none       181359    1.2%    base
   3328x4096    ws   512   1024  104    8   104   8  coalesce    163831    0.9%  1.107x
   3328x4096    ws   512   1024  104    8   104   8  affinity    140187    0.9%  1.294x
   3328x4096    ws   512   1024  104    8   104   8  both       128335    0.4%  1.413x
    coalescing added to affinity: 1.092x
    affinity added to coalescing: 1.277x

   6656x4096    ws   512   1024  208    8   104  16  none       343166    1.3%    base
   6656x4096    ws   512   1024  208    8   104  16  coalesce    294634    0.6%  1.165x
   6656x4096    ws   512   1024  208    8   104  16  affinity    280249    0.5%  1.225x
   6656x4096    ws   512   1024  208    8   104  16  both       257800    0.4%  1.331x
    coalescing added to affinity: 1.087x
    affinity added to coalescing: 1.143x

```

## Validation

All **21 tests passed** (20 exact-output/layout checks plus the six-shape,
four-arm performance sweep). All **20 correctness cases also passed with
watcher/NoC sanitization**, including ragged work splits, every supported shard
width, and repeated in-kernel copies. Profiling was disabled for sanitizer
checks to avoid the known profiler-plus-watcher firmware size limit.

```bash
TT_METAL_DEVICE_PROFILER=0 scripts/run_safe_pytest.sh --dev --no-precompile \
  tests/ttnn/unit_tests/operations/examples/test_read_coalescing.py -k correctness
```
