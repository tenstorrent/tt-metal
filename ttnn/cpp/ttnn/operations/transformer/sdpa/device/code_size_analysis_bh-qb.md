# SDPA Streaming v2 Ring: Code Size Analysis (BH Quiet Box)

Date: 2026-03-11
Branch: `main`
Machine: Blackhole quiet box (4 devices)

## Summary

This analysis measures code size across all q/k chunk permutations on a Blackhole quiet box,
and the impact of removing `__attribute__((noinline, noclone))` and `#pragma GCC unroll 1`
from the 4 hot functions in `compute_streaming.hpp`. The baseline code is main: all 4
attributes and all 4 unroll pragmas are present.

Three levels of analysis:
1. **Baseline code size** across all 12 q/k chunk permutations (2 seq_lens x 3 q_chunks x 2 k_chunks).
2. **All-or-nothing removal** of attrs and unroll pragmas on the worst-case config (q256/k256).
3. **Per-function removal** — each of the 4 functions' `noinline,noclone` removed individually,
   with code size and math utilization measured on two reference configs (q224/k512, q288/k512).

**Conclusion**: Remove all `noinline,noclone` attributes. The ~5 KB code size increase (worst
case ~62.5 KB) is well within device limits, and yields +1.0–1.1pp math utilization.
`sub_exp_block_bcast_cols` accounts for ~92% of the code growth and ~70–80% of the perf gain.
Removing `#pragma GCC unroll 1` adds code size with no perf benefit — keep them.

## Current State (Baseline)

All 4 functions have `__attribute__((noinline, noclone))`. The 4 `#pragma GCC unroll 1`
directives in `sub_exp_block_bcast_cols` are present.

```cpp
// compute_streaming.hpp — 4 attributed functions:

blocked_matmul_and_pack(...)              // 5 call sites
sub_exp_block_bcast_cols(...)             // 5 call sites
sub_exp_first_col_blocks(...)             // 2 call sites
apply_lightweight_mask_streaming(...)     // 1 call site
```

## Test Configs

All tests from `test_ring_joint_attention_scaled_dot_product_attention_sprint.py`,
accuracy test (`test_ring_joint_attention_sdpa_accuracy`).

Wan 2.2 compatible: `nh=10, d=128, batch=1`, `fp32_dest_acc=false`.

Sweep: `seq_len ∈ {2240, 8544}`, `q_chunk ∈ {224, 256, 288}`, `k_chunk ∈ {256, 512}`.

## Baseline Code Size (all configs)

All measurements with `noinline,noclone` on all 4 functions + `#pragma GCC unroll 1`.
Sizes are `text + data + bss` (dec column from `riscv-tt-elf-size`).

### s=2240

| Core | q224/k256 | q224/k512 | q256/k256 | q256/k512 | q288/k256 | q288/k512 |
|------|----------:|----------:|----------:|----------:|----------:|----------:|
| trisc0 (UNPACK) | 17,036 | 16,972 | 17,440 | 17,388 | 16,948 | 16,880 |
| trisc1 (MATH) | 11,724 | 11,648 | 12,472 | 11,848 | 11,664 | 11,660 |
| trisc2 (PACK) | 16,152 | 15,320 | 16,968 | 16,908 | 16,136 | 15,220 |
| brisc (WRITER) | 6,084 | 6,084 | 6,076 | 6,076 | 6,120 | 6,120 |
| ncrisc (READER) | 4,556 | 4,644 | 4,496 | 4,552 | 4,472 | 4,532 |
| **Compute Total** | **44,912** | **43,940** | **46,880** | **46,144** | **44,748** | **43,760** |
| **Grand Total** | **55,552** | **54,668** | **57,452** | **56,772** | **55,340** | **54,412** |

### s=8544

| Core | q224/k256 | q224/k512 | q256/k256 | q256/k512 | q288/k256 | q288/k512 |
|------|----------:|----------:|----------:|----------:|----------:|----------:|
| trisc0 (UNPACK) | 16,888 | 17,204 | 17,288 | 17,624 | 16,796 | 17,116 |
| trisc1 (MATH) | 11,500 | 11,772 | 12,332 | 11,956 | 11,488 | 11,760 |
| trisc2 (PACK) | 16,024 | 15,572 | 16,808 | 17,124 | 16,028 | 15,496 |
| brisc (WRITER) | 6,116 | 6,116 | 6,076 | 6,076 | 6,080 | 6,080 |
| ncrisc (READER) | 4,492 | 4,476 | 4,500 | 4,484 | 4,556 | 4,536 |
| **Compute Total** | **44,412** | **44,548** | **46,428** | **46,704** | **44,312** | **44,372** |
| **Grand Total** | **55,020** | **55,140** | **57,004** | **57,264** | **54,948** | **54,988** |

### Analysis

- **q256 configs are consistently the largest** (~46–47 KB compute vs ~44 KB for q224/q288),
  ~2–3 KB bigger across all cores.
- **seq_len has minimal impact** on code size — same q/k configs produce very similar sizes
  across s=2240 and s=8544.
- **Worst case is q256/k256** at ~57 KB grand total (both seq_lens).
- **brisc/ncrisc are stable** across all configs (~6 KB / ~4.5 KB), as expected for dataflow
  kernels.
- **Grand total range**: 54.4 KB (q288/k512) to 57.5 KB (q256/k256).

## noinline/noclone + unroll Removal (worst case: s=2240, q256/k256)

Measured on the worst-case config (q256/k256, largest baseline).

| Core | Baseline | No attrs | No attrs + no unroll |
|------|--------:|--------:|--------:|
| trisc0 (UNPACK) | 17,440 | 18,268 (+828) | 18,268 (+828) |
| trisc1 (MATH) | 12,472 | 12,044 (-428) | 12,044 (-428) |
| trisc2 (PACK) | 16,968 | 20,588 (+3,620) | 22,540 (+5,572) |
| brisc (WRITER) | 6,076 | 6,076 (0) | 6,076 (0) |
| ncrisc (READER) | 4,496 | 4,496 (0) | 4,496 (0) |
| **Compute Total** | **46,880** | **50,900** (+4,020) | **52,852** (+5,972) |
| **Grand Total** | **57,452** | **61,472** (+4,020) | **63,424** (+5,972) |

### Analysis

- **No attrs**: +4,020 bytes compute (+8.6%). Almost entirely from trisc2 PACK (+3,620)
  due to inlining/cloning of `sub_exp_block_bcast_cols` across 5 call sites. trisc1 MATH
  actually *shrinks* by 428 bytes — the compiler generates better code when free to inline.
- **No attrs + no unroll**: +5,972 bytes compute (+12.7%). The unroll pragmas add another
  +1,952 bytes on top, all on trisc2 PACK.
- **trisc0 (UNPACK)** growth is identical with and without unroll pragmas — the pragmas are
  in pack loops only.
- **Reader/writer unaffected** as expected (attributes are on compute functions only).
- All three configurations are **well within device limits** on BH quiet box — no overflow
  risk here (unlike T3000/Mochi where the limit is 70,656 bytes).

## Performance Impact (Math Utilization)

CCLs commented out for isolated compute measurement. Two reference configs measured with
tracy profiling via `test_ring_joint_attention_create_perf_table`.

| Config | Baseline | No attrs | No attrs + no unroll |
|--------|--------:|---------:|---------------------:|
| s=2240 q224/k512 | 61.1% | 62.1% (+1.0pp) | 62.1% (+1.0pp) |
| s=8544 q288/k512 | 63.5% | 64.6% (+1.1pp) | 64.4% (+0.9pp) |

### Analysis

- Removing `noinline/noclone` gives a consistent **~1 percentage point** improvement in math
  utilization across both workloads. The compiler generates faster code when free to inline
  the 4 hot functions across their call sites.
- Removing `#pragma GCC unroll 1` on top of removing attrs has **no additional benefit** —
  the numbers are within measurement noise of no-attrs-only. The unroll pragmas add code size
  (+1,952 bytes on trisc2) but do not affect perf on these configs.
- The perf gain comes entirely from inlining/cloning; unroll pragmas are neutral for perf.

## Per-Function Attribute Removal

Each row removes `noinline,noclone` from exactly one function, keeping it on the other three.
Code size measured with CCLs enabled (accuracy test). Perf measured with CCLs commented out
(isolated compute). Delta is compute total vs baseline.

### s=2240, q224/k512

| Variant | Sites | trisc0 | trisc1 | trisc2 | brisc | ncrisc | Compute | Grand | Delta | Math Util |
|---------|------:|-------:|-------:|-------:|------:|-------:|--------:|------:|------:|----------:|
| Baseline (all attrs) | — | 16,972 | 11,648 | 15,320 | 6,084 | 4,644 | 43,940 | 54,668 | — | 61.1% |
| No: `blocked_matmul_and_pack` | 5 | 17,260 | 11,376 | 15,800 | 6,084 | 4,644 | 44,436 | 55,164 | +496 | 61.5% |
| No: `sub_exp_block_bcast_cols` | 5 | 17,420 | 11,808 | 19,240 | 6,084 | 4,644 | 48,468 | 59,196 | +4,528 | 61.8% |
| No: `sub_exp_first_col_blocks` | 2 | 17,332 | 11,640 | 15,308 | 6,084 | 4,644 | 44,280 | 55,008 | +340 | 61.1% |
| No: `apply_lightweight_mask_streaming` | 1 | 16,844 | 11,456 | 14,816 | 6,084 | 4,644 | 43,116 | 53,844 | -824 | 61.3% |
| No attrs (all removed) | — | 18,220 | 11,388 | 19,216 | 6,084 | 4,644 | 48,824 | 59,552 | +4,884 | 62.1% |

### s=8544, q288/k512

| Variant | Sites | trisc0 | trisc1 | trisc2 | brisc | ncrisc | Compute | Grand | Delta | Math Util |
|---------|------:|-------:|-------:|-------:|------:|-------:|--------:|------:|------:|----------:|
| Baseline (all attrs) | — | 17,116 | 11,760 | 15,496 | 6,080 | 4,536 | 44,372 | 54,988 | — | 63.5% |
| No: `blocked_matmul_and_pack` | 5 | 17,488 | 11,472 | 16,100 | 6,080 | 4,536 | 45,060 | 55,676 | +688 | 63.7% |
| No: `sub_exp_block_bcast_cols` | 5 | 17,596 | 11,952 | 19,412 | 6,080 | 4,536 | 48,960 | 59,576 | +4,588 | 64.3% |
| No: `sub_exp_first_col_blocks` | 2 | 17,460 | 11,760 | 15,476 | 6,080 | 4,536 | 44,696 | 55,312 | +324 | 63.4% |
| No: `apply_lightweight_mask_streaming` | 1 | 17,000 | 11,556 | 15,016 | 6,080 | 4,536 | 43,572 | 54,188 | -800 | 63.6% |
| No attrs (all removed) | — | 18,364 | 11,616 | 19,352 | 6,080 | 4,536 | 49,332 | 59,948 | +4,960 | 64.6% |

### Analysis

- **`sub_exp_block_bcast_cols` dominates**: +4.5 KB code size, +0.7–0.8pp perf. 5 call sites
  across both Q@KT and QKT@V phases — inlining this function is the main driver of both code
  size growth and perf improvement. Nearly all growth is on trisc2 PACK (+3.9 KB).
- **`blocked_matmul_and_pack`**: modest +0.5–0.7 KB, +0.2–0.4pp perf. Despite 5 call sites,
  the matmul function body is large enough that inlining yields diminishing returns.
- **`sub_exp_first_col_blocks`**: near-neutral (+0.3 KB, 0.0/−0.1pp). Only 2 call sites in
  the SALAD correction path — not on the hot path often enough to matter.
- **`apply_lightweight_mask_streaming`**: **shrinks code by ~0.8 KB** with +0.1–0.2pp perf.
  With only 1 call site, removing `noinline` lets the compiler eliminate call/return overhead
  and dead code, reducing the binary. This is a pure win.
- **No-attrs total (+4.9 KB, +1.0–1.1pp)** is roughly the sum of `sub_exp_block_bcast_cols` +
  `blocked_matmul_and_pack`, partially offset by the `apply_lightweight_mask_streaming` shrinkage.
- **Recommendation**: Remove all `noinline,noclone` attributes. The marginal cost of going
  from "only `sub_exp_block_bcast_cols`" to "all removed" is ~360 bytes for +0.3pp — cheap
  because `apply_lightweight_mask_streaming` shrinks the binary by ~0.8 KB, offsetting
  growth from the other two. Worst-case config (q256/k256, ~57.5 KB baseline) lands at
  ~62.5 KB with all attrs removed, well under the 70,656 byte T3000/Mochi limit. Simpler
  code with no attributes to maintain, and no perf left on the table.

## How to Measure Code Size

### Prerequisites
```bash
source python_env/bin/activate
```

### Toolchain
The RISC-V binutils are at:
```
runtime/sfpi/compiler/bin/riscv-tt-elf-size
runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
runtime/sfpi/compiler/bin/riscv-tt-elf-nm
runtime/sfpi/compiler/bin/riscv-tt-elf-readelf
```

### Steps

1. **Build and clear JIT cache** (critical — stale cache gives wrong results):
   ```bash
   ./build_metal.sh
   rm -rf ~/.cache/tt-metal-cache
   ```

2. **Run any test** that exercises the kernel to trigger JIT compilation:
   ```bash
   tt-smi -r 0,1,2,3
   pytest "tests/tt_eager/python_api_testing/unit_testing/test_ring_joint_attention_scaled_dot_product_attention_sprint.py::test_ring_joint_attention_sdpa_accuracy" -s
   ```

3. **Find the compiled ELFs**:
   ```bash
   find ~/.cache/tt-metal-cache -path "*/ring_joint_sdpa/*/trisc0" -type d
   ```
   The kernel hash directory contains `trisc0/`, `trisc1/`, `trisc2/` subdirectories.

4. **Measure all 5 RISC-V cores**:
   ```bash
   KERNEL_DIR=<path from step 3, without /trisc0>
   RISCV_SIZE=runtime/sfpi/compiler/bin/riscv-tt-elf-size

   for c in trisc0 trisc1 trisc2; do
       echo "=== $c ==="
       $RISCV_SIZE $KERNEL_DIR/$c/$c.elf
   done

   # Writer and reader are under separate kernel hash directories:
   WRITER_DIR=$(find ~/.cache/tt-metal-cache -path "*/ring_joint_writer/*/brisc" -type d | head -1)
   echo "=== brisc (WRITER) ==="
   $RISCV_SIZE $WRITER_DIR/brisc.elf

   READER_DIR=$(find ~/.cache/tt-metal-cache -path "*/ring_joint_reader/*/ncrisc" -type d | head -1)
   echo "=== ncrisc (READER) ==="
   $RISCV_SIZE $READER_DIR/ncrisc.elf
   ```

   Output columns: `text` (code), `data` (initialized data), `bss` (zero-initialized data),
   `dec` (total).

### Notes
- Different compile-time args produce different kernel hashes → different ELFs. Make sure
  you're measuring the config you care about by clearing the cache and running only one test.
- `trisc0` = UNPACK, `trisc1` = MATH, `trisc2` = PACK, `brisc` = WRITER, `ncrisc` = READER.
- The JIT cache location is `~/.cache/tt-metal-cache`.
- For A/B comparisons: edit `compute_streaming.hpp` → `./build_metal.sh` →
  `rm -rf ~/.cache/tt-metal-cache` → run test → measure.
- The `.elf.xip.elf` variant has the same sizes; use either.

### Sprint test file

The ring joint attention sprint test lives at:
```
tests/tt_eager/python_api_testing/unit_testing/test_ring_joint_attention_scaled_dot_product_attention_sprint.py
```

This file is **not committed**. It originates from `origin/pjosipovic/sdpa_wan_sprint`.
To restore it:
```bash
git show origin/pjosipovic/sdpa_wan_sprint:tests/tt_eager/python_api_testing/unit_testing/test_ring_joint_attention_scaled_dot_product_attention_sprint.py \
  > tests/tt_eager/python_api_testing/unit_testing/test_ring_joint_attention_scaled_dot_product_attention_sprint.py
```

### Restrict to a single config

Edit the sprint test file (look for `# HACK WHEN NECESSARY` comments):
```python
NON_GALAXY_SEQ_LENS_PER_DEVICE = [2240]   # or [8544]
Q_CHUNK_SIZES = [256]                      # or [224], [288]
K_CHUNK_SIZES = [256]                      # or [512]
```

### CCL management for perf testing

For isolated compute perf measurement, comment out the CCL data transfer and synchronization
in the reader/writer kernels. This simulates "perfect" CCLs with zero cross-device latency —
the CB push/pop flow and loop structure remain intact so compute sees realistic CB protocol
timing, but the data in remote-slice CBs is garbage (stale L1).

**Important**: PCC/accuracy is invalid with CCLs commented out (RMSE ~0.26). Always restore
before running accuracy tests.

The two files to modify:
```
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/
  ring_attention_all_gather_reader.cpp
  ring_attention_all_gather_writer.cpp
```

**Reader** — comment out:
- All `noc_async_read()` calls (both initial local reads and remote slice reads)
- All `noc_async_read_barrier()` calls
- The `noc_semaphore_wait_min()` that waits for "slice arrived" signals from remote devices

**Writer** — comment out:
- All `scatter_fabric_write_unidir()` and `fabric_write_unidir()` calls (cross-device data)
- All `fabric_direction_connection->wait_for_empty_write_slot()` calls
- All `fabric_direction_connection->send_payload_flush_blocking_from_address()` calls
- The final `noc_async_write_barrier()` and `noc_async_atomic_barrier()`

A reference commit with these comment-outs applied is `d327858d00`. To toggle:
```bash
CCL_DIR=ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels

# Comment out CCLs (for perf):
git show d327858d00:${CCL_DIR}/ring_attention_all_gather_reader.cpp \
  > ${CCL_DIR}/ring_attention_all_gather_reader.cpp
git show d327858d00:${CCL_DIR}/ring_attention_all_gather_writer.cpp \
  > ${CCL_DIR}/ring_attention_all_gather_writer.cpp

# Restore CCLs (for accuracy):
git checkout HEAD -- ${CCL_DIR}/ring_attention_all_gather_reader.cpp \
  ${CCL_DIR}/ring_attention_all_gather_writer.cpp
```

### Automated per-function sweep

The per-function attribute removal sweep is automated in `run_per_func_sweep.sh` (repo root,
not committed). It handles:
- Applying each variant (sed on `compute_streaming.hpp`)
- Toggling CCLs between accuracy (code size) and perf modes
- Building, clearing JIT cache, resetting devices
- Running accuracy tests (code size) and perf table tests (math utilization)
- Measuring ELF sizes via `riscv-tt-elf-size`
- Collecting results to `/tmp/per_func_sweep_results/`

The script runs Phase 1 (code size, CCLs enabled) then Phase 2 (perf, CCLs commented out),
and restores `compute_streaming.hpp` at the end. Total runtime: ~1.5–2 hours for 9 builds +
18 test runs on a 4-device BH quiet box.
