# SDPA Streaming v2 Ring: Code Size Analysis (T3000)

Date: 2026-03-11
Branch: `dnijemcevic/sdpa_code_size_sweep`
Machine: T3000 (2×4 Wormhole mesh, 8 devices)

## Summary

This analysis measures code size across 3 production model configs on T3000, and the impact
of removing `__attribute__((noinline, noclone))` and `#pragma GCC unroll 1` from the 4 hot
functions in `compute_streaming.hpp`. The baseline code on this branch is identical to main:
all 4 attributes and all 4 unroll pragmas are present.

Key finding: **removing both attributes and unroll pragmas causes Mochi to exceed the device
code size limit** (`state.offset < max_size`), confirming that the attributes are not just a
performance tradeoff but a correctness requirement for the largest configs.

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

All tests from `tests/nightly/t3000/ccl/test_ring_joint_attention.py`:

| Test | Model | nh | d | seq | joint | q_chunk | k_chunk | sp×tp | Grid |
|------|-------|----|---|-----|-------|---------|---------|-------|------|
| `test_ring_joint_sdpa_sd35_model_config` | SD3.5 | 38 | 64 | 4096 | 333 | 256 | 512 | 4×2 | 8×7 |
| `test_ring_joint_sdpa_mochi_model_config` | Mochi | 24 | 128 | 4000 | 118 | 256 | 256 | 2×2 | 8×7 |
| `test_ring_joint_sdpa_wan_14b_720p_model_config` | Wan 14B | 40 | 128 | 75600 | 0 | 256 | 256 | 2×4 | 7×8 |

### Tile dimensions and subblock sizing

All configs use `fp32_dest_acc=false` → `dst_size=8`.

| Config | Sq_chunk_t | Sk_chunk_t | DHt | sbh | qkt_subblock_w | q_num_subblocks |
|--------|-----------|-----------|-----|-----|---------------|----------------|
| SD3.5 | 8 | 16 | 2 | 2 | 4 | 4 |
| Mochi | 8 | 8 | 4 | 2 | 4 | 4 |
| Wan 14B | 8 | 8 | 4 | 2 | 4 | 4 |

All configs satisfy the streaming v2 gate (`use_streaming_compute = true`):
1. `!fp32_dest_acc_en` — true
2. `qk_out_subblock_h <= 2` — true (sbh=2)
3. `Sk_chunk_t % (8 / sbh) == 0` — true (16%4==0, 8%4==0)
4. `qk_in0_num_subblocks > 1` — true (4 > 1)

## Baseline Code Size (all 3 configs)

All measurements with `noinline,noclone` on all 4 functions + `#pragma GCC unroll 1`.
Sizes are `text + data + bss` (dec column from `riscv-tt-elf-size`).

| Core | SD3.5 (d=64, q256/k512) | Mochi (d=128, q256/k256) | Wan 14B (d=128, q256/k256) |
|------|--------:|--------:|--------:|
| trisc0 (UNPACK) | 17,192 | 17,752 | 17,460 |
| trisc1 (MATH) | 12,396 | 13,528 | 13,320 |
| trisc2 (PACK) | 14,204 | 15,692 | 15,388 |
| brisc (WRITER) | 6,340 | 6,292 | 6,048 |
| ncrisc (READER) | 4,940 | 4,936 | 4,448 |
| **Compute Total** | **43,792** | **46,972** | **46,168** |
| **Grand Total** | **55,072** | **58,200** | **56,664** |

### Analysis

- **Mochi produces the largest compute code** (46,972 bytes), making it the binding constraint
  for code size optimizations.
- Mochi and Wan share `DHt=4, Sk_chunk_t=8` but differ in joint sequence (Mochi: 118, Wan: 0)
  and masking config. The ~800-byte difference comes from `constexpr` mask branches, not the
  joint iteration loop itself (which is runtime, not `if constexpr`).
- SD3.5 is smallest on compute (43,792) because `DHt=2` (d=64) produces simpler template
  instantiations than `DHt=4` (d=128).
- Reader/writer (brisc/ncrisc) vary slightly across configs but are small and unaffected by
  compute kernel attributes.

## noinline/noclone Removal (Mochi)

Removed `__attribute__((noinline, noclone))` from all 4 functions. Unroll pragmas kept.

| Core | Baseline | No attrs | Delta | % |
|------|--------:|--------:|------:|----:|
| trisc0 (UNPACK) | 17,752 | 19,552 | +1,800 | +10.1% |
| trisc1 (MATH) | 13,528 | 15,108 | +1,580 | +11.7% |
| trisc2 (PACK) | 15,692 | 20,608 | +4,916 | +31.3% |
| brisc (WRITER) | 6,292 | 6,292 | 0 | 0% |
| ncrisc (READER) | 4,936 | 4,936 | 0 | 0% |
| **Compute Total** | **46,972** | **55,268** | **+8,296** | **+17.7%** |
| **Grand Total** | **58,200** | **66,496** | **+8,296** | **+14.3%** |

### Analysis

- **trisc2 (PACK)** sees the largest growth (+31.3%), consistent with `sub_exp_block_bcast_cols`
  being the dominant contributor (5 call sites, heavy pack loops).
- PCC is identical across baseline and no-attrs (same values to the decimal).
- Reader/writer unaffected as expected (attributes are on compute functions only).

## noinline/noclone + unroll Removal (Mochi)

Removed both `__attribute__((noinline, noclone))` from all 4 functions AND all 4
`#pragma GCC unroll 1` directives from `sub_exp_block_bcast_cols`.

| Core | Baseline | No attrs | No attrs + no unroll |
|------|--------:|--------:|--------:|
| trisc0 (UNPACK) | 17,752 | 19,552 (+1,800) | 19,552 (+1,800) |
| trisc1 (MATH) | 13,528 | 15,108 (+1,580) | 15,108 (+1,580) |
| trisc2 (PACK) | 15,692 | 20,608 (+4,916) | 23,264 (+7,572) |
| brisc (WRITER) | 6,292 | 6,292 (0) | 6,292 (0) |
| ncrisc (READER) | 4,936 | 4,936 (0) | 4,936 (0) |
| **Compute Total** | **46,972** | **55,268** (+8,296) | **57,924** (+10,952) |
| **Grand Total** | **58,200** | **66,496** (+8,296) | **69,152** (+10,952) |
| **Headroom** (70,656 limit) | **12,456** | **4,160** | **-192 (OVERFLOW)** |

### Analysis

- Removing unroll pragmas adds +2,656 bytes to trisc2 only (pack loops in
  `sub_exp_block_bcast_cols`). trisc0/trisc1 are unaffected — the pragmas are in pack loops.
- **This configuration crashes at runtime**:
  ```
  TT_FATAL: Program size (70848) too large for kernel config buffer (70656) on TENSIX
  ```
  The TENSIX kernel config buffer limit is **70,656 bytes**. The "no attrs + no unroll"
  program is 70,848 bytes — overflowing by just **192 bytes**. (The program size includes
  per-core metadata beyond raw ELF text+data+bss, which is why it exceeds the grand total.)
- Even the intermediate "no attrs, keep unroll" config at 66,496 bytes grand total passes
  but leaves only ~4 KB of headroom below the 70,656 limit.

## Conclusion

**The baseline (noinline/noclone + unroll 1) must be kept for T3000 production configs.**

- Removing attributes alone costs +8.3 KB (+17.7% compute) on the largest config (Mochi).
- Removing both attributes and unroll pragmas exceeds the device code size limit entirely.
- The attributes are not a performance optimization but a **code size constraint** — without
  them, GCC's inlining and IPA-CP cloning of the 4 hot template functions (especially
  `sub_exp_block_bcast_cols` with 5 call sites) bloats the binary past device limits.

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

1. **Clear JIT cache** (critical — stale cache gives wrong results):
   ```bash
   rm -rf built/tt-metal-cache*
   ```

2. **Run any test** that exercises the kernel to trigger JIT compilation:
   ```bash
   tt-smi -r 0,1,2,3
   pytest "tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_mochi_model_config" -s
   ```
   The test will pass or fail, but either way the ELFs are compiled.

3. **Find the compiled ELFs**:
   ```bash
   CACHE=$(find built -maxdepth 1 -name "tt-metal-cache*" -type d | head -1)
   SDPA_HASH=$(ls "$CACHE/kernels/ring_joint_sdpa/")
   READER_HASH=$(ls "$CACHE/kernels/ring_joint_reader/")
   WRITER_HASH=$(ls "$CACHE/kernels/ring_joint_writer/")
   ```

4. **Measure all 5 RISC-V cores**:
   ```bash
   RISCV_SIZE=runtime/sfpi/compiler/bin/riscv-tt-elf-size

   echo "=== Compute (trisc0/1/2) ==="
   for c in trisc0 trisc1 trisc2; do
       echo "--- $c ---"
       $RISCV_SIZE "$CACHE/kernels/ring_joint_sdpa/$SDPA_HASH/$c/$c.elf"
   done

   echo "=== Writer (brisc) ==="
   $RISCV_SIZE "$CACHE/kernels/ring_joint_writer/$WRITER_HASH/brisc/brisc.elf"

   echo "=== Reader (ncrisc) ==="
   $RISCV_SIZE "$CACHE/kernels/ring_joint_reader/$READER_HASH/ncrisc/ncrisc.elf"
   ```

   Output columns: `text` (code), `data` (initialized data), `bss` (zero-initialized data),
   `dec` (total).

### Notes
- Different compile-time args produce different kernel hashes → different ELFs. Make sure
  you're measuring the config you care about by clearing the cache and running only one test.
- `trisc0` = UNPACK, `trisc1` = MATH, `trisc2` = PACK, `brisc` = WRITER, `ncrisc` = READER.
- The JIT cache location is `built/tt-metal-cache*` (under the repo root). The `built/`
  directory is a symlink to `build_Debug` or `build_Release`.
- The test does not need to pass for ELFs to be measurable — JIT compilation happens before
  the size check that may reject the kernel.
- For A/B comparisons: edit `compute_streaming.hpp` → `rm -rf built/tt-metal-cache*` → run
  test → measure. No rebuild needed (kernel source is JIT-compiled, not part of the host build).

## Available Model Config Tests

```bash
# SD3.5 — smallest compute code (d=64)
pytest "tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_sd35_model_config" -s

# Mochi — largest compute code (d=128, joint=118), binding constraint
pytest "tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_mochi_model_config" -s

# Wan 14B 720p — large compute code (d=128, joint=0), long-running (~2.5 min)
pytest "tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_wan_14b_720p_model_config" -s
```
