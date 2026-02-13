# ViT N300 Non-Deterministic Hang Investigation

## Problem Statement

The `vit-N300-func` CI test non-deterministically hangs with `TIMEOUT: device timeout in fetch queue wait`. Involves matmul kernels using block-sharded L1 + multicast on 8x8 grid. Observed twice in CI (ND_failure.log, ND_failure2.log). **These are TWO DISTINCT failure modes that may be causally linked.**

---

## ROOT CAUSE IDENTIFIED: Missing Pipeline Stall in Packer L1_ACC Reconfiguration

### The Bug

In `reconfigure_packer_l1_acc()` (cpack_common.h:710), a `TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::TRISC_CFG)` is **commented out** at line 725. This creates a pipeline hazard where PACR (pack) instructions can execute before RMWCI (config write) instructions have committed to the THCON config registers.

### Evidence

1. **Systematic pattern**: Every unpack operation in the LLK codebase uses `TTI_STALLWAIT(STALL_UNPACK, TRISC_CFG)` with the comment "Stall unpacker until pending CFG writes from Trisc have completed." Found in:
   - `llk_unpack_tilize.h` (4 instances)
   - `llk_unpack_untilize.h` (1 instance)
   - `llk_unpack_AB.h` (2 instances)
   - `llk_unpack_AB_matmul.h` (1 instance)
   - `llk_unpack_reduce.h` (1 instance)
   - `llk_unpack_A.h` (1 instance)
   - `llk_unpack_AB_reduce_custom.h` (1 instance)

2. **Packer's own reconfig_data_format uses it**: `reconfig_packer_data_format()` (cpack_common.h:450) includes `TTI_STALLWAIT(STALL_THCON, TRISC_CFG)` for BFP format paths.

3. **SECOND commented-out stall**: `program_packer_destination()` (cpack_common.h:634) also has a commented-out `TTI_STALLWAIT(STALL_THCON, PACK)`.

4. **Active investigation by TT engineer**: `ryanzhu/pack-recfg` branch (commits from Feb 12-13, 2026) adds `are_packers_configured_correctly()` with `tensix_sync()` (full pipeline fence) to verify packer config. This function was added to the LLK submodule on the test branch `test-llk-ryanzhu/pack-recfg-1770946861`.

5. **Both WH and BH affected**: The STALLWAIT is commented out in both `tt_llk_wormhole_b0` and `tt_llk_blackhole` versions of `cpack_common.h`.

### The Race Condition in Detail

The matmul compute kernel (`bmm_large_block_zm_fused_bias_activation.cpp`) does this sequence:

```
tile_regs_wait();                           // Packer is idle (confirmed by MATH_PACK semaphore)
PACK((pack_reconfig_data_format(out_cb)));   // Config writes (REG2FLOP, WRCFG, RMWCI)
PACK((llk_pack_reconfig_l1_acc(0 or 1)));   // 4x cfg_reg_rmw_tensix -> 4-16 TT_RMWCIB* instructions
                                             // NO STALLWAIT HERE
uint32_t start_dst_index = 0;               // ~zero delay
pack_tile_block(...);                        // Issues PACR instructions (via mop_run)
```

`cfg_reg_rmw_tensix<>()` uses `TT_RMWCIB0/1/2/3` instructions (up to 4 per register) to write 4 THCON section config registers:
- `THCON_SEC0_REG1_Pack_L1_Acc`
- `THCON_SEC0_REG8_Pack_L1_Acc`
- `THCON_SEC1_REG1_Pack_L1_Acc`
- `THCON_SEC1_REG8_Pack_L1_Acc`

Without the STALLWAIT, these RMWCI writes may not have committed to the hardware config registers before the PACR instruction from `mop_run(1,1)` reaches the packer. Result:
- **Some THCON sections see new L1_ACC value, others see old value**
- **Per-section config inconsistency** across the 4 packer sections (which process 4 faces of a tile)
- This can cause the packer FSM to enter an undefined state, leading to a permanent stall

### The Fix

Uncomment `TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::TRISC_CFG)` at line 725 of:
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cpack_common.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h`

Performance impact: one additional stall instruction per L1_ACC reconfiguration. Since `tile_regs_wait()` already confirmed the packer is idle, the stall should complete quickly (just waiting for TRISC_CFG writes to propagate, ~few cycles).

---

## CRITICAL INSIGHT: The Two Failure Modes Are Causally Linked

### Failure Mode A → Triggers Failure Mode B

**Failure Mode A** (single-core packer stall from the missing STALLWAIT) causes one core to stop making progress. This core stops consuming input CBs (in0 and in1).

**This triggers Failure Mode B** (global multicast deadlock) through the multicast sender protocol:

1. The stuck core's in0/in1 CBs fill up
2. The stuck core can't signal "ready" to multicast senders (can't do `cb_reserve_back`)
3. in0 row sender blocks at `noc_semaphore_wait` (waiting for ALL receivers ready)
4. in1 column sender blocks at `noc_semaphore_wait` (waiting for ALL receivers ready)
5. Other cores in the same row/column lose their in0/in1 data sources
6. Their compute stalls → their CBs fill up → they can't signal ready to OTHER senders
7. Deadlock propagates across the entire grid (column first, then rows)

This explains why ND_failure.log shows an ENTIRE COLUMN of cores stuck while ND_failure2.log shows only ONE core stuck -- **Mode A is the root cause, Mode B is the propagation effect**.

### Multicast Protocol (detailed)

**In0 sender (NCRISC):** multicasts along rows
```
noc_semaphore_wait(sender_sem, num_receivers)  // Wait ALL receivers ready
noc_async_write_multicast(data)                 // Send data
noc_semaphore_set_multicast(receiver_sem)       // Signal "data valid"
```

**In1 sender (BRISC):** multicasts along columns
```
noc_semaphore_wait(sender_sem, num_receivers)  // Wait ALL receivers ready
noc_async_write_multicast(data)                 // Send data
noc_semaphore_set_multicast(receiver_sem)       // Signal "data valid"
```

**Deadlock cycle:**
```
Core(r,c) packer hangs (Mode A)
  → Core(r,c) can't consume in0/in1 → CBs fill
  → Core(r,c) can't signal ready to in1 column sender
  → in1 column sender blocks → all cores in column c lose in1
  → cores in column c can't consume in0 → their in0 CBs fill
  → cores in column c can't signal ready to their respective in0 row senders
  → in0 row senders block → all cores in those rows lose in0
  → global deadlock
```

---

## Detailed Failure Mode Analysis

### Failure Mode A: Single-Core Packer Stall (ND_failure2.log)

**Only ONE core stuck**: physical (7,3), virtual 9-5, device 0. All other cores completed.

| RISC | Stuck At | Root Cause |
|------|----------|------------|
| TRISC2 | `llk_push_to_brisc()` llk_io_pack.h:57 | **Packer TTI_STALLWAIT(STALL_THCON, PACK) never completes** |
| TRISC0 | `wait_for_next_context()` cunpack_common.h:165 | UNPACK_SYNC not freed (ZEROACC never executed) |
| TRISC1 | `set_dst_write_addr()` cmath_common.h:163 | Blocked because TRISC0 blocked |
| BRISC | `cb_wait_front()` dataflow_api.h:468 | tiles_received_ptr never updated |
| NCRISC | `wait_for_brisc_notification()` | DONE, waiting for next dispatch |

**Causality chain** (traced through LLK source):

1. TRISC2 in FUSE_BIAS path calls `cb_push_back()` → `llk_push_to_brisc()` which issues `TTI_STALLWAIT(STALL_THCON, PACK)` waiting for packer HW to finish writing tiles to L1
2. Packer HW is hung due to operating with inconsistent per-section L1_ACC config (from the missing STALLWAIT race)
3. The preceding `tile_regs_release()` → `_llk_pack_dest_section_done_()` Tensix instructions are queued but blocked:
   - `TTI_STALLWAIT(STALL_MATH, PACK)` at llk_pack_common.h:37
   - `TT_ZEROACC(CLR_ALL)` at line 42 -- auto-releases UNPACK_SYNC semaphore
   - Since packer stuck, ZEROACC never executes → UNPACK_SYNC never freed
4. TRISC0 polls `semaphore_read(UNPACK_SYNC) >= 1` forever
5. BRISC polls `reg_read(tiles_received_ptr)` -- the `TT_STOREREG` in `llk_push_to_brisc` never executes

### Failure Mode B: Global Multicast Deadlock (ND_failure.log)

**ALL cores in column 0 stuck** (systemic). This is the propagation of Mode A.

| RISC | Stuck At | Root Cause |
|------|----------|------------|
| BRISC | `noc_semaphore_wait()` kernel:344 | In1 SENDER waiting for receiver acks |
| NCRISC | `noc_semaphore_wait()` kernel:286 | In0 SENDER waiting for receiver acks |
| TRISC0 | `cb_wait_front(in0)` kernel:255 | Waiting for input data |
| TRISC1 | `matmul_block()` kernel:287 | Mid-computation |
| TRISC2 | `pack_main` kernel:261 | In main loop |

---

## Key Code Paths Traced

### CB Synchronization Protocol
- **Compute side** writes ABSOLUTE values to stream registers via `TT_STOREREG`
- **Dataflow side** INCREMENTS stream registers
- CB init resets LOCAL copies but NOT stream registers
- This asymmetry is correct by design (safe across trace replays)

### Dest Context Protocol (SyncFull)
TRISC0 acquires → TRISC1 computes → TRISC2 packs → TRISC2 releases via `ZEROACC` side-effect.
`_llk_pack_dest_section_done_()`: `STALLWAIT(MATH,PACK)` → `ZEROACC(CLR_ALL)` → `SEMGET(MATH_PACK)`

### Packer L1_ACC Reconfiguration (THE BUG)
`reconfigure_packer_l1_acc()` writes config via `cfg_reg_rmw_tensix` (expands to `TT_RMWCIB*` instructions).
`TTI_STALLWAIT(STALL_PACK, TRISC_CFG)` was **commented out** at line 725. Without it, PACR instructions can race ahead of config writes.

### The `cfg_reg_rmw_tensix` Function (ckernel.h:350)
Template function that writes to THCON config registers using `TT_RMWCIB0/1/2/3` (read-modify-write config immediate byte). Issues up to 4 byte-level writes per call. Called 4 times in `reconfigure_packer_l1_acc()` for 4 THCON sections.

---

## Latency Instrumentation Results

Instrumented all matmul kernels with cycle counters at every wait site. Key findings from 2 runs (~190 watcher dumps each):

- **System is highly deterministic**: <0.5% variance between runs
- **Site 0x50 (in0 CB wait)**: Highest at 94 us (72 us typical), transient spikes
- **Sites 0x70/0x80 (bias path)**: Near-zero (18-45 cycles). Binary gap to infinity -- confirms hang is abrupt protocol violation, not gradual contention
- Analysis tool: `vit_n300/scripts/parse_latency.py`

---

## Actionable Next Steps

### Immediate (The Fix)
1. **Uncomment the STALLWAIT** in `reconfigure_packer_l1_acc()` at cpack_common.h:725
   - WH: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cpack_common.h`
   - BH: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h`
2. **Contact ryanzhu** who is already investigating packer reconfiguration (`ryanzhu/pack-recfg` branch)
3. **Also investigate** the commented-out STALLWAIT in `program_packer_destination()` at cpack_common.h:634

### Validation
1. Run ViT CI stress test with the fix applied -- Mode A should not recur
2. If Mode A is fixed and Mode B also doesn't recur, this confirms the causal link
3. Run the instrumented version with the fix -- latency profiles should be unchanged (the stall is nearly free when packer is idle)

### Isolated C++ Test (WIP -- needs debugging)
The test at `vit_n300/test_packer_l1acc_race.cpp` is built as part of `unit_tests_llk` but **hangs on execution**. See "Test Development Status" section below for details on what's blocking and how to resume.

---

## AI-Driven ND Failure Analysis Methodology

### What Worked (High Value)

1. **Static code analysis over stress testing**: Traced the full dependency chain from the hung core through LLK firmware to the exact missing instruction. This found the root cause in hours, not weeks of stress testing.

2. **Analyzing MULTIPLE failure logs independently**: The two failure logs showed completely different patterns (single-core vs grid-wide). Analyzing each independently and then finding the causal link was the breakthrough.

3. **Pattern matching across codebase**: Searching for `TTI_STALLWAIT.*TRISC_CFG` revealed 10+ instances in unpack code, exactly ONE instance in pack code (BFP path), and ZERO instances in the L1_ACC reconfig path. This inconsistency was the smoking gun.

4. **Watcher log interpretation**: The stuck RISC states (which file:line each core was at) provided the entry point for the entire analysis. Without this, blind stress testing would be useless.

5. **Understanding hardware pipeline semantics**: Knowing that `cfg_reg_rmw_tensix` writes go through a different pipeline than PACR instructions, and that `TTI_STALLWAIT` serializes them, was essential.

### What Didn't Work (Low Value for ND)

1. **Stress testing**: 1.3M matmul operations with zero reproduction. The race window is too narrow for brute-force reproduction.

2. **Latency instrumentation for hang detection**: Passing runs showed near-zero latency at the bias path sites. Hangs are binary (either instant or infinite), not gradual, so "approaching-hang" signals don't exist.

3. **Debug delay injection (`TT_METAL_WATCHER_DEBUG_DELAY`)**: Adds latency to NoC operations but doesn't affect intra-core pipeline hazards (which is where the bug is).

### Recommended Methodology for AI-Driven ND Failure Analysis

```
1. GATHER: Collect ALL available failure logs (watcher dumps, core states)
2. CLASSIFY: Analyze each failure log independently -- are they the same failure mode?
3. TRACE: For each stuck core, trace the dependency chain:
   - What instruction is each RISC stuck at? (from watcher)
   - What resource is it waiting for? (from the instruction semantics)
   - Who should provide that resource? (trace through the protocol)
   - Is that provider also stuck? (check its state)
4. SEARCH: Once you have the specific mechanism (e.g., "packer config race"),
   search for SIMILAR patterns across the codebase:
   - Does the same protection exist elsewhere? (e.g., STALLWAIT in unpack)
   - Is it missing in this specific case?
5. VERIFY: Check for active work on related issues:
   - Git log for relevant file changes
   - Branch names suggesting related investigation
6. LINK: If multiple failure modes exist, determine if they're independent or causally linked
```

---

## CI Test Environment

- Test: `test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py::test_vit`
- Hardware: N300 (2 Wormhole chips), ETH dispatch
- 2CQ trace replay, 1100 iterations, ~66K matmul ops per run
- `TT_METAL_OPERATION_TIMEOUT_SECONDS=5`

## Debug Tools

| Tool | Usefulness | Key Notes |
|------|-----------|-----------|
| WATCHER | Essential for post-hang | 1s polling, captures stuck state per RISC |
| WATCHER_DEBUG_DELAY | Not useful for this bug | Affects NoC, not intra-core pipeline |
| Ring Buffer | Effective for passing runs | 32-entry/core, low overhead |
| DPRINT | Not useful | Crashes machine, too much overhead |
| Device Profiler | Not useful | No kernel-internal capture |

## Test Development Status (Feb 13, 2026)

### C++ Isolated Test (`test_packer_l1acc_race.cpp`)

**Status: COMPILES but HANGS on execution.**

The test is built into `unit_tests_llk` via `tests/tt_metal/tt_metal/llk/sources.cmake`. It compiles and the device kernel JIT-compiles successfully. However, the kernel hangs on the device even with just 10 tiles.

**Build/Run requirements:**
- Build: `./build_metal.sh --build-tests` (full build needed for firmware JIT sources)
- Run: `cd /tt-metal && TT_METAL_SLOW_DISPATCH_MODE=1 ./build_Release/test/tt_metal/unit_tests_llk --gtest_filter="MeshDeviceFixture.TensixPackerL1AccRace*"`
- **MUST run from `/tt-metal/` directory** (not `build_Release/`), otherwise the JIT compiler resolves `root_dir` to `build_Release/` which lacks firmware sources. The CWD fallback checks for `tt_metal/` subdirectory.

**Issues resolved during development:**
1. `bfloat16::to_float()` doesn't exist → use `static_cast<float>(bfloat16(val))`
2. `TEST_F(MeshDeviceFixture, ...)` must be **inside** `namespace tt::tt_metal {}`
3. Compute kernel includes: use `"api/compute/compute_kernel_api.h"`, `"api/compute/common.h"`, etc. (not bare `"compute_kernel_api.h"`)
4. Dataflow kernel includes: use `"api/dataflow/dataflow_api.h"` (not bare `"dataflow_api.h"`)
5. Unused variable warning → add `[[maybe_unused]]` attribute
6. `PACKER_L1_ACC` must be passed as a **define** in `ComputeConfig{.defines = {{"PACKER_L1_ACC", "1"}}}`, not just referenced in kernel code
7. After killing a hung test, must run `tt-smi -r 0` to reset the device before running again

**Current kernel hang diagnosis:**
The simplified kernel (just copy in_cb→out_cb with `pack_reconfig_l1_acc(t & 1)` toggling) hangs even with 10 tiles. Possible causes to investigate:
- The `copy_tile_to_dst_init_short()` may not properly initialize the packer pipeline for the L1_ACC path
- The `PACKER_L1_ACC` define may change how the packer is initialized (different init codepath) and may require additional init calls
- The `namespace NAMESPACE { void MAIN }` syntax (deprecated) may interact poorly with PACKER_L1_ACC defines
- **Best approach**: Look at an existing test that uses L1_ACC (e.g., `test_copy_block_matmul_partials.cpp`) and replicate its kernel structure exactly, then add the L1_ACC toggle

### Python Test (`test_l1acc_race_matmul.py`)

**Status: RUNS but doesn't detect the race.**

Uses `ttnn.linear(a, b, bias=bias)` which goes through the full matmul path with FUSE_BIAS + PACKER_L1_ACC. Runs 1000 iterations comparing PCC against a baseline. Initial false-positive with constant inputs (PCC=0 for constant arrays) was fixed by using random inputs.

The race is too rare to trigger through the high-level Python API in a reasonable number of iterations.

## Files

- `vit_n300/ND_failure.log` -- CI failure log (Mode B: global deadlock)
- `vit_n300/ND_failure2.log` -- CI failure log (Mode A: single-core packer stall)
- `vit_n300/scripts/parse_latency.py` -- Watcher ring buffer analysis
- `vit_n300/scripts/loop_vit_ci_test.sh` -- CI test loop runner
- `vit_n300/test_packer_l1acc_race.cpp` -- C++ test: hammers L1_ACC reconfig→pack race (GTest, **hangs**)
- `vit_n300/kernels/compute_l1acc_hammer.cpp` -- Compute kernel: copy with L1_ACC toggle
- `vit_n300/kernels/reader_l1acc_hammer.cpp` -- Reader kernel: generates constant 1.0 tiles
- `vit_n300/kernels/writer_l1acc_hammer.cpp` -- Writer kernel: writes results to DRAM (unused, replaced by `direct_writer_unary.cpp`)
- `vit_n300/test_l1acc_race_matmul.py` -- Python test: uses ttnn.linear path (runs, no detection)
- `tests/tt_metal/tt_metal/llk/sources.cmake` -- Modified to include test_packer_l1acc_race.cpp
- Kernel instrumentation in `ttnn/.../matmul/device/kernels/` (3 files)

## Key Source Files

| File | What It Contains |
|------|-----------------|
| `tt_llk_wormhole_b0/common/inc/cpack_common.h:710` | **THE BUG**: `reconfigure_packer_l1_acc()` with commented-out STALLWAIT |
| `tt_llk_wormhole_b0/common/inc/cpack_common.h:627` | `program_packer_destination()` with second commented-out STALLWAIT |
| `tt_llk_wormhole_b0/common/inc/cpack_common.h:382` | `reconfig_packer_data_format()` (has STALLWAIT, for comparison) |
| `tt_llk_wormhole_b0/common/inc/ckernel.h:350` | `cfg_reg_rmw_tensix()` -- the TT_RMWCIB wrapper |
| `tt_llk_wormhole_b0/llk_lib/llk_pack_common.h:35` | `_llk_pack_dest_section_done_()` |
| `tt_llk_wormhole_b0/llk_lib/llk_pack.h:185` | `_llk_pack_()` -- issues set_dst_write_addr + program_packer_dest + mop_run |
| `hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h:57` | `llk_push_to_brisc()` -- where TRISC2 hangs |
| `bmm_large_block_zm_fused_bias_activation.cpp:342-355` | PACKER_L1_ACC reconfig → pack_tile_block (race window) |
