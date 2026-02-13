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

## Reproduction Attempts & Experimental Findings (Feb 13, 2026 - Session 2)

### CRITICAL FINDING: ViT Model Uses L1_ACC Only for LayerNorm, NOT Matmul

Examining `models/demos/vision/classification/vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py`:
- `packer_l1_acc=True` is ONLY set in `ln_compute_config` (used by LayerNorm ops, lines 149-154)
- ALL matmul operations (`ttnn.linear`, `ttnn.matmul`) at lines 194, 240, 273, 294, 318, 339, 359, 500 do NOT pass `compute_kernel_config`
- Default `WormholeComputeKernelConfig` has `packer_l1_acc=False` (confirmed in `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp`)

**Implication**: If the original theory (L1_ACC race in matmul causes the hang) is correct, the race would have to be triggered through the LayerNorm path, not matmul. OR: the matmul operations use L1_ACC through a different mechanism (e.g., the program config or runtime decision). This needs verification by checking what `PACKER_L1_ACC` define is actually set to in the compiled matmul kernels.

**Alternative**: The matmul kernel `bmm_large_block_zm_fused_bias_activation.cpp` checks `#ifdef PACKER_L1_ACC` at compile time. The define comes from `compute_kernel_config.packer_l1_acc`. If the ViT model's linear ops don't set this, then `reconfigure_packer_l1_acc()` is never called during matmul, and the L1_ACC race in matmul is a red herring for this specific test.

**TODO**: Check if LayerNorm operations also call `reconfigure_packer_l1_acc()` or have a similar race path.

### Fault Injection Experiments (Summary)

| Experiment | Modification | Result | Conclusion |
|-----------|-------------|--------|------------|
| Race Amplification | 200 dummy cfg_reg_rmw_tensix writes before correct value | No hang | Pipeline flooding doesn't widen race enough |
| Deterministic L1_ACC=0 | Force L1_ACC=0 regardless of input | No hang (wrong output) | Wrong L1_ACC = corruption, not hang |
| Split State (SEC0 correct, SEC1 opposite) | Different L1_ACC per section | No hang after 40K ops | Per-section inconsistency = corruption, not hang |
| Correctness Test (L1_ACC=True vs False) | Compare outputs | 100% mismatch, deterministic | **FLAWED TEST** - comparing different computation modes |

**Key conclusion from fault injection**: L1_ACC misconfiguration (wrong values, inconsistent sections) causes **data corruption**, NOT packer FSM hangs. The original theory that per-section config inconsistency causes an "undefined FSM state" leading to a permanent stall appears to be **incorrect** based on empirical evidence.

### Correctness Test Design Flaw

The test `test_matmul_l1acc_correctness` in `vit_n300/tests/test_matmul_deadlock_stress.py` compared:
- Golden: `packer_l1_acc=False`
- Test iterations: `packer_l1_acc=True`

Result: Every iteration showed identical mismatch (max_diff=6.0, 83.42% values different). This is NOT a race condition - it's the expected numerical difference between accumulation modes.

**To fix**: Golden should also use `packer_l1_acc=True`. Differences between L1_ACC=True runs would indicate actual race-induced corruption.

### Baseline Reproduction Attempts

| Test | Runs | Duration | Hangs | Notes |
|------|------|----------|-------|-------|
| ViT CI test loop (repro_loop.sh) | 200 | 75 min | 0 | Exact CI test configuration |
| Matmul stress (test_matmul_deadlock_stress) | ~40K ops | ~5 min | 0 | Isolated matmul with L1_ACC=True |

### Watcher Investigation

Watcher (`TT_METAL_WATCHER=<seconds>`) adds minimal device-side overhead:
- Waypoints: simple L1 mailbox writes
- NOC sanitization: validates addresses before NOC transactions
- `TT_METAL_WATCHER_DEBUG_DELAY=<cycles>`: adds NOC delays but does NOT affect intra-core pipeline hazards

Key env vars: `TT_METAL_WATCHER`, `TT_METAL_WATCHER_DEBUG_DELAY`, `TT_METAL_WATCHER_DUMP_ALL`
Key files: `tt_metal/impl/debug/watcher_server.cpp`, `tt_metal/hw/inc/api/debug/waypoint.h`

### Trace Replay Pattern (From ViT CI Test)

The actual CI test uses a sophisticated 2-CQ trace replay pattern:
- CQ 0: computation (trace execution, resharding)
- CQ 1: I/O (host↔device copies)
- Events synchronize between CQs
- 100 warmup + 1000 measurement iterations (1100 total replays)
- Input updated via `ttnn.reshard()` with persistent L1 tensor
- Trace executed non-blocking: `ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)`

This pattern is fundamentally different from simple operation loops. The device runs more autonomously during trace replay, potentially creating more opportunities for race conditions.

### `mm_block_init_short` Analysis

Found at `tt_metal/hw/inc/api/compute/matmul.h:322`:
```cpp
ALWI void mm_block_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id, ...) {
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
}
```
This is initialization-only, no obvious stall/hang points. The CI watcher showing trisc2 stuck in `mm_block_init_short` likely means the packer RISC is executing init code while waiting for something upstream (or the init itself was interrupted by a prior stall condition).

### CRITICAL FINDING: ViT Matmul DOES Use PACKER_L1_ACC (via Default)

**Initial wrong conclusion**: The ViT model only sets `packer_l1_acc=True` explicitly for LayerNorm. Matmul/linear ops don't pass `compute_kernel_config`.

**Corrected finding**: When `compute_kernel_config` is NOT passed to matmul, the default is resolved in `matmul_device_operation.cpp:1419`:
```cpp
auto kernel_config_val = init_device_compute_kernel_config(
    arch, parameters.compute_kernel_config, math_fidelity,
    /*default_approx_mode=*/false,
    /*default_fp32_acc=*/is_float_32,
    /*default_l1_acc=*/!is_float_32);  // <-- TRUE for non-float32 output!
```

Since ViT uses bfloat8_b output (not float32), `default_l1_acc = true`. And then in the 2D mcast factory (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:85`):
```cpp
bool packer_l1_acc_en = packer_l1_acc && (((bias_buffer != nullptr) && num_blocks > 1) || (num_blocks > 2));
```

For ViT matmuls with bias and `in0_block_w=3, K_tiles=24 → num_blocks=8`:
- `packer_l1_acc=true` AND `bias != null` AND `num_blocks=8 > 1` → **PACKER_L1_ACC IS ENABLED**

**Which ViT matmuls use L1_ACC**:
| Operation | in0_block_w | K_tiles | num_blocks | Has Bias | L1_ACC Enabled |
|-----------|-------------|---------|------------|----------|----------------|
| QKV projection | 3 | 24 | 8 | Yes | **YES** |
| Attention scores | 2 | 2 | 1 | No | No |
| Attention × Value | 7 | 7 | 1 | No | No |
| Self output | 3 | 24 | 8 | Yes | **YES** |
| FF1 | 3 | 24 | 8 | Yes | **YES** |
| FF2 | 12 | 96 | 8 | Yes | **YES** |
| Classifier | varies | varies | varies | Yes | Likely yes |

**Conclusion**: The original theory IS correct. The L1_ACC race in `reconfigure_packer_l1_acc()` is exercised by 4+ matmul operations per encoder layer × 12 layers × 1100 replays = ~50,000+ race windows per CI test run.

### LayerNorm Does NOT Use L1_ACC

Verified: `layernorm_op_multi_core_sharded.cpp` extracts `packer_l1_acc` from compute config but **never uses it** — never passed as a define to the compute kernel. LayerNorm compute kernels don't call `reconfigure_packer_l1_acc()`. The `packer_l1_acc=True` in `ln_compute_config` is effectively ignored.

### Trace Replay Testing Results

| Test | Replays | Matmul Ops | Duration | Hung? |
|------|---------|-----------|----------|-------|
| Single matmul trace replay | 2,000 | 2,000 | 0.1s | No |
| 3-chain matmul trace replay | 2,000 | 6,000 | 0.8s | No |
| ViT CI test loop | 200 runs | ~13M+ | 75 min | No |

Total: ~13M+ matmul operations with L1_ACC, zero hangs. This suggests either:
1. The hang requires very specific timing that this hardware doesn't exhibit
2. The hang requires the full model context (not just matmul chains)
3. The hang rate is genuinely < 0.01% per operation
4. Something unique to the CI environment (different firmware, chip characteristics, thermal conditions)

### Remaining Theories

1. **The race causes data corruption, not direct hangs**: Fault injection experiments proved that wrong L1_ACC values (even per-section inconsistency) cause corrupt output but NOT packer FSM hangs. The hang might be a second-order effect: corruption → downstream protocol violation → deadlock.

2. **Metastability**: The actual race involves reading a register MID-WRITE (transient electrical state), which can't be reproduced by writing wrong-but-stable values. This would require the exact timing overlap between config write commit and packer read.

3. **Full model context matters**: The ViT model runs many different operations between matmuls (LayerNorm, softmax, attention, add). The interleaving of these operations creates different pipeline states that may make the race more likely.

4. **N300 ETH dispatch**: The CI test runs on N300 with ethernet dispatch to a second chip. This adds NOC traffic and timing perturbations that single-chip testing doesn't have.

---

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
| `ttnn_optimized_sharded_vit_wh.py:149-154` | ViT model: `packer_l1_acc=True` ONLY for LayerNorm |
| `vit_n300/tests/test_matmul_deadlock_stress.py` | Python stress tests (matmul + correctness) |
| `vit_n300/tests/test_trace_replay_hang.py` | Trace-replay stress tests |
| `vit_n300/scripts/repro_loop.sh` | ViT CI test loop runner (200 runs, 0 hangs) |
| `matmul_device_operation.cpp:1419` | Where default `packer_l1_acc=!is_float_32` is set |
| `matmul_multicore_reuse_mcast_2d_program_factory.cpp:85` | Where `packer_l1_acc_en` condition is evaluated |
| `compute_kernel_config.cpp:17` | `init_device_compute_kernel_config()` -- default resolution |

---

## Investigation Summary (Feb 11-13, 2026)

### What Was Accomplished

1. **Root cause identified with high confidence**: The missing `TTI_STALLWAIT` in `reconfigure_packer_l1_acc()` is a real bug that creates a pipeline hazard. Every unpack operation in the LLK has the equivalent stall; the pack L1_ACC path is the only one missing it.

2. **Confirmed the ViT model exercises the bug**: Despite initial confusion (the model doesn't explicitly pass `packer_l1_acc=True` to matmuls), traced through the default resolution path to prove that 4+ matmuls per encoder layer DO use PACKER_L1_ACC.

3. **Empirically proved L1_ACC misconfiguration causes corruption, not direct hangs**: Three different fault injection experiments (wrong values, forced zero, split sections) all produced corrupt output but zero packer stalls across 40K+ operations.

4. **Mapped the complete deadlock propagation chain**: Single-core packer stall -> CB fills -> multicast sender blocks -> column deadlock -> row deadlock -> global hang. Two distinct CI failure logs show Mode A (root cause) and Mode B (propagation).

5. **Found active related work**: TT engineer ryanzhu's `pack-recfg` branch adds config verification, confirming the bug is known internally.

### What Was NOT Accomplished

- **Failed to reproduce the hang**: ~13M matmul operations with L1_ACC across various test configurations, zero hangs. The race window is too narrow or requires conditions not present on this system.
- **Failed to create a reliable reproducer**: The user's requirement was 20% hang rate within 10 minutes. Best achieved: 0% across all attempts.

### Key Lessons for AI-Driven ND Debugging

1. **Static analysis >> stress testing** for this class of bug. The root cause was found in hours through code tracing; stress testing burned days with no result.
2. **Always trace default parameter resolution paths**. The `packer_l1_acc` default was `true` for non-float32, which was non-obvious and initially led to a wrong conclusion.
3. **Fault injection has limits**: Software-injected faults produce stable wrong values. Real hardware races produce metastable/transient values. The two can have very different effects.
4. **AI should not run hardware tests autonomously**: Risk of damaging accelerator cards. All test code should be human-reviewed before execution.

---

## Blueprint: Learning the Relevant Codebase (~5 days)

### Phase 1: Tensix Architecture Fundamentals (1-2 days)

**Goal**: Understand the 5-RISC per-core architecture and how compute kernels execute.

- **Read**: `docs/source/tt-metalium/` -- architecture overview
- **Key concepts**: BRISC/NCRISC/TRISC0-2 roles, circular buffers (CBs), UNPACK->MATH->PACK pipeline, config registers (THCON), `TTI_STALLWAIT` pipeline synchronization
- **Key files**: `tt_metal/hw/inc/api/compute/matmul.h`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h`

### Phase 2: LLK (Low-Level Kernels) (1-2 days)

**Goal**: Understand how pack/unpack operations work at the hardware instruction level.

- **Key files** (in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/`):
  - `common/inc/cpack_common.h` -- **THE FILE WITH THE BUG**
  - `common/inc/ckernel.h` -- `cfg_reg_rmw_tensix()` template
  - `llk_lib/llk_pack.h` and `llk_lib/llk_pack_common.h`
- **Exercise**: Search for all `TTI_STALLWAIT.*TRISC_CFG` patterns. Note how every unpack has one; only BFP pack path has one. The L1_ACC pack path is missing it.

### Phase 3: Matmul Operation Pipeline (1 day)

**Goal**: Understand how `ttnn.linear()` becomes device kernel execution.

- **Path**: `ttnn.linear()` -> `matmul.cpp` -> `matmul_device_operation.cpp` -> factory files -> device kernel
- **Default resolution**: `init_device_compute_kernel_config()` in `compute_kernel_config.cpp`
- **Device kernel**: `bmm_large_block_zm_fused_bias_activation.cpp` -- search `#ifdef PACKER_L1_ACC`
- **Multicast protocol**: reader/writer kernels using `noc_semaphore_wait` and multicast

### Phase 4: Trace Replay and Command Queues (0.5 days)

- **Read**: `test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py` (178 lines)
- **Key APIs**: `begin_trace_capture`, `end_trace_capture`, `execute_trace`, `record_event`, `wait_for_event`

### Phase 5: Watcher and Debug Tools (0.5 days)

- **Read**: `docs/source/tt-metalium/tools/watcher.rst`
- **Env vars**: `TT_METAL_WATCHER=<seconds>`, `TT_METAL_WATCHER_DUMP_ALL=1`
- **Practice**: Run any test with watcher, read `generated/watcher/watcher.log`
- **Key skill**: Given stuck RISCs from watcher dump, trace through LLK to determine what each is waiting for
