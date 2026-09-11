# Quasar Uplift Report — `ttnn.experimental.minimal_matmul`

**Date:** 2026-09-11
**Branch:** `vsureshTT/quasar_uplift_round_2` (uncommitted; delete this file before merge)
**Op directory:** `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/` (namespace `ttnn::experimental::prim` / `ttnn::prim`, unchanged)
**Base:** the Metal 2.0 port of PR #56245 (`iwrosz/minimal-matmul-metal2`), cherry-picked as `08893f1d81c..01c39499c58`
**Recipe:** `quasar_porting.md` (+ `quasar_audit.md`, `gen2_hardware_configs.md`, `sync_free_dfbs.md`, `dm_self_loop_dfbs.md`)

---

## Status: GREEN

The Metal 2.0 `MinimalMatmulDeviceOperation::ProgramFactory` builds, validates and runs correctly on Quasar
(craq-sim functional simulator, 8x4 grid, 2 DRAM banks) after the guarded fixes below: every shape with one
output block per core passes with PCC >= 0.99996 (with and without LLK/lightweight asserts), including
K-spill (up to 32 K-blocks of packer L1 accumulation), fp32-DEST accumulation, the transposed-grid writer
and the model's `mlp_w2_down` prefill site. Shapes with **two or more output blocks per core** stop on the
known **craq-sim false abort** on the 2nd push into an intra-tensix (PACK->UNPACK) self-loop DFB
(tenstorrent/craq-sim#355) — not an op defect, and per protocol not worked around. On the **ZEBU RTL emulator**
(1 worker core) all six single-core probes pass with PCC >= 0.99998, **including the 2-output-block shape that
craq-sim aborts on**, which pins #355 on the simulator. No RED-stop condition applies. WH/BH keep their original
path (parity argued structurally below).

---

## 1. Audit findings (recipe §1 step 1-2, `quasar_audit.md`, §4-§8, §11)

### 1.1 It is a Metal 2.0 op (step 1)
- Factory `device/minimal_matmul_program_descriptor.cpp`: `create_program_artifacts` -> `ProgramArtifacts`,
  `DataflowBufferSpec`/`DFBBinding`, `TensorParameter`/`TensorBinding` (+ a `TensorBindingSequence` for the N
  output chunks), `SemaphoreSpec`/`SemaphoreBinding`, named CTAs/RTAs, `WorkUnitSpec`s. No
  `override_runtime_arguments` (base spec concept).
- Kernels `device/kernels/{dm_in0_sender,dm_in1_sender_out,compute}_metal2.cpp` +
  `matmul_dataflow_common_metal2.hpp`: `api/dataflow/*`, `api/compute/*`, `Noc`, `DataflowBuffer`,
  `TensorAccessor(tensor::x)`, `make_tensor_accessors(tensor::outputs)`, `Semaphore s(sem::x)` (CTAD already),
  `get_arg(args::x)`, `get_tile_size(dfb::x)`. No `cb_*`, `get_arg_val`, `noc_async_*` free functions,
  `get_local_cb_interface`, `fifo_page_size`.
- **Out of scope / not Metal 2.0:** `device/minimal_matmul_program_factory.cpp` (`Program&` helper
  `minimal_matmul_factory_helper_common`, legacy `CreateKernel`/`create_cb`/`CreateSemaphore`) and the legacy
  kernels `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp`, plus the whole
  `minimal_matmul_fabric_bound_program_factory.cpp`. They are reached only by the fused CCL composites
  (`minimal_matmul_strided_reduce_scatter_async`, AG-fused matmul). They stay legacy and are **RED on Quasar**
  (legacy `DataMovementConfig` kernels; owner: CCL op owners). Not touched here.

### 1.2 Checklist vs. the code (applied / not needed)

| Recipe item | Finding | Action |
|---|---|---|
| `gen2_hardware_configs.md` compute (`hw_config`) | Shape 3 already converted by the port review commit: `to_compute_hardware_config(arch, ...)` (returns `ComputeGen2Config` on Quasar) + arch-agnostic accessors `double_buffer_dest(compute_hw) = true`, `unpack_modes(compute_hw)` for Float32 DFBs when `enable_32_bit_dest`. | none |
| `gen2_hardware_configs.md` DM (`hw_config`) | **Shape 4**: `DataMovementGen1Config{.processor, .noc}` hand-written in `make_dm_kernel` -> `ValidateProgramSpec` `"targets Gen2 (Quasar) but its DataMovementHardwareConfig holds a DataMovementGen1Config"`. | **applied**: `DataMovementGen2Config{}` when `device->arch()==QUASAR`; Gen1 initializer textually unchanged. `unpack_modes` marker N/A (DM has no such field). |
| `quasar_audit.md` check 2 — non-zero-init semaphores | `SEM_IN0_VALID` / `SEM_IN1_VALID` carry `initial_value = VALID (1)`. `program_spec.cpp:1755-1762` `TT_FATAL(init_value == 0, "... only zero is supported on Quasar")`. Both DM kernels execute `in*_valid_semaphore.set(VALID)` before the first `relay_unicast` that reads it, so the initial value is never observed. | **applied**: `initial_value = is_quasar ? 0 : initial_value` (WH/BH keep VALID). Owner note: the non-zero init is redundant on every arch and could be dropped (op-owner). |
| `quasar_audit.md` check 1 — DM self-loop / sync-free DFBs | `in0`, `in1`, `out`, `in2`, `ternary_*` are 2-party FIFOs (DM<->compute) with full credit use. `intermediate` is PRODUCER+CONSUMER on **compute** only (legal on Gen2). No sync-free DFB (every handle sees `reserve/push/wait/pop`). No `borrowed_from`. | none |
| §7 partials aliased onto the output (`partials_cb_uses_output`) | `intermediate` is its own DFB (`interm_cb_num_tiles = out_block_num_tiles`, own L1); output is a separate double-buffered DFB. | none |
| `Semaphore<>` explicit default template args | Kernels already use CTAD `Semaphore x(sem::y)`. | none |
| §11 multicast rectangle / +1 corner on 1-wide grids | No multicast: in0/in1 are forwarded core-to-core by `noc.async_write(..., UnicastEndpoint{...})` + `relay_unicast`. Direction comes from `preferred_noc_for_dram_*` via `build_core_order_for_axis`, but only picks the chain order; every hop is unicast. | none for mcast. The analogous degenerate-grid problem exists in the **work-unit regions** (below). |
| Degenerate grid (Quasar emu = 1 worker core, `functional_workers: [0-1]`) | `top_row`/`left_col`/`interior` regions are built from `CoreRange(core_1_0, core_endx_0)` etc. -> `CoreRange` `TT_FATAL` (start > end) on a 1-wide/1-tall grid; four **unused** `in*_cores` `CoreRange` locals would fatal the same way; receiver `KernelSpec`s with no `WorkUnitSpec` -> `"Kernel '{}' is not referenced by any WorkUnitSpec"`; `validate_on_program_cache_miss` demands a >= 2x2 config grid (comment said 1x1). | **applied**: regions and receiver kernels/run-args declared only when their axis has > 1 core; dead locals removed; validate accepts >= 1x1 on Quasar only. |
| §7 pack destination switch without re-init | Per output block the packer goes `intermediate` (matmul, L1-acc) -> `out` (`copy_and_pack_block` / `add_bias_block` / `swiglu_block`) -> `intermediate` ...; the ternary path also flips it inside `add_bias_and_addcmul_block`. On Quasar `llk_pack_init` programs/bakes the pack BFD and `llk_pack` writes through the last-baked one; `pack_reconfig_data_format` is gasket-only and none of `matmul_block_init`/`copy_init`/`add_bcast_rows_init`/`mul_*_init`/`add_init` has a PACK step on Quasar (checked in `api/compute/*.h` `#ifdef ARCH_QUASAR` branches). | **applied**: `retarget_packer(ocb)` = `#ifdef ARCH_QUASAR pack_init(ocb)` after every `pack_reconfig_data_format(...)` (8 sites). |
| §7 bare `wait_front->pop_front` (compute, 3 shapes) | `in0`/`in1`: `matmul_blocks` (>= 1 `matmul_block`) between; `intermediate`: `copy_tile`/`add_tiles_bcast`/`mul_tiles*`/`add_tiles` between; `in2`, `ternary_a/b`: real UNPACR between; guarded path `if (!reuse_in0_block) pop` only skips a pop. | none |
| §7 bare `reserve_back->push_back` (compute) | Two BARE pairs on `dfb_intermediate` in `add_bias_and_addcmul_block` ("restore/refill", `FUSE_TERNARY` only). | **applied**: `dummy_pack(intermediate_dfb)` between them — the sanctioned no-write PACR (`pack.h`); `llk_pack_dummy` is an empty function on WH/BH. Not on the Quasar test path (no ternary case) — compiled but unexercised. |
| §7 `compute_kernel_hw_startup` exactly once | Once at `kernel_main` start (`SrcOrder::Reverse`, in0/in1/intermediate). Exception: the `FUSE_TERNARY && TERNARY_B_IS_FLOAT32` branch re-calls it mid-kernel (pre-existing, tagged `TODO(#52395)`). | not on path; flagged |
| §7 re-`*_init` on every DFB-id change | Each block: `reconfig_data_format` + `matmul_block_init`; `copy_init` / `add_bcast_rows_init` / `mul_*_init` / `add_init` before each op family. | none |
| §5 `fifo_page_size` / `get_local_cb_interface` | none in the `_metal2` kernels (`get_tile_size(dfb::x)`, CTA tile sizes). | none |
| §7 formats (no Bfp8/Bfp4/UInt) | The op forwards tensor dtypes; the captured graph cases are `BFLOAT8_B` -> `ValidateProgramSpec` rejects the DFB format on Quasar (model-level dtype decision). Intermediate is `Float32` (fp32 acc) or `Float16_b`; `unpack_modes` entries are emitted for Float32. No UInt formats. | none (temp test copies use bf16) |
| §4 `disable_dfb_implicit_sync_*` | not set. `noc.async_write_zeros(dfb, ...)` (padding fill) passes the DFB straight to the NOC — implicit-sync credit interaction not exercised (no padded K/N in any case run). | watch item |
| §4 `opt_level` | compute explicit `O3` (legacy value); DM absent (-> O2 = legacy default). Base-port concern, unchanged. | none |
| §8.3 DM uncached DFB pointers | DM kernels use `get_write_ptr()/get_read_ptr()` + `CoreLocalMem` / `UnicastEndpoint{.addr}` as NOC operands; `Noc::get_src/dst_ptr` and `UnicastEndpoint` map the uncached alias back (`l1_cached_view`). Passes on the sim. | none |
| `#ifdef ARCH_BLACKHOLE noc.async_writes_flushed()` before `relay_unicast` (DM) | Considered for Quasar (single NOC). No symptom on 32-core sim runs (receivers' output PCC >= 0.99996). | not applied (reactive) |
| Compute-API Quasar gap found while building | `bcast.h::any_tiles_bcast` forwards the kernel's `MATH_FIDELITY` (HiFi2 here) into an ELWADD; Quasar `llk_math_binary_api.h:132` `static_assert(... == ELWMUL || fidelity == LoFi)` -> TRISC1 JIT failure. It fired on the **bias-less** path because `add_bias_block` is a plain function that is compiled regardless of `FUSE_BIAS`. | **applied**: `#ifdef FUSE_BIAS` around `add_bias_block` (codegen-neutral on WH/BH: the function was unused). A Quasar `FUSE_BIAS` build still fails there -> owner: compute API (`bcast.h`, forward LoFi for ADD/SUB like `add_tiles` does) / LLK. |

---

## 2. Files changed (all under the op directory)

| File | Change | Guard |
|---|---|---|
| `device/minimal_matmul_program_descriptor.cpp` | `is_quasar` flag; semaphores `initial_value = is_quasar ? 0 : VALID/INVALID`; DM `hw_config` = `DataMovementGen2Config{}` on Quasar (Gen1 initializer untouched); removed the four never-read `in0/in1_{sender,receiver}_cores` `CoreRange` locals; `has_in0_receivers`/`has_in1_receivers` gate the receiver `KernelSpec`s, their `KernelRunArgs` and the `top_row`/`left_col`/`interior` `WorkUnitSpec`s (regions only when the axis has > 1 core). | `is_quasar` for semaphores + hw_config. Grid gating is arch-neutral but a provable no-op on WH/BH (grid >= 2x2 there => all regions/kernels exist, identical spec). |
| `device/minimal_matmul_device_operation.cpp` | `validate_on_program_cache_miss`: config grid must be >= 2x2 on WH/BH (unchanged), >= 1x1 on Quasar. | `arch() == QUASAR` |
| `device/kernels/compute_metal2.cpp` | `retarget_packer()` helper (`#ifdef ARCH_QUASAR pack_init(ocb)`) after all 8 `pack_reconfig_data_format` sites; `dummy_pack(intermediate_dfb)` in the two bare `reserve_back->push_back` pairs (`FUSE_TERNARY`); `#ifdef FUSE_BIAS` around `add_bias_block`. | `ARCH_QUASAR` (pack_init); `dummy_pack` = empty on WH/BH; `FUSE_BIAS` guard is dead-code-only |
| `QUASAR_UPLIFT_REPORT.md` | this file | — |

Not changed: DM kernels, `matmul_dataflow_common_metal2.hpp`, legacy factory/kernels, nanobind, CMake, tests
(only **temporary** test copies, deleted after the run — see §4/§5). Nothing copied from
`experimental/quasar/`; no new namespace; directory unchanged.

---

## 3. Parity claim (WH/BH)

- Every Quasar-specific value is behind `device->arch() == tt::ARCH::QUASAR` (host) or `#ifdef ARCH_QUASAR`
  (kernel): semaphore init, DM Gen2 config, grid-size validation, `pack_init`.
- The grid gating in the factory is arch-neutral code, but on WH/BH the grid is >= 2x2 (validated when a
  config is given; every WH/BH device grid otherwise), so all four regions and all four DM kernels are
  declared exactly as before — the emitted `ProgramSpec`/`ProgramRunArgs` are identical.
- The four removed `CoreRange` locals were never read (dead code).
- `dummy_pack` -> `llk_pack_dummy` is `{}` on `wormhole_b0` and `blackhole`
  (`tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_pack_tile_api.h:23`).
- `#ifdef FUSE_BIAS` only removes an unused function from bias-less builds.
- No WH/BH device run in this session (none available). Confirm with:
  `pytest tests/ttnn/unit_tests/operations/experimental/test_minimal_matmul.py -v` (and the split / fused variants
  the port PR listed) on BH then WH.

---

## 4. craq-sim run

**Environment.** `source /localdev/vsuresh/qsr-sim/env.sh minimal_matmul` (craq-sim `libttsim.so`, `Arch.QUASAR`,
compute grid **8x4**, `dram_grid_size` 2 banks, slow dispatch, force-JIT, private kernel cache); host libs rebuilt
from this tree via `qsr_rebuild` (REBUILD_OK, `scratchpad/qsr_minimal_matmul/rebuild1.log`); all tests through
`qsr_test` (shared lock), foreground, one process at a time. Inputs bf16, host-tilized, DRAM interleaved (weights
DRAM width-sharded for the graph twin). Logs: `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_minimal_matmul/`
(`sim_*.log`), probe script `mm_probe.py` there.

Compute configs: **model** = `WormholeComputeKernelConfig(HiFi2, approx=False, fp32_dest_acc_en=False, packer_l1_acc=True)`
(the llama call sites); **default** = op default (HiFi2, fp32 acc, packer_l1_acc).

| # | Case | M x K x N | cfg | Output blocks/core | Result | PCC / detail |
|---|---|---|---|---|---|---|
| 1 | probe `S1` | 32x256x64 | model | 1 | **PASS** (also LLK+lightweight asserts) | 0.999988 |
| 2 | probe `S1_fp32acc` | 32x256x64 | default (Float32 intermediate + `unpack_modes`) | 1 | **PASS** (+asserts) | 0.999999 |
| 3 | probe `S2_kspill` | 32x512x64 | model | 1, 2 K-blocks (L1 acc) | **PASS** (+asserts) | 0.999987 |
| 4 | probe `S2b_kspill8` | 32x2048x64 | model | 1, 8 K-blocks | **PASS** (+asserts) | 0.999980 |
| 5 | probe `S4_transposed` | 128x256x32 | model | 1, M>N -> transposed grid, in1 is writer | **PASS** (+asserts) | 0.999988 |
| 6 | probe `S3_2blocks_sim` | 32x256x2560 | model | **2** (10 N tiles/core) | **SIM-ABORT (known craq-sim#355)** | `[16342] UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=128 exceeds capacity=64 (posted=128 acked=0)` = 2nd push into `intermediate` (capacity 64 = out block). Not an op defect; no change. -> RTL. |
| 7 | prototype `test_minimal_matmul[attn_qkv-512]` (temp copy, host tilize) | 512x2048x3072 | model | 2 (12 N tiles/core) | **SIM-ABORT (craq-sim#355)** | `[126815] ... occupancy=128 exceeds capacity=64 (posted=128 acked=0)` after the first output block. -> RTL. |
| 8 | prototype `test_minimal_matmul[mlp_w2_down-512]` (temp copy) | 512x8192x2048 | model | 1, **32 K-blocks** | **PASS** | 0.99996 (floor 0.98), 174 s |
| 9 | prototype `[*-1024]` (seq 1024) | 1024x{2048,8192}x{3072,2048} | model | attn_qkv: 2; mlp_w2: 1 | NOT RUN (sim budget) | attn_qkv-1024 would hit #355; mlp_w2-1024 = 2x row 8 |
| 10 | graph `01_1024x8192_bf8_int-dram` as **bf16 twin** (temp copy: bf8->bf16, grid 8x8->8x4, weight DRAM shard 12 banks->2, host tilize) | 1024x8192x2048, subblock 1x1 | default | 1, 32 K-blocks | **PASS** | harness golden `x @ w` PCC floor 0.98 (bf16 + K>=4096) met; shape/dtype/layout/memory-config/finiteness checks passed; 333 s (a first attempt hit `pytest-timeout`'s 300 s default -> rerun with `--timeout=565`) |
| 11 | graph `00_1024x2048_bf16_int-dram` (temp copy: grid/shard re-gridded) | 1024x2048x3072, subblock 1x1 | default | 2 (12 N tiles/core) | **SIM-ABORT (craq-sim#355)** | `[241051] UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=128 exceeds capacity=64 (posted=128 acked=0)` after the first output block (188 s of sim). -> RTL. |
| — | repo tests **as written** | — | — | — | FAIL before the op (harness) | `U.to_tt` / `G.build_tensor` = `from_torch(device=, layout=TILE)` -> legacy on-device tilize -> `kernel.hpp DataMovementKernel is not supported on Quasar` (bf16), and the captured bf8 weights -> `DFB ... Bfp8_b ... not supported on architecture QUASAR`. Same harness blocker as every sibling op on this branch. |

First run of the session (`sim_S1.log`) failed at the compute-kernel JIT (`llk_math_binary_api.h:132` LoFi assert)
-> fixed by the `FUSE_BIAS` guard (§1.2 last row); everything above is after that fix. No hang, no watcher/LLK
assert in any run; the only failures are the simulator's #355 abort.

Test-side workarounds were made only in temporary copies (`prototype_ops/test_tmp_minimal_matmul_qsr.py`,
`graph_ops/test_tmp_minimal_matmul_qsr.py`; archived in the scratch dir as `archived_test_tmp_minimal_matmul_*_qsr.py`,
deleted from the repo after the run): host tilize + `ttnn.to_device`, bf8 -> bf16, config grid -> device grid,
DRAM shard grid 12 banks -> `dram_grid_size()` banks. The repo tests themselves were not modified.

---

## 5. RTL emulator run

**Environment.** `source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3` -> `TT_METAL_SIMULATOR=.../emu-quasar-1x3` (ZEBU RTL,
`functional_workers: [0-1]` -> compute grid **1x1**, 1 DRAM bank), slow dispatch, private kernel cache; jobs via `qsr_emu`
(exclusive emulator lock, foreground, `timeout 2700`), one device session per job (`ttnn.open_device(0)`), same `mm_probe.py`
and host-tilized bf16 DRAM-interleaved inputs as on the sim. Boot ~1.5 min per job. Logs: `emu_S1.log`, `emu_job2.log`,
launcher logs `emu_launcher_S1.log`, `emu_launcher_job2.log` in the scratch dir. The 1x1 grid exercises the new
degenerate-grid path (only the `corner` work unit; no receiver kernels; sender == sink, no forwarding, prompt writes).

| # | Case | M x K x N | cfg | Notes | Result | PCC |
|---|---|---|---|---|---|---|
| 1 | probe `S1` (job 1) | 32x256x64 | model | 1 K-block, 1 output block | **PASS** | 0.999988 |
| 2 | probe `S2_kspill` (job 2) | 32x512x64 | model | 2 K-blocks: packer L1 accumulation into `intermediate` | **PASS** | 0.999987 |
| 3 | probe `S3_2blocks` (job 2) | 32x256x512 | model | **2 output blocks on one core = 2nd push into the intra-tensix `intermediate` self-loop** (the shape craq-sim#355 aborts on) + two pack-destination round trips | **PASS** | 0.999988 |
| 4 | probe `S2b_kspill8` (job 2) | 32x2048x64 | model | 8 K-blocks | **PASS** | 0.999980 |
| 5 | probe `S4_transposed` (job 2) | 128x256x32 | model | M > N: transposed grid, in1 kernel is the output writer | **PASS** | 0.999989 |
| 6 | probe `S1_fp32acc` (job 2) | 32x256x64 | default | fp32 DEST acc, Float32 intermediate + `unpack_modes` | **PASS** | 0.999999 |
| — | prototype / graph cases | 512..1024 rows, K 2048..8192 | — | not run: none fits one RTL core in the 45-min job budget (`mlp_w2_down-512` alone is ~8 G-MAC on a single core); their single-core geometry (K-spill, multi-block, both writers) is covered by rows 2-5, and the multi-core forwarding chains by the craq-sim rows | SKIP (budget) | — |

No hang, no assert, no launcher failure (`[ZTDB0349F]` never occurred). Row 3 is the RTL confirmation that the craq-sim
abort on the 2nd `intermediate` push (sim rows 6, 7, 11) is a simulator defect, not an op or runtime defect.


---

## 6. Open blockers / deferred items (symptom -> owner)

| # | Item | Symptom | Owner |
|---|---|---|---|
| B1 | craq-sim false abort on the 2nd push into an intra-tensix self-loop DFB (tenstorrent/craq-sim#355) | rows 6, 7 (and any shape with >= 2 output blocks per core: attn_qkv prefill, graph case 00) — `qsr_tile_counter_check_error ... posted=2*capacity acked=0`; same programs run on RTL | craq-sim team; no op change |
| B2 | Quasar `FUSE_BIAS` path does not build | `bcast.h::any_tiles_bcast` forwards `MATH_FIDELITY` into ELWADD; Quasar LLK `static_assert` "Math fidelity must be LoFi for non-ELWMUL ops" | compute API owners (`tt_metal/hw/inc/api/compute/bcast.h`) — pass LoFi for ADD/SUB as `eltwise_binary.h::add_tiles` already does; not an op edit |
| B3 | Repo tests cannot run on Quasar as written | `from_torch(device=..., layout=TILE)` runs the legacy on-device tilize; captured `BFLOAT8_B` weights | llama32_1b_quasar harness owners (`op_utils.to_tt`, `graph_case.build_tensor`: host tilize + `to_device` on Quasar); model owners (dtype) |
| B4 | Fused CCL composites (`minimal_matmul_strided_reduce_scatter_async`, AG-fused, fabric-bound factory) are legacy `Program&`/`CreateKernel` | RED on Quasar (legacy `DataMovementKernel`) | CCL op owners — needs its own Metal 2.0 port first |
| B5 | `TERNARY_B_IS_FLOAT32` path calls `compute_kernel_hw_startup` mid-kernel (`TODO(#52395)`) | recipe §7 forbids; not on the Quasar path | op owner |
| B6 | Non-zero-init `*_valid` semaphores on WH/BH are redundant (kernel sets VALID first) and slated for deprecation | none today | op owner — drop `VALID` init on all archs in a follow-up |
| W1 | `noc.async_write_zeros(dfb, ...)` padding fill passes the DFB to the NOC (implicit-sync path) while the kernel also does explicit `push_back` | not exercised (no padded K/N run); would show as a credit stall / double count on Quasar | watch; runtime team if it fires |
| W2 | DM `#ifdef ARCH_BLACKHOLE noc.async_writes_flushed()` before `relay_unicast` | none observed on 32 cores; if a receiver ever sees stale data on Quasar, extend the guard to `ARCH_QUASAR` | watch |

---

## 7. Repro commands

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh minimal_matmul
S=/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_minimal_matmul
# host rebuild after factory edits (exclusive lock; prints REBUILD_OK)
qsr_rebuild
# probes (one device session; PASS/FAIL + PCC per case)
qsr_test timeout 1700 ./python_env/bin/python $S/mm_probe.py S1 S1_fp32acc S2_kspill S2b_kspill8 S4_transposed
qsr_test timeout 1700 ./python_env/bin/python $S/mm_probe.py S3_2blocks_sim          # craq-sim#355 abort
TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 qsr_test timeout 1700 ./python_env/bin/python $S/mm_probe.py S1 S1_fp32acc S2_kspill S2b_kspill8 S4_transposed
# temp test copies (recreate from $S/archived_test_tmp_*.py next to the repo tests; delete afterwards)
export TT_DEVICE_LOCK_PATH=tt_device_minimal_matmul.lock TT_DEVICE_LOCK_TIMEOUT=1200
qsr_test timeout 570 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_minimal_matmul_qsr.py -k "mlp_w2_down and 512" -v -s
qsr_test timeout 570 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_minimal_matmul_qsr.py -k "attn_qkv and 512" -v -s   # craq-sim#355
qsr_test timeout 575 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_minimal_matmul_qsr.py -k "01_" -v -s
qsr_test timeout 575 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_minimal_matmul_qsr.py -k "00_" -v -s   # craq-sim#355
# debug env (prefix any command)
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 ...

# ZEBU RTL emulator (1x3 = 1 worker core; exclusive emu lock; never kill a job, let the timeout expire)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
qsr_emu timeout 2700 ./python_env/bin/python $S/mm_probe.py S1 S2_kspill S3_2blocks
```
