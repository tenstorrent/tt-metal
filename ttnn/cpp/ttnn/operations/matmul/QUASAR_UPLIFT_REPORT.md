# Quasar Uplift Report — `ttnn.linear` (matmul op family), DRAM-sharded factory

**Date:** 2026-09-11 (supersedes the 2026-09-10 RED report, written while this factory was still legacy)
**Branch:** `vsureshTT/quasar_uplift_round_2` (uncommitted; delete this file before merge)
**Op directory:** `ttnn/cpp/ttnn/operations/matmul/`
**Uplifted factory:** `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`
(`device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`, Metal 2.0 since #55961,
cherry-picked here as `01a11f6091f`) and the three kernels it binds.
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py` (6 of 8 cases route here).
**Recipe followed:** `quasar_porting.md` §1 workflow (step 1 confirm M2 → §2/§7/§8/§11 audit → in-place, guarded
uplift → craq-sim → RTL) plus the canonical passes it points at (`gen2_hardware_configs.md` shape 4,
`sync_free_dfbs.md` borrowed → `LocalTensorAccessor`, `quasar_audit.md` check 2).

---

## Status per factory

| Factory | Cases | Status | One-line state |
|---|---|---|---|
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` | 00, 01, 02, 03, 04, 07 (1872/1968 captured calls) | **GREEN** (uplifted in place) | Builds, validates and runs on Quasar. craq-sim: single-K-block and 2-K-block DRAM-sharded linear **PASS, PCC 0.999997**; every ≥3-K-block shape (all six captured cases) stops at the known craq-sim false abort tenstorrent/craq-sim#355 on the 2nd partials push. ZEBU RTL (1 core, 1 bank): 1-K-block **PASS PCC 0.999997**, 4-K-block K-spill **PASS PCC 0.999995**. The captured shapes themselves cannot be run as captured on either target (12 DRAM banks / 32–64-core grids). |
| `MatmulMultiCoreReuseMcast2DProgramFactory` | 05, 06 (96 calls) | **RED — not Metal 2.0** | Still `create_descriptor` → `ProgramDescriptor` + legacy device API; not touched (recipe RED-stop "Not Metal 2.0 on Gen1 yet"). Cases 05/06 are SKIP until the 2D-mcast Metal 2.0 port lands. |

---

## Audit findings (new Metal 2.0 code re-checked against the 2026-09-10 heads-ups + recipe §4/§6/§7/§11)

| # | Check | Finding in the Metal 2.0 factory / kernels | Disposition |
|---|---|---|---|
| A1 | `quasar_audit.md` check 2 — non-zero-init semaphore | Still present: `IN0_MCAST_SENDER_VALID_SEM` with `.initial_value = VALID` (`= 1`). **No kernel binds it** — the in0 sender publishes VALID by multicasting its own `in0_mcast_receiver` L1 word (`receiver_sem.set(VALID)` + `set_multicast`). Dead state carried over from the legacy factory. `ValidateProgramSpec` rejects it on Quasar (`program_spec.cpp:1756`). | Omitted on Quasar (`if (!is_quasar)`); WH/BH keep it so their semaphore layout is untouched. **Follow-up for the op owner: delete it outright** (see B4). |
| A2 | §11 — NOC_1 multicast start/end swap | Still present (`if (in0_noc == NOC_1) std::swap(start, end)`); `preferred_noc_for_dram_write()` returns NOC_1 by default (Quasar included). | Swap made Gen1-only. |
| A3 | §11 — `+1` mcast corner / degenerate-grid clamp | **Not applicable to this factory**: it multicasts the whole compute grid `top_left_physical → bottom_right_physical`; there is no `+1` corner (2D-mcast factory only). | None. |
| A4 | §7 — bare `wait_front`→`pop_front` (TEN-4746) in the compute fork | Still present in `bmm_large_block_zm_fused_bias_activation_metal2.cpp`: two `for (s …) { wait_front; pop_front; }` drain loops under `PACKER_L1_ACC` (BARE shape), plus a GUARDED-PATH shape: under `SKIP_COMPUTE` nothing unpacks in0/in1 between their wait and pop. All other pairs have a real UNPACR/PACR between (reload `copy_block`, untilize `copy_tile`, bias `add_tiles*`, matmul `matmul_block`; every `reserve_back`→`push_back` has `pack_tile`/`pack_block`). | `dummy_unpack(<dfb>)` inserted under `ARCH_QUASAR` at the 3 sites (#54948 `dummy_unpack` is in `api/compute/tile_move_copy.h`). |
| A5 | `cb_dfb_quasar_audit_helper.md` GATE — raw `fifo_rd_ptr` write | The port replaced it with `mm_partials_reload_dfb.evil_set_read_ptr(...)` under `#ifdef MM_PARTIALS_RELOAD_ALIAS` (whitelisted on Gen1). `evil_set_*` is `#ifndef ARCH_QUASAR` in the DFB API; **this factory never defines `MM_PARTIALS_RELOAD_ALIAS`** (FUSE_BIAS+FP32 reload alias of other factories), so it is compiled out here. | Not on path. Flagged as a §7 missing-feature (DFB rewind API on Gen2) for whichever factory needs it (B5). |
| A6 | Pack destination baked at `pack_init` on Quasar (lesson from `experimental/quasar/matmul` run 2026-09-10; confirmed in `hw/ckernels/quasar/.../llk_pack_tile_api.h`: `llk_pack_init` programs the BFD from `output_id`, `llk_pack`/`llk_pack_block` only compute the tile index; `llk_pack_reconfig_data_format` re-programs formats only) | Kernel switches the pack output between `intermed0` (partials) and `out` with only `pack_reconfig_data_format` (and only under FP32/L1ACC defines). | `pack_init(<dfb>)` under `ARCH_QUASAR` at every pack-destination switch: block-loop start (back to partials), in0-transpose stage (both directions), the last-K-block switch to `mm_out_dfb` (once per block, before the subblock loops), the FUSE_BIAS epilogue switch to the output. The untilize epilogue already goes through `pack_untilize_dest_init(out)` (a Quasar init). |
| A7 | §7 — `partials_cb_uses_output` / borrow-with-offset | `out` and `intermed0` are `alias_with` each other when formats match (same base, **same size**, output is a single block per core: `num_blocks_w/h_dim = 1`, `batch = 1`). No `address_offset`, so the §7 clobber rule does not apply. | Kept (WH/BH behaviour). Result correct on craq-sim with aliasing on (PCC 0.999997) once A6 was applied. Note the 2026-09-10 quasar-copy PCC-0.0 was A6 masked/unmasked by aliasing, not aliasing itself. |
| A8 | §6 — DM self-loop DFBs (Gen2 validator `program_spec.cpp:1486` rejects) | **New**: `in0_sharded` (borrowed from `in0`, bound PRODUCER+CONSUMER by the in0 sender) and `out_reshard` (borrowed from `output`, bound PRODUCER+CONSUMER by the in1 sender/writer). Both kernels only take the base address (`get_read_ptr()` / `get_write_ptr()`), no credit call on either (all 6 methods grepped, nothing opaque takes the handle). Sync-free, borrowed → `LocalTensorAccessor` per `sync_free_dfbs.md`. | Converted (host: 2 DFB specs + 4 bindings removed, 2 `TensorBinding`s added; kernels: `LocalTensorAccessor<uint32_t>(tensor::in0_shard / tensor::out_shard).get_bank_base_address()`). Unguarded by design (canonical Gen1 style pass; same L1 addresses). |
| A9 | `gen2_hardware_configs.md` shape 3/4 | Compute: helper-built (`to_compute_hardware_config`, returns Gen2 on Quasar) with `unpack_modes(compute_hw)` accessor — shape 3 already done by the port. DM: two hand-written `DataMovementGen1Config{processor, noc}` (shape 4) → validator `program_spec.cpp:884` FATALs on Quasar. | Hoisted to `DataMovementHardwareConfig` locals, `DataMovementGen2Config{}` on Quasar (implicit sync left default). No `unpack_modes` marker needed (DM only). |
| A10 | Host: DRAM bank → reader-core placement | **New**: `get_dram_bank_reader_assignments()` → `IDevice::get_optimal_dram_bank_to_logical_worker_assignment()` → `tt_metal/common/core_assignment.cpp:279 TT_THROW("Invalid Arch Name specified")` (no QUASAR arm). Same failure the legacy path hit on 2026-09-10 (G02/G03). Runtime file — not edited. | Interim arch-checked fallback in the factory: bank `b` → `b`-th compute core in row-major logical order (`workers_per_bank` is already pinned to 1 off Blackhole). Blocker B2 for the runtime. |
| A11 | Compute fork compiles on Quasar? | **New**: `ckernel::ThreadId::BriscThreadId` (sparse-matmul `get_batch_from_reader` mailbox read) does not exist in Quasar's `ckernel_defs.h`; `kernel_main` is not a template so the discarded `if constexpr` branch is still compiled. | Mailbox reads `#ifndef ARCH_QUASAR`; `static_assert(!get_batch_from_reader)` on Quasar (this factory passes 0). |
| A12 | Narrowed last in1 subblock (`last_subblock_padded`, DRAM-sharded planner pads `per_core_N_compute` to widen `out_subblock_w`) | **New, found on craq-sim (case 07: 84 tiles/worker → padded to 88, `ct_dim` 8 with a 4-lane last subblock): HANG** (900 s timeout, 40 M cycles at idle speed). Quasar's matmul LLK bakes `ct_dim`/`rt_dim` into the unpack and math MOPs at `matmul_block_init` (`_llk_math_matmul_init_`, `_llk_unpack_matmul_init_`); a per-call `ct_dim` smaller than the init's desynchronizes the src-register handshake. | Host: the widening loop is skipped on Quasar (keeps the divisor-based `out_subblock_w`, e.g. 7 for 84; no padded lanes). Kernel: `static_assert(!last_subblock_padded)` under `ARCH_QUASAR`. Post-fix case 07 runs past the old hang point. LLK FYI (B6). |
| A13 | §4 `unpack_modes` for FP32 DFBs | `intermed0` (Float32 when `fp32_dest_acc_en`) carries `UnpackToSrc`, in0/in1/bias too. | Nothing to do. |
| A14 | §4 `opt_level` | Compute `O3` explicit, DM absent (`O2`) — matches legacy. Base-port concern, untouched. | None. |
| A15 | §5 `fifo_page_size` / `get_local_cb_interface` | None in the three kernels. | None. |
| A16 | §7 `compute_kernel_hw_startup` once / re-init on DFB-id change | Once at start. Reload path does `copy_init(partials)` then `matmul_block_init(in0,in1)` before the next `matmul_block`; bias/untilize epilogues re-init. | None. |
| A17 | §7 implicit sync | DM kernels use explicit `reserve/push/wait/pop`; `disable_dfb_implicit_sync_*` **not** set. | None. |
| A18 | §7 Quasar formats | Every captured case has BFLOAT8_B/BFLOAT4_B weights (two also bf8 outputs); Quasar has no Bfp8/Bfp4 DFB format. `in0_last_ktile_w` padding (`pad_last_ktile`, `pad_tile.hpp` `#ifndef ARCH_QUASAR` for BFP mantissas) is compiled out for these shapes (K multiple of 32). | Test-level bf16 substitution; model-level decision (B7). |
| A19 | §6 local self-copy on the emulator | On a single-node grid this op performs two NoC transfers whose source and destination are the same L1: the storage+worker core multicasts in0 to itself (`MCAST_INCL_SRC`, `num_dests = 1`) and reshards its output block to itself (unicast `async_write` to its own coordinates). Fine on craq-sim (flat memory); **on the 1-core ZEBU config the unmodified kernels hung** (first RTL job, see RTL section). Multicast/unicast endpoint traits do map the uncached DFB alias back (`l1_cached_view`), so the address is not the issue. | Applied reactively, `ARCH_QUASAR`: `local_l1_copy()` (RISC word copy through the uncached L1 alias, mirroring `data_movement/common/kernels/common.hpp`'s source-coherency idiom) replaces the multicast when `in0_mcast_num_cores == 1` (compile-time) and the reshard `async_write` when `noc.is_local_bank(x, y)`. craq-sim re-run PASS (both workers' write-backs there are self-targeted, so the copy path is numerically verified). |

---

## Files changed (4, all under `ttnn/cpp/ttnn/operations/matmul/`; directory and namespace untouched)

| File | Change | Why | Guard |
|---|---|---|---|
| `device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` | `is_quasar` local; multicast rectangle swap made Gen1-only | §11 ascending `[min..max]` on Quasar (A2) | `&& !is_quasar` |
| same | Quasar DRAM-bank → reader-core placement fallback (bank `b` → `b`-th compute core, row-major) instead of `get_dram_bank_reader_assignments()` | `core_assignment.cpp` has no QUASAR arm (A10) | `if (is_quasar)` |
| same | Subblock-widening loop skipped on Quasar | narrowed `ct_dim` unsupported by the Quasar matmul LLK (A12) | `and !is_quasar` |
| same | `in0_sharded` / `out_reshard` borrowed DFB specs + 4 self-loop bindings removed; `TensorBinding{IN0, "in0_shard"}` on the in0 sender, `TensorBinding{OUTPUT, "out_shard"}` on the in1 sender/writer; `dataflow_buffers.reserve` 7/6 → 5/4; unused `in2_block_tiles` / `out_reshard_*` sizing removed | DM self-loop rejected on Gen2; sync-free borrowed views are `LocalTensorAccessor` by the canonical pass (A8) | none (behaviour-preserving; same L1 addresses) |
| same | `IN0_MCAST_SENDER_VALID_SEM` (VALID-init) not created on Quasar | dead semaphore, non-zero init rejected on Quasar (A1) | `if (!is_quasar)` |
| same | DM `hw_config`s hoisted to `DataMovementHardwareConfig` locals, `DataMovementGen2Config{}` on Quasar; `using` for `DataMovementGen2Config`/`DataMovementHardwareConfig` | validator requires Gen2 DM config on Quasar (A9) | `if (is_quasar)` |
| `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` | `pack_init(<dfb>)` after every pack-destination switch (block start → partials; transpose stage → in0 and back; last K block → `mm_out_dfb`; FUSE_BIAS epilogue → output) | Quasar bakes the pack BFD at init (A6) | `#ifdef ARCH_QUASAR` |
| same | `dummy_unpack(mm_partials_dfb_id)` in both bare wait→pop drain loops; `dummy_unpack(in0/in1)` before their pops under `SKIP_COMPUTE` | TEN-4746 (A4) | `#ifdef ARCH_QUASAR` (`&& SKIP_COMPUTE` for the third) |
| same | BRISC mailbox reads `#ifndef ARCH_QUASAR`; `static_assert(!get_batch_from_reader)` on Quasar | no `ThreadId::BriscThreadId` on Quasar (A11) | `ARCH_QUASAR` |
| same | `static_assert(!last_subblock_padded)` on Quasar | turn the A12 hang into a build error for any future factory | `#ifdef ARCH_QUASAR` |
| `device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | `DataflowBuffer(dfb::in0_sharded).get_read_ptr()` → `LocalTensorAccessor<uint32_t>(tensor::in0_shard).get_bank_base_address()`; include `api/tensor/local_tensor_accessor.h` | A8 | none |
| same | `local_l1_copy()` helper; the sender+compute core's `MCAST_INCL_SRC` multicast becomes a RISC L1→L1 copy when `in0_mcast_num_cores == 1` | single-node self-multicast hangs the emulator (A19) | `#ifdef ARCH_QUASAR` + `if constexpr` |
| `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | `DataflowBuffer(dfb::out_reshard).get_write_ptr()` → `LocalTensorAccessor<uint32_t>(tensor::out_shard).get_bank_base_address()`; include | A8 | none |
| same | `local_l1_copy()` helper; reshard `async_write` to a storage core that is this core (`noc.is_local_bank`) becomes a RISC L1→L1 copy | self-unicast hangs the emulator (A19) | `#ifdef ARCH_QUASAR` |

No `_metal2` fork was needed (the compute kernel already is the port's fork; both DM kernels are bound only by this
factory). Nothing copied into `experimental/quasar/`, no `::qsr` namespace, no runtime/LLK/CMake/doc/test edit.
The legacy `bmm_large_block_zm_fused_bias_activation.cpp` (bound by the six unported factories) is untouched.

---

## Gotchas (§7–§8, §11) — applied vs considered

**Applied (each fired or is a validator gate on Quasar):** A1 (semaphore), A2 (mcast rectangle), A4 (TEN-4746
bare pairs), A6 (pack_init at pack-destination switch), A8 (DM self-loop → LocalTensorAccessor), A9 (Gen2 DM
config), A10 (core_assignment fallback), A11 (BriscThreadId), A12 (narrowed subblock hang).

**Considered, not needed (and why):** A3 (no +1 corner in this factory); A5 (`evil_set_read_ptr` compiled out —
`MM_PARTIALS_RELOAD_ALIAS` never defined here); A7 (`alias_with` is same-base/same-size on a single-block
output — the §7 offset-clobber rule needs an offset borrow, and the sim result is correct with aliasing on);
`disable_dfb_implicit_sync_*` (never set — explicit credit pattern in both DM kernels); `fifo_page_size`
(absent); `unpack_modes` for FP32 (present); `opt_level` (base-port, correct); `MEM_ZEROS_BASE`,
`flush_l2_cache_range`, `common.hpp`-in-DM, `REDUCE_OP`, `sfpu_reduce` (not referenced); A19 self-mcast on a 1-core
grid (reactive — see RTL section).

---

## craq-sim run (Quasar functional simulator, 8x4 worker grid, 2 DRAM banks, ~1 kHz; slow dispatch, force-JIT)

Host libs rebuilt from this tree via `qsr_rebuild` (REBUILD_OK 17:42 and again 18:08 after A12). Probe script:
`<scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py` (M=32, in0 L1 width-sharded on core (0,0), weights
DRAM width-sharded across both banks, `per_core_N=2`, bf16, host tilize + `ttnn.to_device`). Graph cases: temp copy
`test_tmp_linear_ds_qsr.py` next to `graph_ops/test_linear.py` (deleted after the run; archived in scratch) that
programmatically re-grids the captured cases — the exact changes are in the table.

| Case | Shape / config | Re-gridded for the sim | Result | Detail |
|---|---|---|---|---|
| probe `tiny` | K=64, N=128, `in0_block_w=2` → 1 K-block (no partials) | — | **PASS** | PCC **0.999997**, max abs err 0.117; output width-sharded L1 `[32,64]` on 2 cores as derived |
| probe `kspill2` | K=64, N=128, `in0_block_w=1` → 2 K-blocks (one partials push + reload) | — | **PASS** | PCC **0.999997** |
| probe `tiny` (re-run after A19) | as above; both workers reshard to themselves → exercises `local_l1_copy` | — | **PASS** | PCC **0.999997** (RISC copy path numerically verified; 32-core multicast stays on the NoC) |
| probe `kspill` | K=128, N=128, `in0_block_w=1` → 4 K-blocks | — | **FAIL (sim abort)** | `qsr_tile_counter_check_error: tile counter occupancy=4 exceeds capacity=2 (posted=4 acked=0)` on the 2nd partials push — exact craq-sim#355 signature (capacity = `out_block_tiles`, posted = 2×capacity, acked 0). Not an op change; → RTL |
| G00 `00_32x2048_bf16_ws-l1` (2048x8192, bf4 w) | `in0_block_w=1`, 64 K-blocks | bf4→bf16; in1 DRAM grid 12→2 banks `[2048,4096]`; in0 grid 8x8→8x4 shard `[32,64]`; `per_core_N` 4→8 (32 out cores); out spec rewritten | **FAIL (sim abort)** | `occupancy=136 exceeds capacity=128 (posted=136 acked=0)` — #355 (capacity = 128-tile worker out block, +8 = 2nd block's first subblock) |
| G01 `01_…` (2048x8192, bf8 w, bf8 out) | `in0_block_w=2`, 32 K-blocks | bf8→bf16 (weights + output dtype); in1 12→2 banks `[2048,4096]` | **FAIL (sim abort)** | `occupancy=136 exceeds capacity=128` — #355 |
| G02 `02_…` (2048x3072, bf8 w) | `in0_block_w=2`, 32 K-blocks | bf8→bf16; in1 12→2 banks `[2048,1536]` | **FAIL (sim abort)** | `occupancy=56 exceeds capacity=48` — #355 |
| G03 `03_…` (2048x2048, bf8 w) | `in0_block_w=2`, 32 K-blocks | bf8→bf16; in1 12→2 banks `[2048,1024]` | **FAIL (sim abort)** | `occupancy=40 exceeds capacity=32` — #355 |
| G04 `04_32x8192_bf8_ws-l1` (8192x2048, bf8 in0+w) | `in0_block_w=4`, 64 K-blocks | bf8→bf16 (in0, weights); in1 12→2 banks `[8192,1024]`; in0 grid 8x8→8x4 shard `[32,256]`; `per_core_N` 1→2 | **FAIL (sim abort)** | `occupancy=40 exceeds capacity=32` — #355 |
| G07 `07_…` (2048x5376, bf8 w, bf8 out) | `in0_block_w=2`, 32 K-blocks; 84 tiles/worker | bf8→bf16 (weights + output dtype); in1 12→2 banks `[2048,2688]` | **pre-A12: HANG** (900 s timeout, 40 M cycles at ~45 kHz idle speed); **post-A12: FAIL (sim abort)** | post-fix `occupancy=91 exceeds capacity=84` — #355 (+7 = one 7-wide subblock, confirming no padded lanes) |
| G05, G06 | 2D-mcast (`MatmulMultiCoreReuseMultiCastProgramConfig`) | — | **SKIP** | legacy factory (RED, not ported); not run |

Reading: the two probes prove in0 mcast (2 senders → 32-core grid, ascending rectangle), the DRAM-sharded in1
reader (per-bank `AllocatorBank` reads with trid barriers), compute with one partials round-trip, the pack
re-target to `out`, and the reshard write-back all work on Quasar. Every captured case necessarily has ≥32 K-blocks
(in0 is width-sharded over ≥32 storage cores and each shard is one block), so **no captured case can be validated
numerically on craq-sim until craq-sim#355 is fixed**; the RTL emulator is the validation target for K-spill.

---

## RTL emulator run (ZEBU emu-quasar-1x3: 1 worker core (0,0), 1 DRAM bank, 4 MB L1)

Same probe script; with one bank/one core it becomes M=32, N=64 (2 tiles), weights DRAM width-sharded on the single
bank `[K,64]`, in0 on core (0,0), `per_core_N=2` → the single core is in0 storage core, DRAM reader/worker and output
storage core at once (`worker_core_type = 2`, `in0_mcast_num_cores = 1`). One device session per job; launcher logs
`emu_launcher_job*.log` and watcher dumps `watcher_emu_job*.log` are in the scratch dir. Boot ≈ 1.5 min per job.

| Job | Case | Debug env | Result | Detail |
|---|---|---|---|---|
| 1 | probe `tiny` (K=64, 1 K-block), kernels **before** A19 | off | **HANG** | program dispatched 18:12:42, no completion until the 2700 s timeout; the single core multicasts in0 to itself (`MCAST_INCL_SRC`, 1 dest) and unicasts its output block to itself — the §6 emulator self-transfer hazard |
| 2 | probe `tiny`, after A19 (`local_l1_copy` for both self-transfers) | WATCHER=5 + DUMP_ALL, NoC sanitizer off, LLK + lightweight asserts on | **PASS** | PCC **0.999997**, max abs err 0.155, 35 s on device; no assert fired |
| 3 | probe `kspill` (K=128, `in0_block_w=1` → 4 K-blocks: partials pushed/reloaded 3× — the shape craq-sim#355 aborts) | same debug env | **PASS** | PCC **0.999995**, max abs err 0.307, 53 s on device; no assert fired. Confirms craq-sim#355 is a simulator artifact and that the compute-kernel Quasar fixes (pack re-target to `out` on the last block, partials round-trips) are correct on RTL |
| 4 | probe `kspill` | off (recipe §9: run both ways) | **NOT RUN** | queued 19:09 behind two other agents' emulator jobs (each up to 45 min); cancelled while still waiting on the emulator lock (never started, device untouched). Follow-up: `qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py kspill` with the debug variables unset. |

Not re-gridded beyond the bank/core count: the emulator has a single DRAM bank, so the weights are one DRAM shard.

---

## Open blockers (owner)

| # | Blocker | Symptom | Owner | Op-side state |
|---|---|---|---|---|
| B1 | craq-sim false abort on the 2nd push into an intra-tensix (PACK→UNPACK self-loop) DFB — tenstorrent/craq-sim#355 | `qsr_tile_counter_check_error: occupancy=2*capacity … acked=0` at the 2nd partials push; hits every K-spill matmul (all 6 captured cases) | craq-sim | none (do not change the op); validate K-spill on RTL |
| B2 | `tt_metal/common/core_assignment.cpp:279` has no `ARCH::QUASAR` arm | `TT_THROW("Invalid Arch Name specified")` from `IDevice::get_optimal_dram_bank_to_logical_worker_assignment()` | runtime | interim factory-local placement (A10) — delete it when the runtime arm lands |
| B3 | 2D-mcast factory (`MatmulMultiCoreReuseMcast2DProgramFactory`, cases 05/06) not Metal 2.0 | `create_descriptor`, legacy device API (see 2026-09-10 report for the kernel list; in1 sender carries an `ENABLE_GLOBAL_CB` path) | matmul Metal 2.0 port | RED, untouched |
| B4 | Dead `in0_mcast_sender_valid` semaphore (VALID-init, bound by no kernel) | rejected by the Gen2 validator | op owner (cleanup PR) | omitted on Quasar only; recommend deleting for all archs |
| B5 | `evil_set_read_ptr` (`MM_PARTIALS_RELOAD_ALIAS` reload path) is `#ifndef ARCH_QUASAR` | would fail to build on Quasar if a factory defines it (FUSE_BIAS + FP32 reload alias) | runtime (DFB ring-rewind API on Gen2, §7) | not on this factory's path |
| B6 | Quasar matmul LLK bakes `ct_dim`/`rt_dim` at init; a narrower per-call `ct_dim` hangs | case 07 hang (A12) | LLK (FYI / doc) | avoided on Quasar (no padded subblocks) + `static_assert` |
| B7 | Model dtypes: BFLOAT8_B/BFLOAT4_B weights (and bf8 outputs) in every captured case | no Bfp8/Bfp4 DFB format on Quasar | llama32_1b_quasar owners | tests run bf16 |
| B8 | Harness: `from_torch(layout=TILE, device=)` runs a legacy on-device tilize; captured 12-bank / 8x8 grids do not exist on the sim/emulator | `DataMovementKernel is not supported on Quasar`; `graph_case.build_memory_config` skips | graph_ops harness | temp copy host-tilizes + re-grids |

---

## Parity claim (WH/BH)

Every Quasar-specific edit is `is_quasar`-checked on the host or `#ifdef ARCH_QUASAR` in kernels (A1, A2, A4, A6,
A9, A10, A11, A12), so WH/BH execute the pre-uplift instruction stream. The one unguarded change is A8, the
canonical `sync_free_dfbs.md` conversion of two sync-free **borrowed** DFBs to `LocalTensorAccessor`: the kernels
read the same L1 base addresses (a borrowed DFB's base is the tensor shard's address; `LocalTensorAccessor` reads
the same address from the tensor binding's CRTA), perform the same reads/writes in the same order, and no credit
call ever existed on those buffers. Side effect on WH/BH: the program declares two fewer DFB slots (CB indices of
the remaining buffers shift), no L1 footprint change (borrowed views own no L1). No WH/BH device was available in
this session; run the control commands below before merging.

---

## Repro commands

```bash
# craq-sim (Quasar functional simulator)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh linear_ds
qsr_rebuild                                                    # after host edits; expect REBUILD_OK
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py tiny     # PASS
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py kspill2  # PASS
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py kspill   # craq-sim#355
# graph cases: temp copy test_tmp_linear_ds_qsr.py placed next to graph_ops/test_linear.py (archived in scratch)
qsr_test timeout 900 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_linear_ds_qsr.py -k "03_" -v -s -p no:cacheprovider

# ZEBU RTL emulator (1x3)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py tiny
qsr_emu timeout 2700 ./python_env/bin/python <scratch>/qsr_linear_dram_sharded/qsr_linear_ds_tiny.py kspill

# Debug env when something hangs (copy generated/watcher/watcher.log immediately)
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1

# WH / BH control (must be unchanged; run before merge)
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "in1_dram_sharded" -v
pytest tests/ttnn/unit_tests/operations/matmul/test_linear.py -k "dram_sharded" -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -v
```

---

## Definition-of-done checklist (§10)

- [x] Uplifted in place (existing directory + namespace); nothing copied into `experimental/quasar/`, no `::qsr`.
- [x] Factory is `create_program_artifacts`/`ProgramArtifacts`; kernels use `dfb::`/`args::`/`tensor::`/`sem::`.
- [x] `opt_level` matches legacy (compute O3 explicit, DM absent → O2).
- [x] Every DFB has `data_format_metadata`; no `fifo_page_size` reads.
- [x] Sync-free borrowed DFBs converted (`LocalTensorAccessor`); no DM self-loop left; borrowed capacities n/a.
- [x] No `disable_dfb_implicit_sync_*`.
- [x] No offset-borrow ported as-is (`alias_with` is same-base/same-size; single-block output).
- [x] Every multicast rectangle ascending on Quasar (one call site, arch-normalized).
- [x] Re-`*_init` before each execute on new DFB ids (reload path) and pack re-init on every pack-destination switch.
- [x] No bare `wait_front`→`pop_front` / `reserve_back`→`push_back` in the compute kernel on Quasar (3 unpack sites fixed; pack pairs all carry a real PACR).
- [x] No non-zero-init semaphore on Quasar (dead VALID semaphore omitted there).
- [ ] BH and WH pass the existing suites — **not run here (no device)**; commands above.
- [x] Quasar builds, validates and runs (craq-sim PASS on the K≤2-block probes; K-spill gated by craq-sim#355 → RTL).
- [x] No DIAG/debug leftovers; functional workarounds documented (A10 interim placement).
- [x] Missing runtime/LLK items flagged (B1, B2, B5, B6) rather than bundled.
- [x] Report written with GREEN/RED per factory, changed-file list, parity claim, RED-stop conditions checked (2D-mcast RED).
