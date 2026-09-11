# Quasar craq-sim run — `ttnn.experimental.quasar.linear` / `.matmul` (2026-09-10)

**Branch:** `vsureshTT/quasar_uplift_round_2` (uncommitted; delete this file before merge)
**Op:** `ttnn/cpp/ttnn/operations/experimental/quasar/matmul` (the Quasar Metal-2.0 matmul copy)
**Simulator:** craq-sim `libttsim.so` (Quasar, 8x4 = 32 worker cores, **2 DRAM banks** (`dram_grid_size()=2x1`), ~1 kHz),
slow dispatch, force-JIT, `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-linear`.
Host libs rebuilt from this tree via `qsr_rebuild` (REBUILD_OK, `build_Release/lib/_ttnncpp.so` 20:34).

## Result table

| # | Case | Route (factory) | Result | Detail |
|---|---|---|---|---|
| A0 | prototype `test_linear[attn_wo-decode-batch1]` as checked in (`U.to_tt` = `from_torch(..., device=)`) | — | FAIL (harness) | `kernel.hpp:450 DataMovementKernel is not supported on Quasar` from the on-device tilize inside `from_torch`; op never reached. Temp copy host-tilizes + `ttnn.to_device`. |
| A1 | prototype `attn_wo-decode-batch1` (M=32,K=2048,N=2048, bf16 DRAM interleaved), host-tilized | auto → `MatmulMultiCoreReuseMcast1DProgramFactory` (mcast_in0; per_core_M=1, per_core_N=2, in0_block_w=2, 32 K-blocks) | FAIL (build) → fixed | JIT: `reader_bmm_tile_layout_in0_receiver_metal2.cpp:59: 'Semaphore::get_l1_addr() const' is private`. Fixed (see Fixes #1). |
| A1' | same, after fix #1 | 1D mcast_in0 | **FAIL (sim abort)** | `[~8.2k cycles] UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=4 exceeds capacity=2 (posted=4 acked=0)`. Blocker B1. |
| A2..A29 | remaining prototype sites/sizes (mlp_w1/w3/w2, attn_qkv, lm_head × seq128/512/1024, batch1/32) | 1D (M≤128 narrow) / 2D (M≥512) | NOT RUN | Every model shape has Kt=64..256 with in0_block_w≤8 → K-spill through `cb_intermed0` → same blocker B1 (see probes P1/P2/P5, graph 05/06). |
| P1 | probe M=32,K=2048,N=2048, explicit 1D cfg (= auto) | 1D mcast_in0 | FAIL | as A1'. Identical after fixes #2/#3 (cycle 8327 all three runs). |
| P2 | probe N=1024 (per_core_N=1) | 1D mcast_in0 | FAIL | `occupancy=2 exceeds capacity=1 (posted=2 acked=0)` → capacity == `out_block_tiles` → DFB is `cb_intermed0`/`cb_out`. |
| P3 | probe fp32_dest_acc_en=True | 1D mcast_in0 | FAIL (host) | `TT_FATAL: Compute kernel 'compute' consumes FP32 DFB 'cb_intermed0' with enable_32_bit_dest=true, but provides no unpack_modes entry` — separate host-spec gap (Blocker B3). |
| P4 | probe K=64 (1 K-block, no partials) | 1D mcast_in0 | **PASS** PCC 0.99999 (pre and post fixes) | Proves DM readers/mcast/writer/compute/out path work on the sim. After fix #3 alone it produced PCC 0.0 → exposed the pack-destination issue → fix #2. |
| P5 | probe packer_l1_acc=True (model's compute config) | 1D mcast_in0 | FAIL | same counter fault (`4 > 2`, cycle 8261). |
| P6 | probe K=128,N=1024 (2 K-blocks: one partials push + reload, no 2nd push) | 1D mcast_in0 | **PASS** PCC 0.99999 (with fixes #2+#3) | Partials round-trip itself works; only the *second* partials push after a pop trips the sim check. |
| G00 `00_32x2048_bf16_ws-l1` | DRAM-sharded cfg, in0 L1 grid 8x8 | — | SKIP | captured L1 shard grid needs 64 cores; device 8x4. |
| G01 `01_…` (bf8 weights→bf16) | DRAM-sharded | NOT RUN | same factory as G02/G03 (fails at host before kernels). |
| G02 `02_32x2048_bf16_ws-l1` (2048x3072, bf8→bf16, DRAM grid 12→2 banks) | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (legacy `create_descriptor`) | FAIL (host) | `TT_THROW @ tt_metal/common/core_assignment.cpp:279: Invalid Arch Name specified` — reached from `device/utilities/matmul_utilities.cpp:382 get_optimal_dram_bank_to_logical_worker_assignment(noc)` (no `ARCH::QUASAR` case); this precedes the expected `DataMovementKernel is not supported on Quasar` guard. |
| G03 `03_32x2048_bf16_ws-l1` (2048x2048) | DRAM-sharded (legacy) | FAIL (host) | identical to G02. |
| G04 `04_32x8192_bf8_ws-l1` | DRAM-sharded | SKIP | in0 L1 grid 8x8 (64 cores) exceeds device. |
| G05 `05_1024x2048_bf16_int-dram` (2048x8192 bf4→bf16; grid 8x8→8x4, per_core_M 4→8, weights DRAM-sharded(12)→interleaved) | `MatmulMultiCoreReuseMcast2DProgramFactory` (Metal 2.0), in0_block_w=8, out_block 8x32 | **FAIL (sim abort)** | `[63599] occupancy=260 exceeds capacity=256 (posted=260 acked=0)` — capacity = out_block_tiles (8x32), one extra subblock (4) = 2nd-block partials push. Blocker B1. |
| G06 `06_1024x2048_bf16_int-dram` (2048x2048 bf8→bf16; grid 8x8→8x4) | 2D mcast (Metal 2.0), in0_block_w=1, out_block 8x8 | **FAIL (sim abort)** | `[12245] occupancy=68 exceeds capacity=64 (posted=68 acked=0)`. Blocker B1. |
| G07 `07_…` (2048x5376, bf8) | DRAM-sharded | NOT RUN | same as G02/G03. |
| `prototype_ops/test_minimal_matmul.py` | — | NOT IN SCOPE | targets `ttnn.experimental.minimal_matmul` (mainline), not the quasar copy. |

Model-level dtype decision recorded: captured BFLOAT8_B/BFLOAT4_B weights and bf8 outputs were run as BFLOAT16 (Quasar has no Bfp8/Bfp4 DFB format; `matmul.cpp:482-514` rejects bf8 on Quasar).

## Fixes applied (all inside `experimental/quasar/matmul`)

1. `device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp` — replaced the raw-pointer read through the private
   `Semaphore::get_l1_addr()` with the public `receiver_sem.value()` (the sender multicasts `VALID` or `IGNORE_BATCH`, so the
   read is load-bearing). Symptom: JIT build failure on Quasar (`get_l1_addr() is private`). Recipe §8.1.
2. `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` — `#ifdef ARCH_QUASAR pack_init(<cb>)` after
   every pack-destination switch (block-start back to `cb_intermed0`; in0-transpose path; the last K-block's switch to
   `mm_out_cb` (once per block, before the subblock loops); FUSE_BIAS epilogue; untilize epilogue). On Quasar the pack BFD is
   baked at `pack_init`, `pack_reconfig_data_format` is gasket-only (recipe §7, same idiom as `conv_bmm_tilize_metal2.cpp`).
   Symptom: PCC 0.0 on P4/P6 once `cb_out` and `cb_intermed0` no longer shared L1 (the final pack landed in the partials ring).
   With the fix P4 = 0.99999, P6 = 0.99999.
3. `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (2 builders), `..._mcast_2d_program_factory.cpp`,
   `..._reuse_optimized_program_factory.cpp` — `separate_out_interm` forced on `ARCH::QUASAR` (no `alias_with` of the
   intra-tensix partials DFB onto the DM-consumed output DFB; +1 out-block of L1/core, WH/BH unchanged). Recipe §7 rule.
   NOTE: applied reactively for B1 but it did **not** change the fault (same cycle); kept because the recipe mandates it and
   because fix #2 makes it correct. Revertible independently if L1 is tight.

## Open blockers

- **B1 (runtime / craq-sim — intra-tensix DFB tile-counter accounting).** Every K-spill matmul (all 8 model cases, both the 1D
  and 2D Metal-2.0 factories) aborts the simulator on the *second* push into the intra-tensix partials DFB `cb_intermed0`
  (PACK producer → UNPACK consumer, same Neo; bound PRODUCER+CONSUMER on the compute kernel exactly as
  `conv2d_op_width_sharded_program_factory.cpp:547-566`): `qsr_tile_counter_check_error: occupancy = 2*capacity, posted = 2*capacity,
  acked = 0`, capacity = out_block_tiles. The packer's `reserve_back` *did* see the unpacker's `pop_front` (otherwise it would hang,
  and P6 shows one push→pop round-trip works and yields correct data), but the counter the simulator checks never receives the
  ack. Not op-level: kernel sequence is the standard `reserve_back → pack_block → push_back` / `wait_front → copy_block →
  pop_front` (TEN-4746 ordering satisfied; drains use `dummy_unpack`). Cross-check: the runtime's own intra-tensix gtests
  (`tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer_intra.cpp`, `TensixIntraTest1xDFB1Sx1S_2_0`,
  `TensixIntraTest1xDFB4Sx4S`) cannot run on this sim either: `UnimplementedFunctionality: tensix_execute_pacr_stride`
  (`unit_tests_api --gtest_filter=*TensixIntra*`). Owner: runtime DFB team / craq-sim (remapper / Tensix-only TC model,
  DataflowBuffer.md §C7). Op-side workaround that avoids partials entirely is `in0_block_w == Kt` (P4), not viable for the
  model shapes (Kt = 64/256).
- **B2 (DRAM-sharded factory — expected, not ported).** `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` is legacy
  `create_descriptor`; on Quasar it fails first in `tt_metal/common/core_assignment.cpp:279` (`Invalid Arch Name specified`, no
  QUASAR arm) via `device/utilities/matmul_utilities.cpp:382`. Behind that sits the `DataMovementKernel is not supported on
  Quasar` guard (`kernel.hpp:450`). 6 of the 8 captured linear cases (1872/1968 calls) route here. Owner: matmul Quasar port
  (Metal 2.0 port of the DRAM-sharded factory + its 2 DM kernels) and runtime (`core_assignment` Quasar arm).
- **B3 (host spec gap, fp32 accumulate).** With `fp32_dest_acc_en=True` the compute KernelSpec has no `unpack_modes` entry for the
  Float32 `cb_intermed0` (`ValidateProgramSpec` FATAL, P3). Owner: this op (all Metal-2.0 builders); not on the model's path
  (model uses fp16 acc).
- Environment: `from_torch(layout=TILE, device=…)` for bf16 runs a legacy on-device tilize → prototype/graph harnesses must
  host-tilize + `ttnn.to_device` on Quasar (harness-level; not changed in repo tests).
- Grid: captured 8x8 L1 grids (cases 00/04) and 12-bank DRAM shards cannot exist on the 8x4 / 2-bank sim (skipped / re-gridded).

## Repro

```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh linear
# probes (script archived in the session scratch dir: qsr_mm_probe.py; args P1_auto|P2_N1024|P3_fp32acc|P4_K64|P5_l1acc|P6_K128_N1024)
qsr_test timeout 1800 ./python_env/bin/python <scratch>/qsr_mm_probe.py P1_auto
# graph cases: temp copy test_tmp_linear_qsr.py (archived in scratch; deleted from repo) next to graph_ops/test_linear.py
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_linear_qsr.py -k "06_" -v -s
# runtime intra-tensix DFB test on the sim
qsr_test timeout 900 ./build_Release/test/tt_metal/unit_tests_api --gtest_filter='*TensixIntraTest1xDFB1Sx1S_2_0*'
```
Debug env used: `TT_METAL_WATCHER=1 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1
TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` (no LLK assert fired before the sim abort; watcher dump never completes because the
sim aborts the process).

## craq-sim re-run 2026-09-11

Same tree as above with the three committed fixes (Semaphore::value() in the in0 receiver, ARCH_QUASAR `pack_init` after every
pack-destination switch, `separate_out_interm` forced on Quasar); host libs from `build_Release` (no rebuild needed).
craq-sim `libttsim.so` (8x4 grid, 2 DRAM banks, ~1-2 kHz), `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-linear_rtl`.
Graph cases via a temporary copy `graph_ops/test_tmp_linear_qsr.py` (`_OP = ttnn.experimental.quasar.linear`, config kinds
resolved from `ttnn._ttnn.operations.experimental.quasar`, bf16, host tilize + `ttnn.to_device`, 2D cases: 12-bank DRAM-sharded
weights -> DRAM interleaved, grid 8x8 -> 8x4 with per_core_M/out_block_h 4 -> 8; deleted after the run). The single-core 2D
probes (`qsr_mm2d_probe.py`, archived in the session scratch dir) were run here first as a host-side check before the emulator.
Wall time = whole process (open device, JIT, run) as measured around `qsr_test timeout ... python`.

| Case | Factory | Shape / config | Result | PCC or exact error | Wall |
|---|---|---|---|---|---|
| `craqsim_mm_kspill_probe.py 128` | 1D mcast_in0 (M2.0), 1x1 grid | M=32 K=128 N=64, in0_block_w=2, 2 K-blocks (1 partials round trip) | **PASS** | PCC 0.999993 (sim 8676 cycles) | 15 s |
| `craqsim_mm_kspill_probe.py 2048` | 1D mcast_in0 (M2.0), 1x1 grid | M=32 K=2048 N=64, in0_block_w=2, 32 K-blocks | **FAIL (sim abort)** | `[8334] UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=4 exceeds capacity=2 (posted=4 acked=0)` — identical signature to 2026-09-10 (craq-sim#355) | 15 s |
| `qsr_mm2d_probe.py p2d` | **2D mcast (M2.0), 1x1 grid** | M=64 K=1024 N=64, in0_block_w=2, out_block 2x2, subblock 1x2 (2 subblocks), 16 K-blocks | **FAIL (sim abort)** | `[8624] ... occupancy=6 exceeds capacity=4 (posted=6 acked=0)` (capacity = out_block_tiles 4, +1 subblock of 2) — #355; host-side validation, JIT and first K-block OK | 14 s |
| `qsr_mm2d_probe.py g06_1core` | 2D mcast (M2.0), 1x1 grid | case-06 per-core tile: M=128 K=2048 N=256, in0_block_w=1, out_block 4x8, subblock 1x4, 64 K-blocks | **FAIL (sim abort)** | `[10420] ... occupancy=36 exceeds capacity=32 (posted=36 acked=0)` — #355 | 14 s |
| `qsr_mm2d_probe.py g05_1core` | 2D mcast (M2.0), 1x1 grid | case-05 per-core tile: M=128 K=2048 N=1024, in0_block_w=8, out_block 4x32, subblock 1x4, 8 K-blocks | **FAIL (sim abort)** | `[87606] ... occupancy=132 exceeds capacity=128 (posted=132 acked=0)` — #355 | 17 s |
| graph `06_1024x2048_bf16_int-dram_qsr` | 2D mcast (M2.0), 8x4 grid | 1024x2048 @ 2048x2048 bf16, in0_block_w=1, per_core 8x8, out_block 8x8, subblock 1x4 | **FAIL (sim abort)** | `[12245] ... occupancy=68 exceeds capacity=64 (posted=68 acked=0)` — identical (same cycle) to 2026-09-10; #355 | 17 s |
| graph `05_1024x2048_bf16_int-dram_qsr` | 2D mcast (M2.0), 8x4 grid | 1024x2048 @ 2048x8192 bf16, in0_block_w=8, per_core 8x32, out_block 8x32, subblock 1x4 | **FAIL (sim abort)** | `[59439] ... occupancy=260 exceeds capacity=256 (posted=260 acked=0)` — same counters as 2026-09-10 (cycle 63599 then); #355 | 39 s |
| graph `02_32x2048_bf16_ws-l1_qsr` | DRAM-sharded (legacy `create_descriptor`, not ported) | 32x2048 @ 2048x3072, in0 L1 WS 8x4, in1 DRAM WS 2 banks | **FAIL (host)** | `TT_THROW @ tt_metal/common/core_assignment.cpp:279: Invalid Arch Name specified` via `get_optimal_dram_to_physical_worker_assignment` <- `Device::get_optimal_dram_bank_to_logical_worker_assignment(NOC)` (matmul_utilities.cpp:382) — unchanged (B2) | 16 s |

Blockers: unchanged — **B1 = craq-sim#355** (every K-spill shape, 1D and 2D factories, single- and multi-core; owner craq-sim), **B2**
DRAM-sharded factory legacy + `core_assignment.cpp` has no QUASAR arm (owner: matmul Quasar port / runtime). No op-side symptom
appeared; no source edits made.

## RTL emulator run 2026-09-11

First run of the Quasar copy's **2D-mcast factory** (`MatmulMultiCoreReuseMcast2DProgramFactory`, Metal 2.0) on ZEBU
`emu-quasar-1x3` (1 worker core, `device_grid=1-1`), same host libs/tree as the craq-sim re-run, `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-emu-1x3`,
force-JIT off (cache cold on the first job). Script `qsr_mm2d_probe.py` (session scratch dir; bf16 DRAM-interleaved in0/in1/out,
host tilize + `ttnn.to_device`, `ttnn.experimental.quasar.linear` with an explicit `MatmulMultiCoreReuseMultiCastProgramConfig` on a
1x1 grid, HiFi4, fp16 acc, no `packer_l1_acc`, PCC vs torch fp32). On a 1x1 grid the in0/in1 senders multicast to zero other cores, so
the run exercises the 2D factory's kernels + the intra-tensix `cb_intermed0` partials path (the one craq-sim#355 aborts on), not the mcast fabric.
"Op wall" = `linear` + `to_torch` readback; "job wall" = whole process incl. emulator boot and JIT. Every job booted on the first try
(no `WRP0625E` slot contention in the launcher logs); lock waits of 1-5 min behind other agents' jobs are excluded from both times.

| Case | Factory | Shape / config (1x1 grid) | Result | PCC | Op wall / job wall |
|---|---|---|---|---|---|
| (1) `p2d` — brief's shape | 2D mcast (M2.0) | M=64 K=1024 N=64, in0_block_w=2, out_block 2x2, out_subblock 1x2 (2 subblocks), 16 K-blocks / 15 partials round trips | **PASS** | 0.99995 | 9.1 s / 52 s |
| (2) `g06_1core` — graph case 06 shrunk to one core | 2D mcast (M2.0) | Captured 06 = 1024x2048 @ 2048x2048 on 8x8, per_core_M=4, per_core_N=8, in0_block_w=1, out_subblock 1x4, out_block 4x8, fuse_batch=1. **Changed:** grid 8x8 -> 1x1 and M,N cut to exactly one core's tile (M=128, N=256); K, per_core_M/N, in0_block_w, sub/out-block, fuse_batch unchanged; bf8 weights -> bf16; DRAM-sharded(12) weights -> DRAM interleaved. 64 K-blocks / 63 round trips, out_block_tiles=32 | **PASS** | 0.99991 | 44.1 s / 88 s |
| (3) `g05_1core` — graph case 05 shrunk to one core | 2D mcast (M2.0) | Captured 05 = 1024x2048 @ 2048x8192 on 8x8, per_core_M=4, per_core_N=32, in0_block_w=8, out_subblock 1x4, out_block 4x32, fuse_batch=0. **Changed:** grid 8x8 -> 1x1, M=128, N=1024 (one core's tile); everything else as captured; bf4 weights -> bf16; DRAM-sharded(12) -> interleaved. 8 K-blocks / 7 round trips, out_block_tiles=128 | **PASS** | 0.99991 | 87.0 s / 134 s |

Conclusion: all three K-spill shapes that abort craq-sim with `qsr_tile_counter_check_error` (see the re-run table above: `p2d` 6>4,
`g06_1core` 36>32, `g05_1core` 132>128) run to correct results on RTL, on the 2D factory as well as the 1D factory (K=2048 probe,
2026-09-10). This confirms **B1 is craq-sim#355** (owner: craq-sim), not this op or the runtime DFB layer. Not covered on RTL:
multi-core mcast (1x3 has one worker core), the DRAM-sharded factory (B2, host-side `core_assignment.cpp:279`, no device needed to
reproduce), fp32 accumulate (B3). No source edits were made; `test_tmp_linear_qsr.py` deleted; logs (`sim_*.log`, `emu_*.log`,
`launcher_*_emu_*.log`, `qsr_mm2d_probe.py`) archived in the session scratch dir `qsr_linear_rtl/`.
