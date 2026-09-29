# Changelog: high_bw_all_reduce

## Phase 0 — Core Implementation
- **Date**: 2026-09-26
- **What was done**: initial implementation via the incremental pipeline (planner → implementer →
  verifier). R1 `chain_line`: each chunk is reduced hop by hop toward the line's tail in fp32 DEST, then
  the final sum is relayed back over Fabric 2D with one Fabric connection per (lane, direction). The op
  makes one `ttnn.generic_op(MeshProgramDescriptor)` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned],
  cluster_axis=[0, 1], topology=[Linear], num_links=[1, 2] (None means every usable link).
- **Accuracy achieved** (bf16, G=2, 4 shapes × 2 axes, `test_high_bw_all_reduce_precision_baseline.py`):
  PCC ≥ 0.9999985, max_abs_err = 0.0156, mean_abs_err ≈ 0.00145, rel_rms_err ≈ 0.0018. Output is within
  1 bf16 ULP of the once-rounded reference, and about 89% is bit-exact.
- **Perf** (BH 2×2, device kernel): 8 MB takes 392 µs at 1 link and 203 µs at 2 links. 32 MB takes
  1.53 ms and 0.77 ms. That is ≈ 21 GB/s per link direction.
- **Golden suite at Phase 0**: 52 / 400 passed (48 supported_pass + 4 regression), 336 xfail_expected,
  12 regression refusals (fp32 / w_non_aligned). supported_fail = xpass_drift = xfail_wrong_mode = 0
  (per `verifier_report.json`).
- **Issues encountered / verifier fixes**:
  - The compute kernel synced per tile. It now uses chunk-granular helper calls (upfront wait / pop at
    end, `InputTileMapping::Block`, one reserve/push per chunk), with `chunk_tiles` passed as a CT arg.
  - `CONTROL_WORD_STRIDE` and `MAX_DATA_HEADERS` were duplicated as literals in both port kernels. They
    are now single-sourced on the host as CT args.
  - Removed the reader's dead `packet_tiles` CT arg.
  - `l1_ledger.md` main body updated to the implemented knob values (64 KiB chunks, W ≤ 4).
  - Test infra: a passing `--dev` fabric run leaves the eth routers dead. Workaround: `touch
    /tmp/tt-device.dirty` before each run.
- **Tests added**: `test_high_bw_all_reduce.py` (acceptance, 36), `test_high_bw_all_reduce_perf.py`
  (profile harness), `test_high_bw_all_reduce_precision_baseline.py` (8).

## Refinement 1 — Ring topology + whole-mesh snake (`cluster_axis=None`)
- **Date**: 2026-09-26
- **What was done**:
  - **R2 `chain_snake_line`** (None, Linear): the host `route()` emits a row-snake line.
  - **R3 `rotated_chain_ring`** (None, Ring): the host emits a snake Hamiltonian cycle with G slices. Slice
    j's chain starts at ring position j, and every edge carries partials one way and finals the other,
    including the closing edge.
  - **Per-block roles**: head, middle or tail is decided per block, not per device. The single source is the
    new `kernels/high_bw_all_reduce_roles.hpp`: reducer r's block stream is cut into `G / gcd(G, W)`
    segments, one slice each.
  - **Port kernels**: both now serve reducers independently (round-robin by readiness, in order per reducer).
  - **Per-reducer counters**: every port-to-port counter is per reducer (`gsem_partial_credit_r`,
    `gsem_final_arrival_r`, `gsem_final_credit_r`). `final_freed[r]` moved into the control array, and a
    new `final_egress` semaphore gates tail writes into the final ring.
  - **Why the global order had to go**: a single global lane order coupled the rotated chains around the
    ring, and 32 MB None-Ring took 6.6 ms. Per-reducer service brought it to 1.5 ms.
  - **Host**: `num_links` is checked against every route edge, including snake corners and the closing edge.
    The gsem cache is keyed on the route kind (line / ring), so Linear and Ring never share counters. Ring
    requests the cluster cannot close raise `ValueError`.
  - **EXCLUSIONS**: per-axis Ring (`{Ring, 0}`, `{Ring, 1}`). These need a torus cluster, which was not
    available to verify.
  - **Perf fix**: the per-packet final-credit poll in `port_fwd` is O(1), checking one reducer per call. A
    W-wide scan per packet had cost the axis lines +19%.
  - **Reused**: every CB and L1 ring and every helper call are unchanged. The existing port kernels were
    extended in place.
  - **Added**: the roles header, per-reducer counters, and the path builder.
- **Accuracy achieved**:
  - Exact (bit-identical) rank-identity and single-contributor sums on the G=4 snake line and ring:
    first, middle and last contributor.
  - Random-data PCC ≥ 0.995 on (1,1,32,32), (3,96,160), (1,1,256,512) and (1,1,2048,2048) at 1 and 2 links.
  - The Phase 0 precision baseline is unchanged (8/8).
- **Perf (BH 2×2, device ns, 1 / 2 links)**:

  | Case | 32 MB | 8 MB |
  |---|---|---|
  | None-Ring | 1.52 / 0.93 ms | 412 / 270 µs |
  | None-Linear | 1.79 / 1.16 ms | 469 / 344 µs |
  | Axis lines | 1.53 / 0.77 ms (unchanged) | 392 / 204 µs (unchanged) |

- **Golden test progress**: all 48 `cluster_axis=None` bf16 tile_aligned cells (Linear and Ring, 1 and 2
  links) pass. Prior representatives pass, as do the golden regression rank-identity and single-contributor
  tests. Unit tests: 36/36 acceptance, 8/8 precision baseline, 28/28 new.
- **Issues encountered**:
  - **Ring 4× slower than the line**: caused by the global-lane-order coupling. Fixed with per-reducer
    service and per-reducer counters.
  - **Axis-line regression**: caused by the W-wide credit scan per packet. Fixed with an O(1) poll.
  - **Perf-test ids**: the `--profile` wrapper's shell cannot take `-k` expressions with spaces or
    parentheses, so the perf test ids are now single tokens (`links1`, `none-ring`, …).
- **Tests added**:
  - `test_high_bw_all_reduce_ring_snake.py` (28): whole-mesh Linear and Ring random data, G=4 exactness
    (rank-identity and single-contributor), a Linear↔Ring↔axis back-to-back alternation, and a per-axis
    Ring refusal.
  - `test_high_bw_all_reduce_perf.py` now covers None-Linear and None-Ring.

## Refinement 2 — float32 end-to-end + non-aligned shapes
- **Date**: 2026-09-26
- **What was done**:
  - **SUPPORTED**: `dtype` gains `float32`; `alignment` gains `w_non_aligned` and `h_non_aligned`. No
    EXCLUSIONS were added.
  - **fp32 compute**: the compute kernel takes a new CT arg `use_sfpu_add`, single-sourced on the host
    as `dtype == float32`.
    - With the flag set, the non-head add is `binary_sfpu<AddBinary<>, in_partial, in_local, out_reduced>`.
      Both operand CBs are tagged `UnpackToDestFp32`, so both copy into fp32 DEST and the add is full
      fp32. The FPU path would truncate operands to tf32 in SrcA/SrcB.
    - The chunk-granular lifecycle is unchanged (`Upfront` / `AtEnd` + `InputTileMapping::Block`), and
      the head is still a `copy`.
    - bf16 keeps the FPU `add`: bf16 is lossless in tf32, and its kernel is identical apart from the
      extra CT arg.
  - **fp32 wire**: every CB page, the L1 rings and the Fabric payload are Float32, with no downcast.
    They were already derived from the input dtype (`tile_bytes = 4096`, `packet_tiles = 1`,
    `chunk_tiles = 16`), so the L1 footprint is unchanged.
  - **Non-aligned shapes**: no kernel change. The op sums physical tile pages, and the output reuses
    the input TensorSpec, so padding never reaches the logical view.
  - **Reused**: every CB, helper call, kernel file and program-descriptor branch.
  - **Added**: one `if constexpr` compute branch, one CT arg, and the `unpack_to_dest_mode` vector on
    the fp32 path.
  - **Signature unchanged**: there is no `compute_kernel_config` kwarg (the spec fixes the signature),
    and `fp32_dest_acc_en=True` stays hard-wired.
- **Accuracy achieved**:
  - **fp32, exact**: sums of `k·2^-12` values (12 significant bits) are bit-exact on axis 0/1 lines,
    the None snake and the None ring, at 1 and 2 links. These sums are exact in fp32 but not in tf32,
    so the test also proves nothing is truncated on the wire or in SrcA/SrcB.
  - **fp32, random** (`randn·1e3`): rel_rms < 1e-6 and PCC ≥ 0.99999 on 2048×2048 and 100×1000 across
    all 4 paths.
  - **Non-aligned, exact**: rank-identity is bit-exact for bf16 and fp32 on 4096×2050, 4001×2048,
    1×1×48×80 and 2×33×65, across all 4 paths.
- **Golden test progress**: 304/304 on the `-k "FLOAT32 or non_aligned or test_regression"` slice,
  which covers every fp32 cell, every non-aligned cell and all 16 `test_regression.py` tests
  (`test_magnitude` fp32 and the 4096×2050 exactness tests included). The 96 bf16 tile_aligned cells
  were deselected; their path is unchanged and the 36/36 acceptance tests cover it.
- **Issues encountered**: two cases in `test_support_refusal` (`float32`, `w_non_aligned`) became
  stale once those values were supported. They were replaced with still-refused values (`bfloat8_b`,
  ROW_MAJOR).
- **Tests added**: `test_high_bw_all_reduce_fp32_nonaligned.py` (64): fp32 beyond-tf32 exactness,
  non-aligned rank-identity (bf16 + fp32), and fp32 random.

## Refinement 3 — Drive the axis lines toward link rate (bf16, 8–64 MB)
- **Date**: 2026-09-26
- **What was done**:
  - **Measured the ceiling first.** A new send-only probe (`probes/perf_ceiling_fabric.py` +
    `probes/kernels/ceiling_*.cpp`) runs the port's exact send loop with no reducers and no credits.
    - It sends 32 MB in 1.44 ms at 1 link and 0.72 ms at 2 links. That is about 237 cycles per 4 KiB
      packet, or 23.3 GB/s per link direction, on BH FABRIC_2D, and it is EDM-bound (38% of the
      loop is slot waits).
    - The 38 GB/s golden is the 1-D NeighborExchange figure.
    - Adding one header-only credit per 16 data packets gives +5.5%, which is exactly the op's
      1.53 ms. Credit packets were the whole gap.
  - **Lever: credit coalescing** on both credit streams: landing grants in `port_bwd`, final-landing
    grants in `port_fwd`. A reducer's credit is sent once `credit_batch` of them are pending, or when
    it is its last one.
    - Batching one stream alone measured flat, because the other direction still binds, so both are
      batched.
    - Deadlock freedom: each ring is at least the batch. The depths are single-sourced as
      `effective + batch − 1` (`_ring_depths`).
    - Host knob: `CREDIT_BATCH_CHUNKS = 4`, passed as a new CT arg to both port kernels. Sweep:
      batch 2 gave −2.8%, 3 gave −3.7%, 4 gave −4.2%.
  - **Re-placed the port scratch**: one per-call lockstep L1 tensor now serves every op core.
    - Reducer CBs sit at shard offsets (`cb_descriptor_from_sharded_tensor(address_offset=…)`); ports
      use the same shard for control and rings. The footprint is max, not sum.
    - This is what lets the batch-4 rings (1.28 MB shard) fit, with chunks still at 64 KiB and W = 4.
    - The global semaphores are now created before the scratch, which fixes the fragmentation OOM
      seen in the unit suites. The batch is clamped to the largest free L1 block.
  - **Path gate** (`_credit_batch`): 2-link chains with middle devices (the None cells at 2 links)
    keep batch 1. Deeper rings alone cost them 2–7%.
  - **Reused**: every kernel, CB id, helper call and program-descriptor branch. Kernel changes are
    one CT arg plus one condition per credit sender.
  - **Added**: the probe, the scratch overlay, the depth derivation, the gate, and the
    `test_perf_guard` perf cases.
- **Perf (BH 2×2, device ns, max over devices, 1 / 2 links)**:

  | Case | Before | After |
  |---|---|---|
  | Axis lines, 32 MB | 1.530 / 0.773 ms | 1.467 / 0.741 ms (−4.1% / −4.1%; 22.9 / 22.6 GB/s per link direction, 98% of ceiling) |
  | Axis lines, 8 MB | 394 / 204 µs | 377 / 198 µs |
  | None-Ring, 1 link, 32 MB | 1.51 ms | 1.38 ms |
  | None-Linear, 1 link, 32 MB | 1.78 ms | 1.74 ms |
  | fp32 16 MB | 780 / 398 µs | 749 / 384 µs |
  | Ragged 4001×2048 | 760 / 388 µs | 729 / 376 µs |

  The 2-link None cells and the single tile are unchanged (within noise).
- **Accuracy achieved**: unchanged. The precision baseline passes 8/8 (PCC ≥ 0.9999985, within 1
  bf16 ULP). The exactness tests stay bit-exact.
- **Golden test progress**: 112/112 on the `(BFLOAT16 and tile_aligned) or test_regression` slice:
  every bf16 tile-aligned cell (axis 0/1, None-Linear and None-Ring, 1 and 2 links) and all 16
  regression tests. fp32 and non-aligned are covered by the unit module (64/64).
- **Unit tests**: 36/36 acceptance, 8/8 precision, 64/64 fp32/non-aligned, 28/28 ring/snake, 22/22 perf.
- **Issues encountered**:
  - 128 KiB chunks at W = 2 were flat or worse (W = 2 limits reducers), and 128 KiB at W = 4 did not
    fit L1 before the scratch overlay.
  - The first batch-4 build OOM'd in the unit suites: the global semaphores were created after the
    scratch, which fragmented L1. Fixed by ordering the allocations and clamping the batch.
- **Tests added**:
  - `test_high_bw_all_reduce_perf.py::test_perf_guard`: fp32, ragged and single-tile guard cells.
  - `probes/perf_ceiling_fabric.py`: the send-only ceiling probe (not collected by default).

## Refinement 4 — Whole-mesh and ring cells: fill and packet-rate tuning
- **Date**: 2026-09-26
- **What was done**:
  - **Measured first**:
    - A knob sweep (`probes/perf_sweep_r4.py`) over ring depths, staging depth, 32/128 KiB chunks,
      W = 2/6/8, headers and NoC swaps moved nothing by more than ±2%.
    - An ablation that removed only `port_bwd`'s output DRAM write gave −14% on the 1-link None cells.
      On snake and ring middles, the relay RISC forwards each final over Fabric and writes it to DRAM
      on one NoC, and that bound the pipeline.
    - A same-bytes single-transfer ablation recovered about a third of that (partly issue-bound,
      partly NoC).
    - The 2-link None cells were bound by both lanes sharing core row 0.
  - **Split ports** (`SPLIT_PORTS = 1`, path-gated to G ≥ 3 by `_split_ports`):
    - Each lane has a `port_fwd` core and a `port_bwd` core.
    - New kernel `kernels/high_bw_all_reduce_port_drain.cpp` runs on the `port_bwd` core's RISCV_0
      (NoC1). It writes the relayed finals to DRAM from local L1 and publishes `drained[r]`.
    - `port_bwd` now relays, then frees a slot once it is relayed and drained (`freed_k[r]`).
    - The kernels take separate `port_fwd` / `port_bwd` core coordinates. When the ports are
      co-located, both coordinates are the same core and the old path is kept.
  - **Lane-per-row placement** (`LANE_ROW_STRIDE = 1`, `_lane_core`): −21% / −17% on the 2-link
    None-Linear / None-Ring cells.
  - **Credit batch on every path**: the R3 2-link gate is removed. It was a row-contention artifact,
    and with a row per lane the batch wins there too.
  - **Bank-run chunk layout** (`BANK_RUN_LAYOUT`, `kernels/high_bw_all_reduce_chunk_io.hpp`,
    shared by the reader, writer, `port_bwd` and `port_drain`):
    - It stores chunks bank-major so each DRAM transfer is one burst per bank.
    - It won 5–6% before the split, but after it is flat at 1 link and costs 2–6% at 2 links.
    - It is parked at 0 (byte-identical page-by-page transfers) as a live knob.
  - **Port NoC selection**: exposed as knobs (`PORT_FWD_NOC` / `PORT_BWD_NOC`) at the old defaults.
    Swaps measured worse.
  - **Reused**: every kernel, CB, helper call, protocol counter and the scratch overlay.
  - **Added**: the drain kernel, one control array, the chunk-I/O header and the host knobs and gates.
- **Perf (BH 2×2, device ns, max over devices, before → after, 1 / 2 links)**:

  | Case | Before | After |
  |---|---|---|
  | None-Linear 32 MB | 1740 / 1167 µs | 1489 / 766 µs |
  | None-Ring 32 MB | 1414 / 938 µs | 1151 / 601 µs |
  | None-Linear 8 MB | 460 / 347 µs | 400 / 222 µs |
  | None-Ring 8 MB | 379 / 275 µs | 320 / 186 µs |
  | fp32 None-Ring 64 MB | 2618 / 1805 µs | 2264 / 1198 µs |
  | Axis lines 32 MB | 1467 / 741 µs | unchanged |
  | fp32 axis line 64 MB | 2920 / 1472 µs | unchanged (99% of the send-only ceiling) |
  | Guard set (fp32 16 MB, ragged 4001×2048, single tile) | — | 744 / 383, 728 / 373, 14 / 14 µs, matching R3 |

- **Findings**:
  - **Fill**: ~40–47 µs fixed at both 8 and 32 MB. It is not dominant, so R5 is not warranted by fill.
  - **fp32**: not packet-rate bound. Its packets are 4096 B, the same as bf16.
  - **Ring slices**: balanced (W = G = 4, one slice per reducer).
  - **Remaining headroom**: 2 links at 90–94% of ceiling.
- **Accuracy achieved**: unchanged. The precision baseline passes 8/8, and the bf16 / fp32 exactness
  tests stay bit-exact (ring/snake 28/28, fp32/non-aligned 64/64).
- **Golden test progress**:
  - 308/308 on `-k "None or test_regression or 2048x2048 or FLOAT32"` with the bank-run layout on.
  - 208/208 on `-k "None or test_regression"` at the final defaults.
- **Unit tests**: 158/158 across the directory.
- **Issues encountered**:
  - Putting both port RISCs on the same NoC (a sweep variant) hangs in DM_DEDICATED_NOC mode. That
    variant was dropped.
  - The bank-run lever changed sign once the split landed, so it is parked.
- **Tests added**: `probes/perf_sweep_r4.py`, the monkeypatching knob/variant sweep (not collected by
  default).

## Perf 1 — perf tournament round 1: measured breakdown, 2 ideas, 0 graduated (op unchanged)
- **Date**: 2026-09-26
- **Focus**: `feature_spec.py` has no `LOOSE_CASES` perf flag, so the cell was free-selected by the
  largest measured gap to its bound: **bf16 (1,1,4096,4096), `cluster_axis=None`, Ring, 2 links**.
  It measures 607 µs against a ring bound of (G−1)/G · S at 23.3 GB/s per link direction ≈ 540 µs
  (89%). Secondary cell: the same config at (1,1,2048,2048), 183 µs vs ≈135 µs (74%).
- **Permanent instrumentation added** (free when the profiler is off; never remove):
  - `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` provides `MaybeDeviceZoneScope`, plus new
    opt-in stall accumulators (`MaybePerfAccum/Begin/End/Report`, one timestamped-data marker per
    kernel).
  - The accumulators compile in only with `-DKERNEL_LIB_PERF_STALLS` under the profiler. When they
    were on by default they cost **+2% measured device time** on fp32 None-Ring 64 MB at 2 links
    (1199 → 1220–1232 µs), because they add two wall-clock reads per packet-slot wait in the port
    loops. With them opt-in, a default `--profile` run matches a zones-off build (1194 vs 1195 µs).
  - Stage zones, all coarse (≤ 4 executions per RISC, far under the 125-zone budget):
    - `reader_wait_go` / `reader_main`
    - `writer_wait_go` / `writer_main`
    - `compute_main`
    - `fwd_setup` / `fwd_main` / `fwd_teardown`
    - `bwd_setup` / `bwd_main` / `bwd_teardown`
    - `drain_wait_go` / `drain_main` / `drain_teardown`
  - Stall totals:
    - reader: `input_reserve`, `dram_read`, `partial_wait`
    - writer: `reduced_wait`, `slot_wait`, `write`
    - port_fwd: `send`, `slot_wait`, `credit`
    - port_bwd: `relay`, `slot_wait`, `free`, `credit`
    - drain: `write`
  - Payload-ablation switches, off unless defined: `HBAR_ABLATE_DRAM_READ`, `HBAR_ABLATE_DRAM_WRITE`
    and `HBAR_ABLATE_COMPUTE`. Each stubs the payload and keeps the CB, semaphore and barrier
    scaffolding.
  - Tooling: `probes/perf1_bench.py` runs the bench cells (with `HBAR_CASES` / `HBAR_LINKS` /
    `HBAR_ABLATE` / `HBAR_DEFINES`), and `probes/perf1_zones.py` prints the per-role zone table.
- **Measured breakdown** (BH 2×2, FABRIC_2D, device-kernel µs, max over the 4 devices):
  - **Size fit (8 vs 32 MB, same config)**: about **41 µs of fixed overhead that does not scale with
    size** at 1 and 2 links. The steady-state rate is 95% of the link ceiling at 2 links and 98% at
    1 link.
  - **Focus zones (per device)**:
    - `bwd_main` 596 µs: relay 410 (of which slot wait 96), free 60.
    - `fwd_main` 565 µs: send 507 (of which slot wait 65).
    - `drain_write` 345 µs.
    - `reader_partial_wait` 338 µs mean.
    - `writer_reduced_wait` 238 µs mean.
    - The ports are the busiest RISCs; the reducers mostly wait.
  - **Cumulative peel** (Ring, 2 links, 32 / 8 MB, from 607 / 183 µs):
    | stubbed | 32 MB | 8 MB |
    |---|---|---|
    | DRAM write | 601 | 180 |
    | + DRAM read | 602 | 178 |
    | + compute | 564 | 150 |
  - **Compute stub alone**: −28 µs on the ring cells (1 and 2 links), −23 / −24 µs None-Linear,
    −21 / −27 µs Ring 8 MB, −9 / −8 µs axis lines. This scales with the number of add hops.
  - **Ranked (by ablation)**:
    1. reducer compute on the partial chain (~8 µs per add hop);
    2. store-and-forward of each transport hop;
    3. steady-state port rate (5% at 2 links).
  - **Correction from the experiments**: rank 1 was an ablation artefact, not payload.
    - The add itself costs **0.83 µs per 32-tile bf16 chunk** in isolation (about 35 cycles per
      tile).
    - A slowdown probe (+1.36 µs per chunk) grows the ring cells by only +8–10 µs, so the whole add
      is worth about 5 µs on the focus cell.
    - The compute stub's large saving comes from somewhere else. The likely cause is that the stub
      pops its inputs and pushes its outputs immediately, which shortens the credit and slot loops;
      that is not measured.
    - Fill-timing markers (idea 2) put the fixed overhead mainly **after the last partial lands**:
      about 31 µs at 32 MB and 35 µs at 8 MB are the finals relayed back hop by hop, chunk by chunk
      (port_bwd). **That is the next round's target.**
- **Portfolio**: the cap was 2 experiments, one subagent each, run in parallel.
  1. `compute_add_fast`: the same math, computed faster.
  2. `subchunk_cut_through`: the partial path signals per sub-chunk instead of per chunk (arrival,
     reader push, compute, staging and port_fwd send), while slots and credits stay whole chunks.
- **Verdicts**:
  - **`compute_add_fast`: NULL end-to-end.** It wins in isolation; every option was bit-identical to
    baseline, including adversarial fp32 inputs:
    | option | bf16 add / copy per chunk | fp32 add per chunk |
    |---|---|---|
    | helper (baseline) | 832 / 1086 ns | 5232 ns |
    | `raw` (init once, 4-tile DEST windows) | 748 / 1019 ns | 4984 ns |
    | `l1acc` (fp32 second operand folded in by packer L1 accumulation, no SFPU add) | 748 / 1019 ns | **2977 ns (1.76×)** |

    End to end it is flat on every bf16 cell (focus 608.1 → 607.2 / 610.9 µs). On fp32 it gains
    0.2–0.6% (None-Ring 64 MB, 2 links: 1225.9 → 1219.7 µs; axis0 16 MB: 383.4 → 381.2 µs). That is
    below materiality and not worth a raw-LLK bypass, so **it was not graduated**. `l1acc` stays the
    recorded option if fp32 compute ever lands on the critical path. Its domain is everywhere
    tested, with no exceptions. Artifacts: `perf_experiments/compute_add_fast/`, drop-in kernel
    `kernels/compute_l1acc.cpp`.
  - **`subchunk_cut_through`: NULL / REGRESSION.** All 90 cells were bit-identical, and the unit
    suites passed against the variant (36 + 28 + 64).
    - Sub-chunks on every chunk are slower everywhere: sub2 +2–4%, sub4 +8–18% (focus: 609.0 →
      635.1 / 720.4 µs). The cost is per-serve overhead on the RISC that paces the port, plus the
      router flushing before every fused increment.
    - Sub-chunks on only the first and last chunk of each reducer: flat at 8 MB (184.4 vs 184.2 µs)
      and +0.6–1.4% at 32 MB (focus 616.8 µs).
    - The first sub-chunk does reach the far tail about 10 µs sooner (19.2 vs 29.0 µs). But the
      first whole chunk and the last partial arrive no earlier, because 4 reducers share each hop's
      link round-robin. The partial stream's finish is throughput-bound.
    - Artifacts: `perf_experiments/subchunk_cut_through/` (variant package `hbar_sub/`, knobs
      `SUBCHUNKS_PER_CHUNK` / `SUB_EDGE_CHUNKS`), plus the re-export shims
      `probes/subchunk_cut_through_{bench,unit}.py`.
- **Graduated**: nothing. The kernels' behaviour is unchanged; only the permanent zones and
  accumulators and the ablation `#ifdef`s were added, and they compile out by default.
- **Golden**: 384/384 in `test_golden.py` (192 bf16 + 192 fp32) and 16/16 in `test_regression.py`.
- **Guard set** (default `--profile`, 1 / 2 links, µs): no regression; every cell matches R4 within
  noise.
  | cell | 1 link | 2 links |
  |---|---|---|
  | None-Ring 32 MB | 1150 | 609 |
  | None-Linear 32 MB | 1489 | 765 |
  | None-Ring 8 MB | 320 | 186 |
  | None-Linear 8 MB | 401 | 220 |
  | axis0 32 MB | 1467 | 742 |
  | axis1 32 MB | 1468 | 742 |
  | axis0 8 MB | 378 | 198 |
  | fp32 axis0 16 MB | 744 | 383 |
  | fp32 None-Ring 64 MB | 2254 | 1191 |
  | ragged 4001×2048 | 730 | 374 |
  | single tile | 14.1 | 14.1 |
- **Summary**: 2 ideas measured, 0 graduated, 2 null. The op is unchanged on every cell, with no
  regression.

### Helper bypasses — none
Nothing graduated, so no raw-LLK path entered the op. For the helper library's information, here
are the measured gaps from the non-graduated `compute_add_fast` experiment:

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::binary_sfpu<AddBinary>` (fp32 add) | capability | cannot express adding the second operand by packing it onto the output tiles with packer L1 accumulation, i.e. a two-pass overlay of a whole block. The helper's L1-accumulation mode folds a stream onto one accumulator tile | 5232 | 2977 | `perf_experiments/compute_add_fast/kernels/compute_l1acc.cpp` (not graduated) |
| `compute_kernel_lib::add` / `copy` (bf16) | ergonomics | re-emits its op and pack init on every call (once per chunk), with no way to keep the init across calls when the role does not change | 832 / 1086 | 748 / 1019 | same file (not graduated) |
