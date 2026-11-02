# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - It is a fused CCL op with one `ttnn.generic_op` dispatch per call.
  - Local half: ported from the base op `rms_norm`, with regimes R1 / R3-i / R1-HS / R3-s re-run on the grid left after
    the 2L port cores.
  - Cross-device half: a relay all-gather of one fp32 Σx² record (128 B) per tile-row, in waves, driven by port cores
    adapted from `examples/fabric_ring_exchange`, and overlapped with the local work by a non-blocking root event loop.
- **SUPPORTED at Phase 0** (equal to TARGET on every axis):
  - dtype = [float32, bfloat16, bfloat8_b]; fp32_dest_acc_en = [True, False]; layout = [TILE]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]; rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [float32, bfloat16, none]; gamma_layout = [TILE, ROW_MAJOR, none]
  - memory_layout = [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED]
  - cluster_axis = [0, 1]; topology = [Linear, Ring]; num_links = [1, 2, None]
  - EXCLUSIONS = [{float32, fp32_dest_acc_en=False}]
- **Accuracy achieved** (worst device, G = 4 Ring on the 1×4 QuietBox 2 ring, bf16 TILE γ,
  `test_distributed_rms_norm_precision_baseline.py`, 4 shapes × 3 dtypes × 2 DEST modes):
  - bf16, fp32 DEST: PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019, ≤ 1.05 ULP.
  - bf16, 16-bit DEST: PCC ≥ 0.999993, rel_rms ≤ 0.0081, ≤ 5.3 ULP, got/true ratio +0.3 to +0.7 %.
  - fp32: PCC 1.000000, rel_rms ≤ 0.0012.
  - bf8b: PCC ≥ 0.999949, rel_rms ≤ 0.014.
- **Golden suite at Phase 0** (after the verifier fixes; `verifier_report.json`):
  - supported_pass 10,908; supported_fail 133; xpass_drift 0; xfail_wrong_mode 1 (harness interaction);
    xfail_expected 2,160; invalid_skipped 33,696; infeasible_skipped 40.
  - The 133 failing cells are tracked by Refinements 1 (120), 2 (12) and 4 (1).
  - First run on the code as handed over: supported_pass 6,197; supported_fail 3,704; xpass_drift 35;
    xfail_wrong_mode 226.
- **Perf at Phase 0** (`perf_measure.py`, 5 experiments, fastest / slowest chip, µs):
  - 640×1792 (PERF FOCUS) 23.93 / 25.77 (target 11.9);
  - 32×1792 12.03 / 13.61 (target 0.6);
  - 2048×1280 44.10 / 47.41 (target 27.3);
  - exposed transport (op − `DRMS_ABLATE_EXCHANGE`) 2.94 / 1.60 / 10.00.
  - The verifier's fixes are perf-neutral (`ccl_perf compare`: null on all three cases).
- **Issues encountered (verifier fixes)**:
  1. **Kernel bug: ragged block × multiple W chunks × γ.**
     - Cause: `cb_x_gamma` (an `R·C` ring) advanced only `R_b·C` per chunk on cores with fewer rows than `R`. The
       next chunk's indexed reads then ran past the ring end.
     - Effect: an LLK "Indexed tile read exceeds CB boundary" assert and hang under `--dev`, garbage output (PCC ≈ 0)
       otherwise. Exposed by the translated DeepSeek prefill 8192.
     - Fix: nominal push count (filler pages) per chunk on ragged blocks.
     - Test: `test_distributed_rms_norm_debug.py::test_ragged_block_chunked_gamma`, verified failing without the fix.
  2. **Persistent exchange resources pinned L1.**
     - Cause: one landing + semaphore set was cached per port placement (sharded inputs move the ports), each created
       mid-call below the caller's L1 tensors and never freed. Over a long session the CB space ratcheted down.
     - Effect: ~340 golden sharded cells were refused late in the run while passing in a fresh process.
     - Fix: the landing is now allocated per call (an op intermediate) and freed after the dispatch; the global
       semaphores are one grow-only set per mesh over the whole grid. Device time unchanged.
  3. **Long sequences crashed.**
     - Cause / effect: `CBs collide with the landing scratch` assert, or an allocator OOM.
     - Fix: the plan is re-solved against the landing's real bytes, and the design's missing port-landing cap
       (`PORT_LANDING_MAX_BYTES`) is now a clean `NotImplementedError` before allocation. Prefill 32768 now passes;
       131072 is refused (Refinement 4).
  4. **EXCLUSION `{bfloat8_b, w_non_aligned}` removed.** It was carried from the base op. The compute-side W-pad
     mask (partial reduce scaler) now also runs for interleaved Bfp8_b, and 720 golden cells pass.
  5. **`num_links=None` declared in SUPPORTED.** It was always accepted; the registry cell of a call records the
     literal `None`.
  6. **DRY.** `cb_inventory()` is the one CB list behind both the L1 solve and the CB descriptors.
  7. **Docs.** L1 ledger made current (stale `cb_dev_partial` / `Rw` / total rows); `op_design.md` implementation
     notes extended.
- **Tests added**:
  - `test_distributed_rms_norm_precision_baseline.py`;
  - `test_distributed_rms_norm_debug.py::test_ragged_block_chunked_gamma`;
  - existing: `test_distributed_rms_norm.py` (acceptance), `test_distributed_rms_norm_knobs.py`,
    `test_distributed_rms_norm_debug.py::test_repeat_rows`, `perf_ablate.py`.

## Refinement 1 — Port the base op's streaming-γ fallback (L1 fit for wide-row sharded shapes)
- Date: 2026-10-30
- What was done:
  - **Streaming γ (base op's L1 fallback), knob `GAMMA_RESIDENT`.** `Geometry.gamma_resident` sizes `cb_gamma_tiles`
    through `gamma_tiles_pages()` (`Wp` resident / `STREAM_DEPTH·C` streaming) inside `cb_inventory()`, so the L1
    solve and the CB descriptors still read one list. `choose_blocking` runs the existing search with resident γ
    first (every Phase 0 plan is unchanged) and, when nothing fits, the same search with streaming γ.
  - Writer: the γ chunk read is one lambda shared by both paths. Resident pushes the window chunk by chunk as before.
    Streaming pushes chunk 0, then per (block, chunk) step pushes step s+1's chunk before draining output step s
    (sharded: a pure γ push loop). Compute: `scale_by_gamma` waits `C` pages at offset 0 and pops them after the
    chain (resident: cumulative wait at offset `c·C`, as before). `gamma_resident` is a writer / compute CT arg and
    in the program hash.
  - **L1 budget bug (found while landing this).** `l1_budget()` used the largest free block, but static CBs must end
    below the lowest L1 buffer. The persistent exchange semaphores are created on a mesh's first call, so they sit
    just below that call's shards (524160 B under a 512 KiB shard). A later wide-row call planned against a hole
    above them and clashed: "CB region ends at 527744". The fix: the budget is also bounded by the lowest address of
    the call's L1 tensors (input / output / γ / landing) and of the semaphores, and the semaphores are created
    before planning. The bf8b + fp32 γ cells failed only in sequence because of this.
- Accuracy achieved: the golden tolerances (unchanged) on all 120 target cells: (1,1,32,8192) / (128,8192)
  × {bf16 + bf16/fp32 γ, bf8b + fp32 γ}, and fp32 (1,1,32,4096) / (1,32,4096) / (1,1,128,4096) + fp32 γ, all
  HEIGHT_SHARDED, TILE and ROW_MAJOR γ, both axes, 1 and 2 links, Linear and Ring.
- Golden test progress: 120/120 target cells pass (previously `supported_fail` with "does not fit one resident
  tile-row"). A 66-cell slice of prior-passing γ cells passes (INTERLEAVED / WIDTH_SHARDED variants of the same
  shapes, bf16 4096 HS). Expected total: 11,028 supported_pass / 13 supported_fail. The 13 are the 12 placement cells
  for Refinement 2 and DeepSeek prefill 131072 for Refinement 4. No sharded fp32 residue was left to fold in.
- Issues encountered: the order-dependent CB / L1-buffer clash above. No cell needed a structural-limit note: every
  target fits with streaming γ. For example, (1,1,32,8192) bf8b + fp32 γ is 416,128 B of CBs at R=1, C=32.
- Tests added: none. The 120 golden cells pin the path. Unit regression net: acceptance 103/103, debug 6/6,
  knobs 12/12, precision baseline 20 passed / 4 skipped (each file run on its own: running the whole directory in
  one process fails 42 setups on a fixture fabric-config reopen, which is unrelated to the op).

## Refinement 2 — Output placement independent of the input placement
- Date: 2026-10-30
- What was done: split "input sharded" (`g.sharded`: `cb_x_tiles` zero-copy on the input shard) from "output native"
  (`g.out_native`: `cb_output` zero-copy on the output shard, only when the output's placement equals the input's
  shard). Every other placement pair takes the existing streaming `cb_output` (`STREAM_DEPTH·C` pages, added to
  `cb_inventory` / the L1 solve) and the writer's output `TensorAccessor`: sharded in → interleaved out, interleaved in
  → sharded out, sharded in → sharded out on another grid / shard shape (non-local writes). The compute / writer
  `SHARDED` CT arg is now `out_native` (it only ever gated the output path). `same_placement` / `output_is_native`
  moved into the program descriptor. `out_native` is in the program hash (both memory configs already were).
  `validate()` regime refusal (b) is retired; a sharded output `memory_config` only needs a shard spec.
  Reused: the shard-backed input CB, the interleaved writer path, the streaming cb_output. Added: one Geometry flag.
  No kernel logic change.
- Accuracy achieved: the golden tolerances (PCC ≥ 0.9997, atol 0.45, rtol 0.05) on all 16
  `test_simulated_distributed_layernorm` cells (bf16 / bf8b input, 2048 W, WIDTH in → WIDTH out on 8x4 / 4x4@(1,0) /
  8x3 / 4x2 grids); unit placement cases at the acceptance tolerance.
- Golden test progress: 16/16 `test_simulated_distributed_layernorm` (the 12 placement-mismatch targets, previously
  NotImplementedError). Regression slice: `test_golden.py -k "8192 and HEIGHT"` 160 passed (native output +
  streaming γ path); unit acceptance file 111/111.
- Issues encountered: None.
- Tests added: `test_distributed_rms_norm.py::test_output_placement` (WS→DRAM, HS→DRAM, DRAM→WS 8x3, WS→WS regrid
  4x2@(1,0); shapes 32x1792 and 64x1000 W-non-aligned), `run_case(out_memory=...)`.

## Refinement 3 — OVERLAP: pipeline read → exchange → write on the perf-flagged case
- Date: 2026-10-31
- What was done (perf, label **OVERLAP**). Each lever was measured with `perf_measure.py` and kept by
  `ccl_perf compare`. The breakdown came from new measurement-only per-stage zones: `DRMS_ZONES=1` and
  `tests/.../perf_zones.py`. On the focus case they showed each root's collect → finalize → distribute chain running
  serially per wave (1.77 + 1.9 + 1.0 µs).
  1. **Per-entry collect (L4).** `NOTIFY_PER_ENTRY`: the ports notify per landed entry, and the root reads and
     scatters each slot as it lands. The column-0 copies are unrolled, the poll order is downstream first, and the
     root's own inv copy is pushed first. 23.87 → 22.09 µs.
  2. **inv_rms multicast per member row-run (L6).** Linked multicast of faces 0/2 plus a multicast set of
     `SEM_INV`, with no ack barrier. 22.09 → 21.79.
  3. **Non-blocking x load cursor.** One row in flight, pushed once landed. 21.79 → 21.23.
  4. **Wave-ordered reads (`READ_WAVE_GATE`).** 21.23 → 21.05.
  5. **Finalize SFPU on the column faces only.** `VectorMode::C` in-kernel chain elements; a documented raw-LLK
     deviation. 21.05 → 20.51.
  6. **`PORT_OWN_AHEAD`** (the port sends wave w+1's own segment ahead of wave w's relays). Correct, measured null
     twice, parked off as a live knob.
  - Reused: the root event loop, the slot / landing layout, the chain helpers, the program semaphores (the root's own
    `SEM_INV` word as the multicast source). Added: the reader's `entry_slot` and member row-run RT args, three host
    knobs, and no new CB (`l1_ledger.md` is unchanged).
- Perf (`perf_measure.py`, 5 experiments, fastest / slowest chip, µs; all `ccl_perf compare` wins):
  - 640×1792 (PERF FOCUS) 23.95 / 25.75 → **20.51 / 21.77**;
  - 32×1792 12.12 / 13.52 → 9.75 / 10.72;
  - 2048×1280 44.09 / 47.18 → 42.53 / 45.67.
  - Exposed transport (op minus `DRMS_ABLATE_EXCHANGE`, fastest chip): 2.94 → 1.97 / 1.60 → 0.71 / 10.0 → 9.1.
- Measured nulls / reverts:
  - 3 waves (`R3_ROWS_PER_GROUP=3`) 20.96 and 4 waves 24.4: still slower than 2.
  - Reader / writer NoC swap 30.2: removed.
  - Split output writes over both NoCs in `DM_DYNAMIC_NOC` mode: hung `test_linear[1x32x1792-bf16-cluster_axis=0]`
    (writer in `ncrisc_noc_fast_write_any_len`), reverted.
- Remaining bottleneck (finding, not a follow-up). The output write burst after inv_rms: 2.3 MB at about 280 GB/s.
  The east-edge cores' NOC1 writes take about 6.5 µs, against 0.3 µs at the west edge, and they end the op.
  Direction-aware writes on both NoCs are the next lever.
- Accuracy achieved: unchanged math (same ops, same slot order). Precision baseline 20 passed / 4 skipped (bf16 PCC ≥
  0.999998 under fp32 DEST, as at Phase 0); golden tolerances on every cell run.
- Golden test progress:
  - Slice (640×1792, 2×1×128×100, 4×128×47 at 2 links / fp32 DEST, plus 32×8192 HEIGHT_SHARDED at 1 link): 311
    passed, 868 skipped (invalid), 10 xfailed, 0 failed.
  - `test_translated.py`: 278 passed, 1 failed (the known prefill-131072 landing refusal, Refinement 4).
  - The three perf cases are checked against the golden reference in `perf_measure.py`.
- Issues encountered: the dynamic-NoC hang above (reverted). `DeviceZoneScopedN` in one scope twice is a compile
  error, so the zones nest in braces.
- Tests added: `tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_zones.py` (measurement only, not collected).
  Unit regression net: acceptance 111/111, debug 6/6 (incl. `test_repeat_rows`), knobs 12/12, precision 20 + 4 skipped.

## Refinement 4 — Bounded exchange landing for long sequences (TRANSPORT scheme change)
- Date: 2026-10-31
- What was done (label **TRANSPORT**, a scheme change on the landing's flow control):
  - **Predicate.** When the call's full landing (`num_waves·G·seg_stride`, bucketed) fits `LANDING_FULL_MAX_BYTES`
    (= `PORT_LANDING_MAX_BYTES`, 512 KiB), the credit-free layout is kept: `K = num_waves`, and every ring branch is
    `if constexpr`-compiled out. Otherwise the landing is a ring of `K = max(LANDING_RING_WAVES=4, nwpb, 2)` wave slots:
    wave w lives in slot `w mod K`. New CT arg `landing_waves` for the reader and port; K is in the program hash.
  - **Credits on the reverse ready path.** For port D (direction d, link l), port E = opp(d) on the same chip already
    sends the upstream sender U its cross-call ready. E now also sends one credit per consumed wave, as the same
    header-only atomic inc on U's ready counter. U sends wave w ≥ K only when `ready ≥ w+1−K` (after the fence's own
    `−1`), and consumes exactly `num_waves−K` at the end. No new persistent resource. Every global semaphore is
    re-armed to 0 (trace-safe), and the cross-call fence semantics are unchanged.
  - **Wave-freed reports.** These go to E's program semaphore `SEM_PORT_FREED` (id 0, which is unused on port cores):
    +1 per wave from each root of the link (after it collected the wave, or right away for a wave that is empty for
    it), and +1 from D after its relays of that wave are flushed.
  - **Own slot.** A root publishes wave w ≥ K only after it collected `w−K`. In ring mode the port flushes its sends
    before each `SEM_OWN_COUNTED` release.
  - Reused: the ready fence's packet and records (now the `send_ready` lambda), the ready global semaphore, the
    program semaphore set, the slot layout. Added: 4 port RT args, 3 reader RT args, 2 CT args, the
    `LANDING_RING_WAVES` / `LANDING_FULL_MAX_BYTES` knobs, and the `LAST_EXCHANGE_LAYOUT` test hook.
- Accuracy achieved: golden tolerance (PCC ≥ 0.98, DeepSeek translated) on prefill 131072 (per chip 131072×1792
  bf16, RM γ, interleaved, 1×4 Linear axis 1). Unit acceptance tolerance on the forced-ring cases: 16384×256 bf16,
  16384×512 fp32, 4×4096×1792 bf16 (K = 2, 3), and Ring 4×4096×1792 at 1 and 2 links on both axes. Each case runs
  twice back to back to catch re-arm bugs.
- Golden test progress: `test_ds_distributed_norm[random_weights-prefill-131072]` now passes (it was
  NotImplementedError). The other DeepSeek random-weights configs (decode WIDTH_SHARDED, 128, 1024, 8192, 32768) still
  pass. Expected total: 11,029 supported_pass, 0 supported_fail. The real-weights cells skip because the HF
  checkpoint is absent.
- Perf (fast path, `perf_measure.py` 5 experiments, focus 640×1792): 20.49 / 21.81 → 20.44 / 21.79 µs
  (fastest / slowest chip). `ccl_perf compare`: null, so unchanged.
- Issues encountered: None. Unit regression: acceptance 111/111, debug 6/6.
- Tests added: `test_distributed_rms_norm_knobs.py::test_bounded_landing` (6 cases) and `::test_bounded_landing_ring`
  (4 cases). Both force the ring through `LANDING_FULL_MAX_BYTES = 0` and assert `K < num_waves`.

## Refinement 5 — TRANSPORT: the per-call ready fence and the link count on the focus case
- Date: 2026-10-31
- What was done (perf, label **TRANSPORT**). Each lever was measured on the focus case (640×1792, Ring, 2 links)
  with `perf_measure.py` (5 experiments) and `ccl_perf compare`.
  - Baseline: 20.42 / 21.81 µs (fastest / slowest), spread 1.39 µs.
  - Zones (`perf_zones.py`, last traced call). On every chip `prt_ready_wait` ends at 11.7–13.2 µs. The first root
    publish starts at 14.5–15.5 µs and the first send at 16.9–18.1 µs. So the fence is fully hidden behind the
    local reads. The ~5 µs fence of `fabric_ring_exchange` (README result 1) is an isolated exchange with no local
    work to hide it.
  1. **Fence-free steady state.** Bounded by `DRMS_DEFER_FENCE` (new measurement-only port define): the fence wait
     moves after the sends, which is better than any real fence-free scheme. Result 20.46 / 21.74, spread 1.28: null.
     The persistent double-buffered landing was **not built**. Its upper bound is 0, and it would bring back the
     persistent L1 landing the verifier removed (fragmentation risk).
  2. **Links.** `num_links = 1` on the same case gives 20.56 / 22.47 (spread 1.91): reject, slowest +0.66. Two links
     halve each port's relay and notify chain, which matters more than the fence in the op. "One fence per port
     pair" is bounded by lever 1 (null).
  3. **Port placement (L8).** New host knob `PORT_X0`, env `DRMS_PORT_X0`; the default `None` = centred, unchanged.
     - x0=2 (west-shifted): 20.45 / 21.66, null.
     - East edge: 20.42 / 21.96, reject (slowest +0.15).
     - x0=0: not measurable. The port cores' BRISC profiler markers overflow, and the harness then sees 539/600
       programs.
     Probe-based placement under the Ethernet cores was not pursued: the payload is latency-bound, not bandwidth-bound.
  - Reused: the port kernel's fence code (now a `ready_fence` lambda, called at the same point unless the define
    is set) and `place_ports`. Added: the `DRMS_DEFER_FENCE` define and the `PORT_X0` knob.
- Finding (not a follow-up): the ~2 µs exposed transport is the hop-latency chain (send → relay → notify → collect),
  not synchronization. The op's end is set by the output write burst (Refinement 3's finding).
- Accuracy achieved: unchanged. Production kernels and plans are byte-identical in behaviour (defaults).
- Golden test progress: unchanged. `perf_measure.py` checks the focus case against the golden reference on every
  variant run.
- Issues encountered: `generated/profiler/.logs/zone_src_locations.log` had built up 16-bit zone-hash collisions
  across builds; deleting it fixed `perf_zones.py` (`Source location hashes are colliding`).
- Tests added: none. Unit regression: acceptance 111/111, debug 6/6 (including `test_repeat_rows`), knobs 22/22.

## Refinement 6 — Local-half headroom on the perf-flagged case (only if Refinements 3 and 5 leave it)
- Date: 2026-10-31
- What was done (perf, local half; the focus case 640×1792 Ring 2 links). Re-measured first: 20.46 / 21.74 µs
  (fastest / slowest), exchange ablated 18.47, well above the base op's ~12.6 µs efficiency, so the refinement ran.
  - Breakdown. New measurement hooks: `DRMS_ABLATE_WRITE` (writes stubbed: 16.56) and `DRMS_ABLATE_FINALIZE` (root
    finalize math stubbed: 19.65; the slot fold alone 19.57, the rsqrt alone 19.93). Zones: the reads take ~5 µs;
    the first inv_rms reaches the members at ~23.9 µs; then 2.3 MB of output is written in ~7.5–8 µs. That is the
    DRAM write ceiling on this box (~285 GB/s; the base op measured 278 GB/s write-only). So the write burst cannot
    shrink, and what moves the end is how early the first wave's inv_rms lands.
  1. **One-pass finalize fold** (`FINALIZE_ONE_PASS`, default on): G copies into D0..D(G−1), then one column-face
     SFPU pass with the same left-fold adds (bit-identical sums), instead of G−1 copy / add pairs that re-init the
     math MOP and the SFPU per element. A documented raw-LLK in-kernel chain element (`SumSlotsCol`), like the
     Refinement 3 column elements. 20.48 / 21.82 → 20.33 / 21.55 (win).
  2. **Output on both NoCs** (`OUT_NOC_SPLIT`, default `alternate`). The writer posts each output quantum on local
     program semaphores and writes its NOC1 tiles. The reader writes the NOC0 share (odd DRAM banks) from its poll
     loop, and the writer pops after the reader reports it flushed, so the writer stays cb_output's only consumer.
     Direction-aware masks (`direction`, by the bank's DRAM column against the core's physical x) were slower:
     21.09, and 21.66 flipped. They flatten the east/west gradient, but the DRAM-bound end stays. `alternate` was
     null alone and a win on top of (1): 20.30 / 21.60 → 20.16 / 21.31, 3 repeats each. It regressed decode
     (114 KB output, +0.23 µs), so it is path-gated to ≥ `OUT_NOC_SPLIT_MIN_BYTES` (512 KiB) of output per chip.
  - Not built: G-MC (γ is the row-0 prefix of 6 tiles per core, < 1 % of the bytes; the base measured it at noise
    on the 640-row shape).
  - Reused: the writer's quantum loop, the reader's non-blocking poll loop (one new cursor), the chain element pattern
    of Refinement 3, the program semaphore set. Added: 3 compute-core program semaphores; 6 + 1 reader CT args (plus
    the output accessor) and 6 writer CT args; 1 reader RT arg (`out_addr`); 1 compute CT arg; the knobs
    `OUT_NOC_SPLIT`, `OUT_NOC_SPLIT_MIN_BYTES`, `FINALIZE_ONE_PASS`. No CB change (`l1_ledger.md` notes the reader's
    read-only access to cb_output).
- Perf (`perf_measure.py`, 5 experiments, `CCL_PERF_CASES=all`, old knobs vs new on the same build; all
  `ccl_perf compare` wins):
  - 640×1792 (PERF FOCUS): 20.50 / 21.82 → **20.18 / 21.33**, spread 1.15;
  - 32×1792: 9.78 / 10.70 → 9.69 / 10.60;
  - 2048×1280: 42.67 / 45.44 → **40.93 / 43.66**.
  - Exposed transport at the focus case: 1.97 → 1.68 µs (exchange ablated 18.50).
- Accuracy achieved: unchanged math. The fold is bit-identical (same SFPU adds in the same order). The output split
  only changes which NoC carries a tile. Precision baseline: 20 passed / 4 skipped (as before).
  `test_repeat_rows` bit-identical repeats pass.
- Golden test progress:
  - Slice (640×768 TILE interleaved, 4×512×512 fp32 interleaved, 2048×2048 bf8b interleaved, 64×1000
    WIDTH_SHARDED): 292 passed, 896 skipped (invalid), 60 xfailed, 0 failed.
  - `test_translated.py` random-weights (decode … prefill-131072): 6 / 6.
  - The three perf cases are checked against the golden reference in `perf_measure.py`.
- Issues encountered: `(a | b) != 0` on two CT constants is a `-Werror=tautological-compare` error. The custom sfpi
  function must sit under `#ifdef TRISC_MATH`. A chain element template argument cannot name a function-local
  `constexpr` from a lambda (made `static constexpr`).
- Tests added: `test_distributed_rms_norm_knobs.py::test_out_noc_split`. It covers 12 cases: every mask mode, the
  split forced below its gate on decode, ragged-W, bf8b R1 with the 2-tile fold, and fp32 multi-block, plus a
  HEIGHT_SHARDED input → DRAM output call. Each is run twice back to back. Unit regression net: acceptance 111/111,
  debug 6/6, knobs 34/34, precision 20 + 4 skipped.
- Finding (remaining headroom, not a follow-up): the focus case is now read (~5 µs at ~450 GB/s) → first-wave
  chain (~7 µs: in-device combine, publish, ~1.7 µs exposed fabric hops, collect, finalize, distribute) → write
  burst at the DRAM write ceiling (~8 µs). DRAM is idle during the chain. The next lever is to overlap the write
  burst with the reads, since mixed traffic exceeds the write-only ceiling (the base measured ~430 GB/s mixed). That
  needs the first wave's inv_rms while later rows are still being read, i.e. a cheaper per-wave chain so that more
  waves pay off (L1 + L2; 3–4 waves were slower in Refinement 3). That is an OVERLAP scheme change, not a local knob.

## Perf 1 — perf tournament, round 1
- Date: 2026-10-31
- Focus case: `LOOSE_CASES[0]`, the PERF FOCUS config exactly as declared. Per chip (1,1,640,1792) bf16 TILE,
  interleaved, fp32_dest_acc_en=True, bf16 TILE γ, Ring, cluster_axis=1, num_links=2, on the 1×4 QuietBox 2 ring.
  Every knob of it is in SUPPORTED.
- **Instrumentation.** `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in
  this tree. The op's permanent stage zones are its own `DRMS_ZONE(...)` macro: a `DeviceZoneScopedN` under the host
  define `DRMS_ZONES=1`, empty otherwise. They are reused and still cover every changed stage (`rt_collect`, `cp_fin`,
  `rt_distribute`).

### Measured breakdown (before)
- **Harness** (`perf_measure.py`, 5 experiments, fastest / slowest chip, µs):
  - 640×1792: 20.168 / 21.332;
  - 32×1792: 9.693 / 10.591;
  - 2048×1280: 40.879 / 43.699.
  - Not dispatch-bound.
- **Cumulative ablation**, fastest chip, µs: full 20.17 → exchange stubbed 18.50 → + writes stubbed 16.12 → + finalize
  math stubbed 14.82. Writes stubbed alone: 16.72.
- **Zone timeline**, µs from kernel start, every chip similar:
  - x of wave 0 lands at 2.2–3.2, wave 1 at 4.4–6.3.
  - Root publish: wave 0 at 5.3–7.5; wave 1 at up to 9.6–9.9.
  - The ports' last notify (every fabric entry landed) is at 11.5–13.1.
  - inv_rms distribute: wave 0 at 12.3–13.4, wave 1 at 15.5–18.8.
  - The output write burst (2.3 MB, at the DRAM write ceiling) runs from 12.3 to 21.2.
- **Ranked bottleneck.**
  1. **The post-exchange chain: 2.5–5.5 µs per wave after the last landing.** The root reader pulls each peer slot
     with a serialized 128 B read + barrier (0.35–0.45 µs each, one of 2.8 µs under the write burst). The root
     finalize takes ~1.2 µs per tile-row, and the distribute ~0.6 µs.
  2. The chain before the exchange: land → publish ~2.5 µs, then the port waits ~1.5 µs for the link's slowest root.
  3. The relay hops: ~3 µs per wave from first send to last landing. The transport itself is small (+0.3 µs data on
     top of the fence, `fabric_ring_exchange`), so these hops are skew and turnaround, not bandwidth.
  4. The DRAM write burst: ~8.5 µs at the ceiling. It cannot shrink; it can only start earlier.
  - The DRAM ceiling gates nothing in 1–2: DRAM sits idle between the end of the reads (~5.5) and the first
    distribute (~12.3).

### Portfolio (hard cap: 2 experiments)
1. **push_landed_records** (OVERLAP, post-exchange chain): the ports push each landed entry's per-root records into
   the L1 of every consuming core, and every core finalizes for itself. No root collect reads, no distribute.
2. **row_layout_finalize** (local compute on the post-exchange chain): a compact finalize tile holding every slot
   record, one fused sfpi pass, and the root re-lays inv_rms into column 0.

Floated but not tested (cap): per-root exchange items (skew tolerance, item 2 above), fabric multicast instead of
relay (TRANSPORT, item 3), and fast multi-row reads in flight.

### Verdicts
- **push_landed_records: REGRESSION. Not graduated.** Bit-identical outputs. Variants, fastest / slowest µs on the
  focus case (base 20.168 / 21.332):
  - grid push from the fabric sender: 23.50 / 31.53;
  - separate pusher on the port's idle RISC: 19.67 / 23.44 (3 rectangles), 19.41 / 24.50 (one whole-grid rectangle),
    20.01 / 25.36 (non-blocking);
  - per-group multicast plus every core finalizing: 31.55 / 47.79;
  - roots only, multicast: 27.19 / 41.17;
  - roots only, unicast: 20.38 / 21.95;
  - pusher on NoC1: hung, not debugged.
  - Cause: once the output write burst starts, every push from the port stalls 2–4 µs. The fan-out from one port is
    slower than 10 roots reading in parallel, and the chip spread grows from 1.2 to ~5 µs.
  - Decode (32×1792) won −0.16 to −0.42 µs on the fastest chip. 2048×1280 regressed +0.34 to +0.51 µs.
  - Not kept as a decode-only path: the measured regressions cover the focus and prefill cases, so the gate would be
    an allow-list around one shape plus a second post-exchange protocol.
  - Artifacts: `perf_experiments/push_landed_records/`.
- **row_layout_finalize: WIN. Graduated.**
  - Finalize micro-bench, ns per tile-row (G=4, fp32 DEST, HiFi4, approx off):

    | Variant | ns |
    |---|---|
    | column chain (base) | 1161 |
    | column layout, fused pass | 814 |
    | 4 tiles, row layout | 602 |
    | **packed**: 1 tile, 2 vectors | **300** |
    | dense: 1 tile, 1 vector | 264 |

    The floors are 509 ns (copy + pack of 4 tiles) and 220 ns (1 tile). The ranking holds for 16-bit DEST
    (983 → 246 ns) and G=2.
  - In the op copy, fastest / slowest µs on the focus case:
    - column layout, fused pass: 19.85 / 21.04;
    - dense: 19.29 / 20.62;
    - packed: 19.16 / 20.60;
    - **packed_direct**: 18.96 / 20.50 (packed, with each fp32 record NoC-read straight into its tile rows).

    Dense loses to packed in the op: the reader pays strided stores with runtime indices for it.
  - Every variant is bit-identical to the real op: outputs compared bitwise on 18 cases (three perf shapes, Linear,
    1 link, fp32 DEST off, fp32, bf8b, no γ, HEIGHT / WIDTH sharded, masked width, odd shapes).
  - Artifacts: `perf_experiments/row_layout_finalize/` (`bench/`, `op/`, `test_slice.py`).

### Graduated: compact finalize tile (packed_direct) as the op's one finalize path
- **Domain.** Every regime (R3-i, R1, R1-HS, R3-s), dtype, DEST mode and placement with G ≤ 16, which is every
  supported mesh. The fp32 direct read and the Float16_b word-by-word copy are the two record datapaths of one layout.
- **One carve-out, for feasibility.** With G > 16, 16 slots × 32 words no longer fit one tile, so the code keeps the
  column-0 layout and the Refinement 3 column chain with the 2-tile fold. It is written as
  `if (cannot) { legacy } else { new }`: host `FIN_COMPACT_MAX_G`, a CT arg to reader (after the accessors) and compute
  (index 14). Forcing it to 0 keeps the legacy path tested (`test_out_noc_split[...-fold]`).
- **Deleted.**
  - `FINALIZE_ONE_PASS` (knob, CT arg, hash field, env `DRMS_FINALIZE_ONE_PASS`) and the `SumSlotsCol` one-pass fold
    (Refinement 6, L11), now superseded.
  - The experiment's dense / col / packed variants and its `FIN_LAYOUT` knob.
- **CBs.** `cb_stats_gather` goes from `G·Rw` to `Rw` pages. R1 roots gain `cb_rms_root` (`l1_ledger.md`).
- **Whole op** (`perf_measure.py`, 5 experiments, `CCL_PERF_CASES=all`, fastest / slowest µs; each `ccl_perf compare`
  vs the round's baseline is a win):

  | Case | Before | After | Δ fastest / slowest |
  |---|---|---|---|
  | 640×1792 (PERF FOCUS) | 20.168 / 21.332 | **18.958 / 20.511** | −1.210 / −0.821 |
  | 32×1792 | 9.693 / 10.591 | **8.616 / 9.838** | −1.077 / −0.753 |
  | 2048×1280 | 40.879 / 43.699 | **40.138 / 42.886** | −0.741 / −0.812 |

- **Zones after** (focus case): wave-1 inv_rms is distributed at 15.8–17.0 µs (was 17.9–18.8), and collect slot reads
  take ~0.22 µs (were 0.35–0.45).
- **Guard set / no regression:**
  - Golden `test_golden.py`: 10,626 passed, 0 failed (INTERLEAVED 3603, HEIGHT_SHARDED 3420, WIDTH_SHARDED 3600, loose
    3; the xfails are the expected ones).
  - `test_ring_mock.py` + `test_regression.py`: 36 passed.
  - `test_translated.py`: 279 passed, 51 skipped.
  - Unit tests: knobs 34/34 (including the forced G > 16 path), acceptance 111/111, debug 6/6, precision 20 passed /
    4 skipped.
  - Precision: inv_rms is bit-identical, with the same SFPU ops in the same order. The precision contract is
    untouched.
- **Remaining headroom.** The focus case is at 18.96 µs against the target of 11.9. Next round's candidates are the
  chain before the exchange (port waits for the link's slowest root; per-root items) and the relay hops. The write
  burst still starts only at wave-0 distribute (~11.4–12.3 µs).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| kernel_lib eltwise chain SFPU elements `AddBinary` + `MulUnary` + `AddUnary` + `Rsqrt` (finalize) | capability | Every element runs `VectorMode::RC` over a whole tile, one op and one SFPU init per element. None can (a) fold N operands at arbitrary DEST offsets *inside one tile*, (b) restrict the work to a chosen subset of SFPU vectors (here 2 of 64), or (c) chain several SFPU ops under one init with a DEST store / reload between stages. `FinalizeFused` is a hand-written sfpi pass for this. | 1161 (column chain, already a raw `VectorMode::C` bypass since Refinement 3) | 300 (micro-bench, per tile-row) | `kernels/distributed_rms_norm_compute.cpp:206` (`FinalizeFused`), `:121` (`drms_finalize_vec`) |
| no helper covers this: dataflow record ↔ tile layout moves (would sit with the `local_copy` / tile-index dataflow helpers) | capability | No helper scatters a contiguous record into tile face rows, re-lays a tile's row-run into column 0, or NoC-reads straight into a face-row offset of a CB page. The reader does these as raw word moves and a raw `noc_async_read`. | — | — | `kernels/distributed_rms_norm_reader.cpp:220` (`record_to_tile_slot`), `:248` (`tile_slot0_to_col0`), `:611` (direct read in `read_slot`) |

Summary: both ideas were measured. 1 graduated, 0 null, 1 regression (not graduated). The op is 1.21 µs faster on the
focus case (fastest chip), and faster on all three perf cases, with no regression.

## Perf 2 — perf tournament, round 2
- Date: 2026-10-31
- Focus case: `LOOSE_CASES[0]` exactly as declared. Per chip (1,1,640,1792) bf16 TILE, interleaved,
  fp32_dest_acc_en=True, HiFi4, bf16 TILE γ, Ring, cluster_axis=1, num_links=2, on the 1×4 QuietBox 2 ring. Every
  knob is in SUPPORTED. SUPPORTED is unchanged.
- **Instrumentation.** `perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) still does not exist in this tree. The op's
  permanent `DRMS_ZONE` / `DeviceZoneScopedN` zones are reused; they cover every changed stage (`prt_send`,
  `prt_ready_wait`, `rd_load`, `rd_land`, `rt_publish`, `rt_collect`, `rt_distribute`, `wr_store`).

### Measured breakdown (before; after Perf 1)
- **Harness** (`perf_measure.py`, 5 experiments, fastest / slowest chip, µs):
  - 640×1792: 18.970 / 20.481;
  - 32×1792: 8.607 / 9.849;
  - 2048×1280: 40.052 / 42.993.
  - Not dispatch-bound.
- **Cumulative ablation** (fastest chip, µs):

  | Step | µs |
  |---|---|
  | full op | 18.97 |
  | exchange stubbed | 17.48 (exposed transport 1.49) |
  | + writes stubbed | 13.84 |
  | + finalize math stubbed | 13.80 |
  | writes stubbed alone | 14.79 |

- **Wave-0 chain** (chip 0, µs from kernel start, zones on):
  1. x lands 2.2–3.9 per root (member reads 1.3–4.2).
  2. Reduce +0.8.
  3. Root gather 4.8–6.4 (waits for the slowest member).
  4. Publish ends 5.8–7.2.
  5. The ports send their own slots at 7.0–7.6 (each waits for the slowest root of its link).
  6. Peer slots are collected at ~8.6, ~9.4 and ~9.9. The last is the 2-hop relayed one.
  7. inv_rms is distributed at 10.5–11.7.
  8. The output write burst runs 11.8–19 at the DRAM write ceiling (~320 GB/s).
  - The x reads end at ~7.0, so DRAM is idle for ~4.8 µs.
- **Ranked bottleneck.**
  1. The first wave's chain (~7 µs from x landed to inv_rms) gates the start of the write burst:
     - exchange ~2.5–3 µs: relay hop 2 lands last, and each chip's 2-hop entry also waits for its neighbour's publish;
     - pre-exchange ~2.5 µs: gather skew, publish, port waits for the slowest root;
     - post-exchange ~1.2 µs.
  2. The write burst is at the write ceiling (`/perf-ceiling-dm`). Its throughput is not a target: it can only start
     earlier, or overlap the reads (mixed traffic beats write-only).

### Portfolio (hard cap: 2 experiments)
1. **direct_exchange** (TRANSPORT): direct delivery replaces the store-and-forward relay. The candidates were a fabric
   line multicast, or one multi-hop unicast per destination. The fence penalty measured in isolation should not apply
   in the op, because the fence is hidden there.
2. **early_small_wave** (OVERLAP): make the first wave small and early, so its inv_rms arrives while later x is still
   being read and the write burst overlaps the remaining reads.

Floated but not tested (cap): per-root exchange items (skew tolerance in the pre-exchange chain), and a cheaper
publish (word-copy + barrier ~0.5 µs).

### Verdicts
**direct_exchange: WIN. Graduated.** Bit-identical on 17 cells:
- the three perf shapes, num_links=1, fp32, bf8b and 64×1000;
- the bounded landing ring ×3;
- Linear, including a bounded case;
- cluster_axis=0 on a 4×1 mesh, Ring and Linear;
- G=2 Ring and Linear.

Variants (fastest / slowest µs, vs 18.97 / 20.47 · 8.61 / 9.85 · 40.06 / 42.99):

| Variant | 640×1792 | 32×1792 | 2048×1280 | Verdict |
|---|---|---|---|---|
| direct, farthest first | **18.81 / 19.78** | **8.60 / 8.90** | **39.76 / 41.72** | win · null fastest, slowest −0.95 · win |
| direct, nearest first | 18.91 / 20.13 | 8.66 / 9.67 | 39.49 / 42.69 | smaller |
| line multicast (hop range ≥ 2) | hangs on FABRIC_2D_TORUS_XY 1×4 | — | — | BLOCKED |

- **num_links=1:**
  - 640×1792: 19.25 / 21.26 → 18.93 / 20.25;
  - 2048×1280: 40.85 / 45.72 → 40.17 / 43.11.
- **Mechanism.** The relay turnaround itself is only ~0.3 µs. The larger effect is that a chip's 2-hop entry no
  longer waits for its neighbour's own publish plus a relay, so skew no longer compounds around the ring and the chip
  spread shrinks by 0.7–1.3 µs. The 2-destination fence still ends ~2 µs before the first publish.
- **Domain:** every topology the fabric routes on the op's ports.
  - Inexpressible: a Linear group on a torus fabric (the table routes 2-hop destinations over the wrap). The plan
    falls back to relay there; this is detected, not listed.
  - G=2: identical to relay.
- Artifacts: `perf_experiments/direct_exchange/`.

**early_small_wave: WIN. Graduated.** Bit-identical on 22 slice cases:
- the three perf shapes, 1 link, Linear, bf16 DEST, fp32, bf8b, no γ;
- HEIGHT- and WIDTH-sharded inputs, masked width, ragged and batched shapes;
- the bounded landing ring on the focus and R1 shapes.

Variants alone (fastest / slowest µs; same-session real op 18.97 / 20.49 · 8.62 / 9.83 · 40.04 / 42.99):

| Variant | 640×1792 | 32×1792 | 2048×1280 | Note |
|---|---|---|---|---|
| **rule: thirds, read start at 300 GB/s** | **18.36 / 19.20** | 8.61 / 9.86 (rule off) | **34.48 / 38.57** | |
| thirds + TRID read isolation | 18.37 / 19.24 | 8.68 / 9.93 | 34.60 / 38.36 | decode slowest regressed |
| split 1:3:1 + TRID | 18.13 / 19.29 | 8.73 / 9.92 | 35.95 / 39.34 | decode slowest regressed |
| TRID alone | 18.73 / 20.12 | 8.65 / 10.01 | null | |
| `2,8` groups, 600 ns/row deferral | 19.23 | – | – | regression |
| 1-row groups, first wave 2 tile-rows | 21.0–23.5 | – | – | not bit-identical (different W slicing) |
| same schedules, no read deferral | 19.4–20.8 | – | 43.2 | regression |

- 3 offset classes beat 2 and 4. On R1, 1:1:1:1 gave 36.4 and 1:1:1:1:1:1 gave 41.8.
- **Read-rate calibration** (2048×1280, fastest chip):
  - 250 GB/s → 36.6;
  - 300 GB/s → 34.5;
  - 382 GB/s → 36.7;
  - 500 GB/s → 40.6.
  - The focus case is flat from 250 to 382 GB/s.
- **Domain sweep** (rule vs real, fastest / slowest µs). Wins:
  - 640×1792 −0.60 / −1.28;
  - 896×1792 −2.31 / −2.68;
  - 1280×1792 −3.62 / −2.95;
  - 4096×1792 −5.05 / −7.29;
  - fp32 640×1792 −3.58 / −4.34;
  - Linear 2048×1280 −4.86 / −3.56;
  - bf8b 2048×1280 −0.45 / −0.75.
  - Cells below 2.2 MB stay on the rule-off path, as listed under Carve-outs below.
- Artifacts: `perf_experiments/early_small_wave/` (`esw.diff` is the full experiment diff).

### Graduated: direct exchange + early small wave, together
**Direct exchange is the op's one exchange schedule.**
- `exchange_plan` (host) builds direct sends, falling back to relay only when a destination's route leaves through no
  port.
- The port kernel sends several own items per wave, farthest first, and counts readies and credits per destination
  chip. Arrival counters are indexed by slot (`get_semaphores(mesh, G + 1)`).
- Deleted: the experiment's multicast route, `relay_mc1`, the `DRMS_MC_PLAIN` debug path and the `EXCHANGE_MODE` /
  `DIRECT_FAR_FIRST` knobs.
- The relay item model stays only as the inexpressible fallback.

**Early small wave** is applied to every cell where it is correct and measured not to regress. It is host
`esw_offsets` plus the reader's `DRMS_ESW_OFFSETS` define and RT args `wave_off` / `read_start`; per-wave port segment
bytes are trimmed to the roots that publish. The early wave is the default, and the cells it skips are written as
carve-outs (`if cannot: offsets 0`).

**Carve-outs, each earned by a measured regression** (offsets forced on):
1. **Sharded input.** It has no reads to reorder. HEIGHT_SHARDED regressed +4.9 µs at 640×1792 and +8.7 µs at
   2048×1280.
2. **x below `ESW_MIN_BYTES` = 2.2 MB per chip.** The extra exchange rounds cost more than the earlier first wave saves:

   | x per chip | 0.46 MB | 0.9 MB | 1.64 MB | 1.84 MB | 1.97 MB |
   |---|---|---|---|---|---|
   | fastest chip | +0.7 | +1.1 | +2.1 | +1.6 to +1.7 | +0.8 |

   The break-even is ~2.1 MB.
3. **Fewer than 3 row-groups.** The 3 offset classes need 3 groups to exist.

**Inside the rule** (re-measured on the combined op, rule on vs off via `DRMS_ESW_MIN_BYTES`):
- **num_links=1, 640×1792: win.** 18.94 / 20.23 → 18.32 / 19.38. The experiment alone had measured +0.47 on the
  fastest chip over relay; the direct exchange removed it.
- **Linear, 640×1792 (relay fallback on the torus mock): mixed, no carve-out.**
  - 19.39 / 22.69 → 19.66 / 20.95: fastest chip +0.27 (noise 0.20), slowest chip −1.75.
  - `compare` calls it a regression on the fastest chip, but the call completes 1.75 µs earlier.
  - A carve-out keyed on "relay fallback" would encode a fabric-mock artifact.
  - Recorded as a measured trade-off; the unified path is kept.

**Whole op** (`perf_measure.py`, 5 experiments, `CCL_PERF_CASES=all`, fastest / slowest µs; `ccl_perf compare` vs
the round's baseline):

| Case | Before | After | Δ fastest / slowest | Verdict |
|---|---|---|---|---|
| 640×1792 (PERF FOCUS) | 18.970 / 20.481 | **17.751 / 18.591** | −1.219 / −1.889 | win |
| 32×1792 | 8.607 / 9.849 | **8.603 / 8.898** | −0.004 / −0.952 | null (fastest within noise; slowest −0.95) |
| 2048×1280 | 40.052 / 42.993 | **33.960 / 38.585** | −6.092 / −4.408 | win |

On the focus case the combination beats either idea alone (18.81 and 18.36).

**Zones after** (focus case, zones on):
- The first inv_rms is distributed at 8.6–9.0 µs (was 10.5–11.0).
- The first wave's writes end from ~10.1 µs (was ~11.8).
- Chip spans are 18.5–18.8 µs (were 19.2–19.9).

### Guard set / no regression
- **Golden `test_golden.py`: 10,626 passed, 0 failed.**
  - INTERLEAVED (including the 3 loose cases): 3603;
  - HEIGHT_SHARDED: 3420;
  - WIDTH_SHARDED: 3600;
  - the xfails are the expected ones.
- `test_ring_mock.py` + `test_regression.py`: 36 passed.
- `test_translated.py`: 279 passed, 51 skipped.
- Unit tests: acceptance 111/111, knobs 34/34, debug 6/6, precision 20 passed / 4 skipped.
- **Sharded guard** (only the direct exchange applies there), 640×1792, real op before → after, fastest / slowest µs:
  - HEIGHT_SHARDED: 14.934 / 16.690 → 14.751 / 15.016;
  - WIDTH_SHARDED: 79.082 / 79.527 → 78.205 / 78.444.
- **Precision:** same math, same slot fold order. Outputs are bit-identical (both experiments' slices), and the
  precision contract is untouched.

### Remaining headroom
- The focus case is at 17.75 µs; the target is 11.9 and the roofline 9.0.
- DRAM still idles ~2.5 µs before the first write.
- The pre-exchange chain (gather skew + publish + the port waiting for the link's slowest root) is the next lever.
- `ESW_READ_GBPS` = 300 is a Blackhole QuietBox 2 calibration; re-measure it on other boxes.
- Observation (not in scope): WIDTH_SHARDED 640×1792 runs ~78 µs, 4× the interleaved case.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this: a timed start relative to kernel launch (would belong with the dataflow scheduling / `noc` helpers) | capability | No dataflow helper can hold back a core's first DRAM read until N cycles after kernel start. The reader reads `RISCV_DEBUG_REG_WALL_CLOCK_L` raw and polls it non-blockingly inside its event loop. | — (no helper path exists) | whole op 640×1792 18.97 → 17.75 µs (with direct exchange) | `kernels/distributed_rms_norm_reader.cpp:89` (`wall_clock_l`), `:477` (read-start gate) |

The direct exchange extends the op's existing raw fabric sender bypass (`port.cpp`, justified since Phase 0: the
kernel_lib CCL helper has no relay-with-arrival-counter item model) with several own items per port. No new helper is
bypassed.

**Summary.** Both ideas were measured; 2 graduated, 0 null. Line multicast was BLOCKED (hang), and TRID read isolation
was not graduated (decode's slowest chip regressed).
- The op is 1.22 µs faster on the focus case (fastest chip; −1.89 µs slowest) and 6.09 µs faster on 2048×1280.
- Decode is flat on the fastest chip and −0.95 µs on the slowest.
- Golden is green, with no regression beyond the one recorded Linear trade-off.

## Exchange modes (manual change, 2026-11-02)
- **What was done.** The Σx² exchange schedule is now selectable: `distributed_rms_norm(..., exchange_mode=...)`, else
  the module default `EXCHANGE_MODE` (env `DRMS_EXCHANGE_MODE`), default `"direct"`. It is not a registry axis: every
  mode gives a bit-identical output (the fold order is fixed by slot).
  - `"direct"`: Perf 2's schedule, unchanged (multi-hop unicast per destination chip, farthest first; the relay fallback
    where the fabric routes a destination off the op's ports).
  - `"relay"`: the balanced store-and-forward schedule (ring: G//2 slots forward, G-1-G//2 backward; one-hop sends).
  - `"mcast"`: hemispheres. Every port sends only its own record as one fabric line multicast over the chips it serves
    (ring: G//2 forward, G-1-G//2 backward; line: every chip downstream), and every receiving port returns its readies
    (and bounded-landing credits) as one header-only multicast over its sources' hop range. 1D fabrics only:
    `validate()` refuses it on a 2D fabric (a 2D line multicast over the bent 1x4 torus mock hangs, Perf 2).
- **Fix found on the way.** Packet headers carried `hops = 1` for every destination, so on a 1D fabric (routing by
  hop count) the direct schedule's 2-hop sends stopped one hop short. Headers now carry the real hop count along the
  port's direction (2D fabrics route by node and ignore it).
- **Code.** Host: `exchange_plan(groups, ring, mesh, mode)` builds the three schedules; port headers are
  `(destination, mcast code, hops)` (4 RT words); the built mode is in the program hash and in
  `LAST_EXCHANGE_LAYOUT["exchange_mode"]`. Port kernel: `route_to` takes the mcast word (2D:
  `fabric_set_mcast_route`; 1D: `to_chip_multicast`).
- **Measured** (`perf_measure.py`, 2 experiments, 640x1792, FABRIC_2D_TORUS_XY, fastest / slowest µs): direct
  17.76 / 18.60 (unchanged), relay 18.37 / 19.19. mcast runs on 1D fabrics only, where the harness does not run yet.
- **Tests.** New `test_distributed_rms_norm_exchange_modes.py` (21 cases): torus2d Ring (direct, relay; mcast refused),
  FABRIC_1D_RING Ring and FABRIC_1D Linear (all three), shapes 640x1792 / 32x1792 / 2048x1280 at 1 and 2 links plus a
  forced bounded landing, each call twice, every mode bit-identical to direct: 21 passed. Regression: acceptance
  111/111, debug 6/6, knobs 34/34, precision 20 passed / 4 skipped; golden test_golden.py 10,623 passed (INTERLEAVED
  3,603, HEIGHT_SHARDED 3,420, WIDTH_SHARDED 3,600), translated + regression + ring mock 315 passed / 51 skipped.
