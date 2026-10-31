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
