# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-18
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` dispatch per call with two built regimes chosen by `select_blocking()`:
  - **LOCAL**: tile-rows split across the full grid, whole row resident per core.
  - **W_SPLIT**: lines of `w_split` cores share an R-range; Σx² partials are gathered to the line root, and rinv is multicast back via `ttnn.Mcast`.

  TILE and ROW_MAJOR are handled natively (tilize/untilize in the kernel). Non-aligned H/W are handled in the kernel (W-pad lanes zeroed in L1, logical `1/W` divisor).
- **SUPPORTED at Phase 0**:
  - dtype=[float32, bfloat16]; fp32_dest_acc_en=[True]; layout=[TILE, ROW_MAJOR]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned]; rank=[2, 3, 4]
  - gamma_mode=[gamma, no_gamma]; gamma_dtype=[float32, bfloat16, "none"]; gamma_layout=[TILE, ROW_MAJOR, "none"]
  - memory_layout=[INTERLEAVED]
  - EXCLUSIONS: {float32, fp32_dest_acc_en=False} (permanent)
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`; 4 shapes × {bf16, fp32} × {TILE, RM}, HiFi4, fp32 DEST):
  - bf16: PCC ≥ 0.999995, max_abs_err ≤ 0.088, rel_rms_err ≤ 0.0036, max 2 ULP
  - fp32: PCC ≥ 0.9999997, max_abs_err ≤ 0.028, rel_rms_err ≤ 0.0017
- **Golden suite at Phase 0** (per `verifier_report.json`):
  - 659 supported_pass, 5566 xfail_expected, 33900 invalid_skipped, 2 infeasible_skipped
  - supported_fail = xpass_drift = xfail_wrong_mode = 0
  - The pytest runner itself reported 672 passed, 0 failed, 0 hangs (`golden_results.txt`).
- **Issues encountered**: no drift fixes were needed. The verifier's code-review change: removed two unused compile-time args (reader `t_f`, writer `OUT_RM_DEPTH`) and renumbered the CT-arg indices to match.
- **Tests added**: test_rms_norm.py (acceptance, 110 cases; planner), test_rms_norm_perf.py (implementer), test_rms_norm_precision_baseline.py (verifier, 16 cases)

## Refinement 1 — Numerical configurability: fp32_dest_acc_en=False + bfloat8_b (input and γ)
- Date: 2026-11-18
- What was done:
  - **SUPPORTED** now includes `fp32_dest_acc_en` = [True, False], `dtype` += bfloat8_b and `gamma_dtype` += bfloat8_b. EXCLUSIONS keeps `{float32, False}` (permanent).
  - **Intermediate format** (ledger audit 3): `create_program_descriptor` makes one decision, `int_fmt = float32 if fp32_dest_acc_en else bfloat16`, with `t_int = tile_size(int_fmt)`. That single value feeds:
    - the six intermediate CBs: `cb_sumsq_raw`, `cb_partial`, `cb_gathered`, `cb_rinv_col`, `cb_rinv_bcast`, `cb_rinv`;
    - `_per_row_bytes()` (L1 budget, via `OpGeometry.t_int`, which replaces `t_f`), so `block_rows` grows when the tiles halve;
    - the writer's gather / multicast byte count. That CT arg was renamed from `f32_tile_bytes` to `int_tile_bytes`.
  - **Compute kernel unchanged.** It uses the same fused helper chains, and the helpers do the format reconfig.
  - **bf8b TILE γ**: the writer's `load_gamma` has a new block-float branch (CT arg 19, `gamma_is_block_float`). It reads the valid γ tiles whole, with one barrier, and zero-fills the nominal tail tiles. The face-row read stays for bf16/fp32 γ.
  - **bf8b input** uses the existing TILE path (`t_in` = 1088 via `tile_size`).
    - Host guard: bf8b requires TILE and an aligned W (ValueError otherwise; those cells are INVALID in feature_spec).
    - Reader `static_assert`: block-float is detected as `tile_bytes != 1024·es` and must never reach the RM / pad-lane paths.
    - `_element_bytes()` replaces `Tensor.element_size()`, which raises for bf8b.
  - The default compute config was left unchanged (HiFi4, fp32_dest_acc_en=True), following numeric-formats §9 "defaults must match prior behaviour". The golden harness always passes an explicit config.
  - Reused: the TILE reader/writer paths, all compute helpers, `select_blocking`. Added: one format variable, one γ read branch, guards.
- Accuracy achieved:
  - bf16 with fp32_dest_acc_en=False: PCC ≥ 0.995 on LOCAL / W_SPLIT / non-aligned shapes, TILE and RM, γ TILE and RM, poisoned padding.
  - Perf-profile config (bf16 / HiFi2 / acc off / TILE γ): PCC ≥ 0.9995 on (1,1,32,1024), (1,1,32,7168), (1,1,640,6144), (1,1,7168,1024).
  - bf8b input: PCC ≥ 0.99 versus the dequantized reference, with fp32 accumulation on and off.
  - bf8b γ: PCC ≥ 0.995 (bf16 x) / 0.999 (fp32 x).
- Golden test progress (targeted slices only; the harness full run is the record):
  - Interleaved loose cases with fp32_dest_acc_en=False: 100/100 pass, including all 12 interleaved `_perf_case` cases at PCC 0.9995 and the interleaved resilience and pad-poison cases.
  - Cartesian interleaved slice (bf8b or acc-off, 6 shapes): 129 passed, 66 xfailed (all `{float32, False}` EXCLUSIONS), 0 failed.
- Perf note (device ns, HiFi2, TILE bf16 γ, fp32-acc → bf16-acc):
  - (1,1,32,7168) W_SPLIT: 9210 → 8484 (−8%; gather and mcast bytes halved).
  - (1,1,640,6144): 48970 → 49639.
  - (1,1,8192,7168): 550390 → 552227.
  - (1,1,7168,1024): 77199 → 78061.
  - The three prefill shapes are DRAM-bound and flat within about 1.5%.
- Issues encountered: `Tensor.element_size()` raises for bfloat8_b. Fixed with the `_element_bytes()` helper. No other issues.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_numeric.py` (80 cases: acc-off matrix, bf8b input, bf8b γ, acc-off poisoned padding, perf-profile config at 0.9995, fp32+acc-off refusal).

## Refinement 2 — Sharded placements: HEIGHT / WIDTH / BLOCK_SHARDED, consumed in place
- Date: 2026-11-18
- What was done:
  - **SUPPORTED** `memory_layout` += HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, for TILE and ROW_MAJOR input. With a sharded input, the output inherits the input's shard spec (`memory_config=None` or that same config). Any other output placement for a sharded input is refused.
  - **Placement = core assignment.** `shard_placement()` derives the lines from the shard spec (grid, orientation, shard shape):
    - HEIGHT → one LOCAL line per shard core.
    - WIDTH → one W_SPLIT line over the whole shard grid, in shard order (multi-row grids included).
    - BLOCK → one W_SPLIT line per shard-grid row (per column for COL_MAJOR; the Mcast core order follows).
    - A ragged grid's all-padding shard row is dropped. Valid rows and W per core are runtime args.
  - **`select_blocking()` sharded branch** derives `w_split`, `core_w_tiles`, `block_rows` (whole shard when it fits) and the new `w_chunk_tiles` from the shard spec. The CB budget is the real region below the resident shard buffers (`min(shard addr) − L1 base − 16 KiB`).
  - **TILE, zero-copy:** `cb_x` and `cb_y` are backed on the input and output shards (`cb_descriptor_from_sharded_tensor`). The reader only publishes the block, compute packs y straight into the output shard, and the writer only consumes it. No DRAM traffic and no accessor read of the own shard.
  - **Ragged / poisoned W without touching caller L1.** A ragged shard's tail tiles are skipped: sumsq is `TileAddressing::Strided` over the fully valid tiles. The partial-W tile is squared into a second per-row Σx² slot and folded through a 0/1 lane mask by the collapse reduce (`ReducePartialScaler::only_partial()`, new `cb_wmask`). The test asserts the input shard is bit-identical after the call.
  - **ROW_MAJOR:** a core-local L1→L1 re-stride (shard stick pitch → tilize pitch) into `cb_x_rm`, and back out of `cb_out_rm` into the output shard.
    - Shards of 1–19 sticks and 4–40-element W slices both work.
    - RM γ slices whose byte offset is off the 64 B DRAM read grain are read through an aligned window in `cb_gamma_stage` and moved core-locally (element copy when off the 16 B L1 grain, e.g. fp32 RM shard + bf16 γ).
  - **W-chunk knob `w_chunk_tiles`** (default = `core_w_tiles`, byte-identical; interleaved never chunks). A HEIGHT shard pins a whole row to one core, so for wide rows (W 3000–11008, or fp32) the full-width γ and RM staging rings no longer fit beside two resident shards. In that case:
    - γ streams through `cb_gamma` in W-chunks on every normalize pass (TILE: Strided chunk chains packing at final positions; RM: in place, tile-row by tile-row, rinv by Offset).
    - RM tilize / untilize run as `tilize<wc>(rows·nchunks)` / `untilize<wc>(rows·nchunks)`.
  - Reused: both regimes, the gather + `ttnn.Mcast` combine, every compute helper chain, the RM tilize / untilize path, γ loaders (now one chunk loader; unchunked = chunk 0). Added: `shard_placement`, `CoreWork` list (shared by interleaved and sharded), sharded `select_blocking` branch, resident sumsq / masked collapse, chunked γ normalize, 2 CBs (`cb_wmask`, `cb_gamma_stage`).
- Accuracy achieved: all sharded unit cases at PCC ≥ 0.995 (bf16) / 0.999 (fp32), including poisoned padding (1000.0) with fp32 accumulation on and off. The sharded perf-profile config (bf16 / HiFi2 / acc off / TILE γ) reaches PCC ≥ 0.9995 on (1,1,32,{1024,5120,7168}) WIDTH and (1,1,7168,2048) BLOCK [736,192] 11×10.
- Golden test progress (targeted slices; the harness full run is the record):
  - All sharded loose cases (`_perf_case`, resilience, pad-poison, `_SHARDED`): 250 passed, 3 infeasible-skipped (input-shard / output L1 OOM), 0 failed.
  - Cartesian sharded slices over 10 shapes (1x1x32x8192, 2x1x64x4096, 1x1x128x4096, 128x8192, 1x32x8192, 1024x1024, 4x1x512x512, 4x8x32x47, 1x1x17x50, 2x1x100x47): 1129 passed, 0 failed (xfails = the permanent {fp32, acc-off} exclusion).
  - Host-side descriptor build over every supported sharded cartesian cell: 3536 OK, 0 refused.
  - Interleaved regression (acceptance + numeric + precision baseline): 200 passed, 6 skipped (unchanged).
- Perf baseline at the sharded perf-case config (device kernel ns, measured vs `achievable_ns`@1350 MHz):
  - WIDTH decode, W = 1024 / 2304 / 5120 / 7168: 4317 / 5436 / 5667 / 6556 vs 4110 / 4617 / 5267 / 5481.
  - BLOCK 640×6144: 17904 vs 19039. BLOCK 640×7168: 22272 vs 21669.
  - BLOCK 7168×1024: 55400 vs 18871. BLOCK 7168×2048: 66142 vs 25947. BLOCK 8192×1024: 76290 vs 25640.
  - The tall BLOCK shards (23–32 tile-rows per core, whole shard = one block) are about 3× off. The combine is one un-overlapped rendezvous per block that gathers `rows·k` whole Σx² tiles into each root. That is Refinement 3/4's scope (L4 column-0 gather, sub-block pipelining, compute fusion).
- Issues encountered:
  - A `static_assert` in a discarded `if constexpr` branch of a non-template kernel still fires (bf8b γ). Moved it to a gated top-level assert.
  - The test-only shard config "6 shards on 5 cores" was rejected by TTNN itself (not the op).
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded.py` (51 cases: HEIGHT/WIDTH/BLOCK × TILE/RM, ragged grids, multi-row WIDTH grids, poisoned padding × acc on/off, RM γ misaligned slices, COL_MAJOR BLOCK, perf-profile config, wide HEIGHT rows via the W-chunk knob, input-shard immutability, default output inherits the shard spec); probe `probes/perf_sharded_probe.py`.

## Refinement 2b — Sharded placements (debug: gate violation "REGRESSION 4491/4904")
- Date: 2026-11-18
- What was done: diagnosis plus a full golden re-run. The kernels and host code are unchanged.
  - The harness's R2 golden run (`io/logs/refinements/golden_refinement_2`) has 3000 `error/infra` cells, all "not reported by its device job". 119 of them were R1-passing interleaved cells, which is what the gate flagged as regressions.
  - All 3000 came from four chunks (002, 004, 006, 012) that the broker reaped as `hung` for "no output for 300s":
    - The eval runner writes pytest output to a file and prints nothing in the meantime.
    - Its precompile warm pass fails in this environment (`ImportError: Error importing plugin "bench_devfilter"`), so each test JIT-compiled its programs inline and serially, about 1.2 s per test cold.
    - A 4000-item chunk holding 250–500 non-skipped sharded/interleaved tests therefore ran past 300 s.
  - The `TT_THROW: trisc* build failed` lines at the end of those chunks are the compiler being killed, not a kernel error. No device hang was involved.
- Accuracy achieved: unchanged from Refinement 2. Every supported golden cell passes its harness tolerance.
- Golden test progress: a FULL suite run through `eval/eval_test_runner.sh` (same chunked broker path as the harness, 40142 items, 11 chunks, 27–254 s each):
  - PASSED=4993, FAILED=2, ERRORS=0, HANGS=0.
  - `verify_supported`: supported_pass 4980, supported_fail 0, xfail_expected 1229, xfail_wrong_mode 0, xpass_drift 0, infeasible_skipped 18, invalid_unexpected 2.
  - The 2 failures are the INVALID-per-feature_spec `test_translated::test_rms_norm_sharded_uneven_multicore_logical_width[w72|w200 … bfloat8_b]` cells (bf8b × non-aligned W). They hit the documented host `ValueError`; before R2 they were xfail.
  - 0 regressions against the R1 harness results (all 1471 R1 passes still pass).
- Issues encountered / findings:
  - Cold-compile cost is the op's exposure to this broker limit: each test is a distinct program (shape-specific CT args `block_rows`, `core_w_tiles`, `Wt`, `w_valid_last`, page sizes and the sharded TensorAccessor args, times the data formats).
  - The JIT cache (`built/`) now holds every golden program, so a re-run is warm. The slowest chunk took 254 s, and the harness's own run excludes test_translated from the cartesian chunks.
  - A structural fix would make the TILE kernels' shape extents runtime args, so binaries are shared across shapes. That rewrites all three kernels and was not done here, because any kernel edit invalidates the warm cache.
  - The bf8b non-aligned resident-shard case needs only the reader's block-float `static_assert` relaxed for `x_resident`. It was deferred for the same reason: the cells are INVALID, and the edit would invalidate the cache.
- Tests added: none.

## Refinement 3 — Speed up the width-sharded decode profile (cross-core combine latency)
- Date: 2026-11-18
- What was done (perf; no SUPPORTED change). Zone profiling of the W_SPLIT rendezvous on the 7×4 decode case (1 tile-row per core, x resident) gave the critical path: γ DRAM load before the partial send (1.3–1.9 µs), 27 × 2 KB fan-in into the root (~1 µs), root finalize (~1.4 µs), multicast (~0.7 µs), then the fused normalize (~1.3 µs). Levers, each measured on device and all kept:
  - **γ off the gather path (writer).** `load_gamma` split into `gamma_issue` / `gamma_commit`. W_SPLIT issues the γ reads at kernel start, inside the writer's wait for the first partial, and commits after block 0's partial send. TILE γ zeroes only the nominal tail tiles, since the Row-broadcast FPU op reads row 0 only; the full 16 KB zero fill was ~0.7 µs of writer time and is gone on every path.
  - **Column-scoped fused finalize (compute, all regimes).** New `RinvColumn` chain element: one SFPU pass `rsqrt(Σ·(1/W)+ε)` over the 8 column-0 vectors (`VectorMode::C` + even-parity stride, catalog `sfpu_tile_scope`), in place on `cb_rinv_col`, then `UnaryBcast<Col>`. The stock MulUnary / AddUnary / Rsqrt hardcode `VectorMode::RC`; the raw element is documented at its definition. Root finalize went from 1.4 µs to ~0.6 µs.
  - **L4 `GATHER_COL_FACES`.** The partial gather sends faces 0 + 2 only, halving the fan-in bytes.
  - **L5 `RINV_MCAST_HANDSHAKE = False`.** Safe by the partial(b+1) ⇒ rinv(b)-consumed causal chain.
  - **x·γ prescale (`Blocking.xg_prescale`, `XG_PRESCALE` knob, new `cb_xg`).**
    - `prescale_block` computes x·γ into `cb_xg` before the rinv wait: non-roots right after collapse; the root after finalize, overlapping its multicast.
    - Normalize is then one FPU mul `xg·rinv`.
    - `select_blocking()` enables it only for W_SPLIT ∧ TILE ∧ resident γ, and only where `cb_xg` fits without shrinking `block_rows`.
    - Same FP order and intermediate precision, so y is bit-identical.
  - Reused: the whole W_SPLIT topology, gather / Mcast helpers, every compute chain, the `CoreWork` / `select_blocking` plumbing. Added: `RinvColumn`, `prescale_block`, `cb_xg` (CB 14), the `gamma_issue` / `gamma_commit` split, 3 host knobs (`GATHER_COL_FACES`, `RINV_MCAST_HANDSHAKE`, `XG_PRESCALE`).
- Perf (device kernel ns, BH p150b at 1350 MHz; `achievable_ns` applies unscaled):

  | WIDTH decode case | Before | After | Speedup | `achievable_ns` |
  |---|---|---|---|---|
  | 8×1 (W=1024) | 4323 | 2978 | 1.45× | 4110 |
  | 9×1 (W=2304) | 5453 | 3347 | 1.63× | 4617 |
  | 8×4 (W=5120) | 5647 | 3750 | 1.51× | 5267 |
  | 7×4 (W=7168) | 6508 | 3792 | 1.72× | 5481 |

  Guard set (R2 commit → now) — every shape equal or faster:
  - BLOCK: 640×6144 17.9 → 12.1 µs; 640×7168 22.3 → 14.8 µs; 7168×1024 55.4 → 31.1 µs; 7168×2048 67.3 → 47.5 µs; 8192×1024 77.4 → 50.7 µs
  - HEIGHT 1024×4096: 38.7 → 31.4 µs
  - RM WIDTH 32×4096: 10.6 → 9.8 µs
  - interleaved TILE: 32×1024 5.8 → 4.2 µs; 32×4096 6.8 → 5.4 µs; 32×7168 8.5 → 6.8 µs; 8192×7168 553 → 543 µs; 7168×1024 77.2 → 76.1 µs
  - interleaved RM 32×4096: 9.6 → 8.8 µs
- Accuracy achieved: the four perf cases at PCC 0.999983–0.999986 (gate 0.9995). Every existing unit tolerance holds.
- Golden test progress (targeted slices; the harness's full run is the record):
  - `test_op_loose` WIDTH + BLOCK sharded (includes the 4 targets): 165 passed, 3 infeasible-skipped
  - HEIGHT / interleaved decode loose + interleaved W_SPLIT cartesian 1×1×32×8192 bf16: 85 passed
- Issues encountered:
  - The first prescale version tilized RM γ twice (`gamma_ready()` ran before the prescale and again before normalize), which hung `test_sharded_tile_rm_gamma_fp32`. Fixed: the trailing call is skipped when the prescale already ran.
  - The fused SFPU body first ran out of LREGs (1/W and ε held live across the rsqrt body). The constants are now re-materialized per vector.
- Remaining headroom (finding, not a follow-up): see the Refinement 3 **Outcome** line in `op_requirements.md`.
- Tests added: probes `probes/perf_r3_width_decode.py` (the 4 perf cases, ns + PCC) and `probes/perf_r3_guard.py` (13-shape guard set, `xg` A/B). Unit suites re-run green: sharded 51, acceptance 110, numeric 74 + 6 skipped, precision baseline 16.

## Refinement 4 — Speed up the block-sharded prefill profile (per-core compute throughput)
- Date: 2026-11-18
- What was done (perf; no SUPPORTED change). Zone profiling of the 8192×1024 8×8 BLOCK case (R3 code, root core, math cycles) showed it compute-bound end to end:
  - normalize 27.2k, finalize 14.1k (12k of it the column-scoped rsqrt at 375 cycles / tile-row), combine 11.4k (256 column-tile adds), collapse 10k (SFPU lane reduce, ~305 / tile), sumsq 5.5k.
  - `cb_gathered` (`k·br` tiles) also split the shard into 2 serialized rendezvous blocks and kept the x·γ prescale off.
- Four levers, each measured on device and kept as a host knob in `rms_norm_program_descriptor.py`:
  - **L13 `NORM_DEST_BLOCK = 8`** (catalog `compute_block_size`). The normalize / prescale chains used the default per-tile output lifecycle, which clamps `block_size` to 1.
    - Now: `PerBlockSize` output (and `cb_xg` input) lifecycle + `.block_size(8)`, wherever a window can never straddle the output ring (`_norm_block`: resident output shard, `cb_xg`, or an interleaved `cb_y` ring and block that are both multiples of 8; RM in-place stays 1).
    - Measured: 8192×1024 50.7 → 45.7 µs on its own; in the final config leave-one-out (`=1`) costs 1.15–1.31× on all 5 BLOCK cases.
  - **L14 `COLLAPSE_REDUCE_TILE`.** The collapse runs on the ReduceTile datapath: the reduce helper with `ReduceAlgorithm::ReduceTile`, against a ones scaler in the new `cb_reduce_scaler` (CB 15), filled by `calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`.
    - Collapse dropped 10k → 1.7k cycles; 8192×1024 45.7 → 40.0 µs.
    - The masked partial-W collapse keeps AccumulateViaAdd and its 0/1 lane mask.
    - The scaler is built after block 0's x is published.
  - **L15 `DENSE_RINV` (`DENSE_MIN_ROWS = 4`).** W_SPLIT blocks of ≥ 4 tile-rows pack the per-row Σx² of up to 32 tile-rows into ONE dense tile: `densify_block`, `D = Σ_r S_r @ J_r`, a `matmul_block` against one-hot column selectors `cb_jsel` (CB 17).
    - Gather, root combine, rsqrt finalize (`RinvColumn<true>`, full tile) and the rinv multicast then run on `⌈rows/32⌉` tiles. The root combines into the new `cb_dense_sum` (CB 16).
    - Every core expands its received dense rinv back to per-row broadcast tiles: `expand_block`, `R @ J_rᵀ`, `matmul_block<transpose>`.
    - `cb_gathered` shrinks 32×, so a whole tall shard is one block and the x·γ prescale fits. The prescale then measures 1.25–1.27× on those shapes (catalog `compute_fusion`: the L1 round trip beats FPU DEST-reuse).
    - The selectors are built by the reader *after* block 0's x is published. It writes the 16 one-hot faces (256 stores) and assembles the rest with local 512 B NoC copies.
    - The first version built them before the x push, which put ~8k cycles in front of sumsq. Moving the build: 8192×1024 22.0 → 16.3 µs.
    - Measured: blocks of 2 tile-rows (640×6144 / 640×7168) are 2–4 % slower dense, hence the `DENSE_MIN_ROWS` gate.
  - Reused: every compute chain, the reduce / matmul / eltwise helpers, the gather + Mcast topology, `select_blocking`.
  - Added:
    - `densify_block` / `expand_block` / `finalize_dense_block`; `RinvColumn<kDense>`
    - `_footprint_dense` / `_dense_block_rows`; `Blocking.dense`
    - CBs 15–17; 3 host knobs + `DENSE_MIN_ROWS`
- Perf (device kernel ns, BH p150b at 1350 MHz; `achievable_ns` applies unscaled):

  | BLOCK perf case | R3 | R4 | Speedup | `achievable_ns` | R4 / ach |
  |---|---|---|---|---|---|
  | 640×6144 9×10 | 11996 | 10710 | 1.12× | 19039 | 0.56 |
  | 640×7168 7×10 | 14800 | 12700 | 1.17× | 21669 | 0.59 |
  | 7168×1024 8×10 | 31122 | 13201 | 2.36× | 18871 | 0.70 |
  | 7168×2048 11×10 | 47550 | 16998 | 2.80× | 25947 | 0.66 |
  | 8192×1024 8×8 | 50706 | 16269 | 3.12× | 25640 | 0.63 |

  Final-config leave-one-out (5 cases, µs):
  - `NORM_DEST_BLOCK=1`: 12.3 / 15.1 / 16.6 / 22.3 / 21.0
  - `DENSE_RINV=False`: 10.7 / 12.8 / 23.6 / 37.4 / 39.5
  - `COLLAPSE_REDUCE_TILE=False`: within ±2 % on the BLOCK cases, since dense has no collapse. Across the guard set (3× medians, after the scaler move): −1 to −4 % on decode / RM WIDTH with the lever, +1–2 % on the 2-row 640 BLOCK cases (noise band), tied on HEIGHT / interleaved LOCAL.

  Guard set (vs R3), all equal or faster:
  - WIDTH decode: 2923 / 3339 / 3691 / 3796 ns (R3: 2978 / 3347 / 3750 / 3792)
  - HEIGHT 1024×4096: 31.4 → 26.6 µs
  - RM WIDTH 32×4096: 9.8 → 9.7 µs
  - interleaved 32×1024 / 4096 / 7168: 4.2 / 5.4 / 6.8 → 4.1 / 5.2 / 6.6 µs
  - interleaved RM 32×4096: 8.8 µs, unchanged
  - interleaved 7168×1024 (LOCAL) and 8192×7168 (W_SPLIT): DRAM-bound and unchanged. 8192×7168 3× median 550.8 µs with the levers vs 551.2 µs without; R3's recorded 543 is run-to-run drift.
- Accuracy achieved: the five BLOCK perf cases at PCC 0.99986–0.999998 (gate 0.9995), identical to R3 to 6 digits; every existing unit tolerance holds.
- Golden test progress (targeted slices; the harness's full run is the record):
  - `test_op_loose`, all placements: 353 passed, 3 infeasible-skipped
  - `test_op` cartesian 1x1x128x4096 + 1x1x17x50 + HEIGHT / interleaved loose: 442 passed, 69 xfailed
  - `test_translated -k sharded`: 48 passed, 2 failed. The 2 are the known INVALID bf8b × non-aligned-W cells; their host `ValueError` is unchanged since R2b.
- Issues encountered:
  - Dense footprint first charged interleaved `cb_x` per block instead of per tile-row. It picked block_rows = 5 for interleaved 8192×7168 and OOMed L1 (2.7 MB). Fixed; interleaved W_SPLIT shapes stay sparse (block_rows = 1), so dense effectively applies to tall sharded W_SPLIT shards.
  - `input(cb_xg, PerTile, PerTile, Block)` fails a chain static_assert when `norm_block == 1`. The mapping is now `Scalar` there.
  - Profiler zone hash collisions when zone names were reused across builds. Profiling only; zones are not committed.
  - Dense rinv changes the blast radius of a NaN / inf in tile padding: `S_r[i,:]·0 = NaN` would reach element-row i of every tile-row in the group. Finite poison (1000.0, the tested case) is exact.
- Remaining headroom (finding, not a follow-up): see the Refinement 4 **Outcome** line in `op_requirements.md`.
- Tests added:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_dense.py`: 15 cases + 1 permanent-exclusion skip. Each asserts the dense schedule was chosen. Covers BLOCK TILE perf shapes (acc on/off, HiFi2 0.9995 gate), a 48-row block (two dense groups), γ none / TILE / RM, fp32 input, bf8b input, ROW_MAJOR BLOCK, and a multi-row WIDTH grid.
  - Probe `probes/blk_probe_r4.py` (blocking per perf case).
  - Unit suites green: 266 passed, 7 skipped.

## Refinement 4b — Speed up the block-sharded prefill profile (debug: gate violation "REGRESSION 4218/4904")
- Date: 2026-11-18
- What was done: diagnosis plus a FULL golden re-run. The kernels and host code are unchanged from the R4 commit (`d4020d75`).
  - The harness's R4 golden run (`io/logs/refinements/golden_refinement_4`) has 668 `supported_fail` cells. All of them are `status=error, failure_category=infra`, items that no device job reported. There are also 4979 `invalid_unexpected` and 169 `xfail_wrong_mode`, from the same unreported items. Every cell that did execute passed: PASSED=4231, FAILED=0, HANGS=0.
  - The unreported items came from 7 chunks (002, 004, 006, 008, 011, 014, 019) that the broker reaped as `hung` for "no output for 300 s". The chunk retries were reaped as well.
  - The cause is the one diagnosed in R2b:
    - The eval runner writes pytest output to a file, so the broker sees no output.
    - Its precompile warm pass fails in this environment (`ImportError: Error importing plugin "bench_devfilter"`).
    - R4's kernel edits invalidated the JIT cache, so every test JIT-compiled inline and serially, at ~1.2 s per test cold.
    - A 4000-item chunk with 250–500 runnable tests therefore ran past 300 s.
  - The `assert.hpp` / `trisc build failed` tails of the reaped chunks are the compiler being killed. They are not kernel errors.
- Accuracy achieved: unchanged from Refinement 4. Every supported golden cell passes its harness tolerance; the BLOCK perf cases hold PCC ≥ 0.9998 at the 0.9995 gate.
- Golden test progress: a FULL suite run through `eval/eval_test_runner.sh eval/golden_tests/rms_norm/ … --ignore=…/test_translated.py` (the harness's exact chunked broker path; 40046 items, 11 chunks, 23–106 s each):
  - PASSED=4899, FAILED=0, ERRORS=0, HANGS=0.
  - `verify_supported`: supported_pass 4886, supported_fail 0, xfail_expected 1229, xpass_drift 0, xfail_wrong_mode 0, invalid_unexpected 0, infeasible_skipped 18. This is identical to the R2 verified state.
  - 0 regressions against the union of the R2 and R4 harness passes (4886 cells).
  - Unit dir `tests/ttnn/unit_tests/operations/rms_norm/`: 284 passed, 7 skipped.
- Issues encountered / findings:
  - The op's exposure to the broker limit is cold-compile volume. Every golden test is a distinct program, because shape-specific CT args (`block_rows`, `core_w_tiles`, `Wt`, `w_valid_last`, page sizes, sharded TensorAccessor args, dense-group extents) are crossed with the data formats.
  - Any kernel edit puts the next harness run back in this regime until the cache is re-warmed. Phases that edit kernels should end with one full golden run through `eval/eval_test_runner.sh`; warm, it takes ~10 min.
  - The structural fix is to make the TILE kernels' shape extents runtime args, so binaries are shared across shapes. It was not done here: it rewrites all three kernels, would invalidate the cache this run just warmed, and is a perf-neutral-at-best change outside R4's scope.
- Tests added: none.
