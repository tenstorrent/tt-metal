# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-14
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op(MeshProgramDescriptor)` per call.
  - x is resident in L1 across a flat fp32 record all-gather over Fabric (port cores, ready fence, arrival counters, one-hop relay).
  - Rows × W-chunks are spread over the full worker grid.
- **SUPPORTED at Phase 0** (after verifier drift fixes):
  - dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE]
  - alignment=[tile_aligned, h_non_aligned], rank=[2, 3, 4]
  - gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"]
  - memory_layout=[INTERLEAVED], cluster_axis=[0, 1], topology=[Linear, Ring], num_links=[1, 2, None]
- **Accuracy achieved** (worst device, 4 shapes, `test_distributed_rms_norm_precision_baseline.py`):
  - bf16: PCC=0.999998, max_abs_err=0.043, rel_rms_err=0.0019, max 1 bf16 ULP.
  - fp32: PCC=1.000000, max_abs_err=0.031, rel_rms_err=0.0015. There is a systematic −0.1 % bias (tf32 FPU operand truncation), noted on Refinement 4.
- **Golden suite at Phase 0**: 931 supported_pass, 0 supported_fail, 0 xpass_drift, 1 xfail_wrong_mode (harness L1-profiling artifact on a correct refusal), 7125 xfail_expected, 7776 invalid_skipped (per `verifier_report.json`). Acceptance 27/27.
- **Device time** (bf16 + gamma, axis 1, L=2, max over 4 devices): 640×1792 24.9 µs, 32×1792 10.9 µs, 2048×1280 44.6 µs.
- **Issues encountered / verifier fixes**:
  - **Capacity failure fixed.** Per-device (2, 2048, 4096) bf16 + gamma raised the op's capacity error: 1.39 MB, of which 0.70 MB was the fp32 `cb_xg` prescale block. Added a per-call `GAMMA_PRESCALE` regime choice: when the prescale does not fit, gamma is fused into normalize via DEST-reuse Mul against row-replicated gamma tiles, and the footprint drops to 0.68 MB.
  - **Registry drift fixed:** `num_links=None` was missing from SUPPORTED (197 xpass_drift).
  - **Promoted on evidence:**
    - `Ring` (ring mock 16/16, plus fp32 / no-gamma / repeat bit-identical probe);
    - `h_non_aligned` (144/144 cells).
  - **DRY fixes:**
    - the writer's output batch now comes from `out_batch_tiles_knob` (RT arg), not a hard-coded `cw`;
    - `RECORD_BYTES` is passed as a CT arg instead of a literal in two kernels.
  - **Caller errors added:** non-ROW_MAJOR gamma, and an explicit `num_links` beyond the port-column cap. The latter was silently clamped before.
  - `l1_ledger.md` was brought current (`MIN_CORE_W_TILES`=8, conditional `cb_xg`, worst-case footprints on the 11×10 grid).
- **Tests added**: test_distributed_rms_norm.py (planner), test_distributed_rms_norm_precision_baseline.py, test_distributed_rms_norm_perf_bench.py (device-time bench for the LOOSE_CASES configs), probes/probe_001.py (Ring verification probe).

## Refinement 1 — Native sharded placement (HEIGHT_SHARDED + WIDTH_SHARDED)
- **Date**: 2026-10-14
- **What was done**:
  - `SUPPORTED["memory_layout"]` gains HEIGHT_SHARDED and WIDTH_SHARDED (L1). A DRAM-sharded input is a caller error. With `memory_config=None`, a sharded input now keeps its shard spec for the output.
  - **Native zero-copy.**
    - `cb_x_resident` is `cb_descriptor_from_sharded_tensor(input)`. The reader only publishes the resident tiles; there is no NoC read and no TensorAccessor on the core's own shard. Compute reads x in place through `TileAddressing::Strided` (row stride = shard width).
    - When the output shard spec equals the input's, `cb_output_tiles` is the output shard. Compute packs into it (Strided) and the writer's store disappears.
    - Otherwise the writer stores through a TensorAccessor to any output config (the translated tests use other output grids).
  - **Worker set = shard grid.**
    - HEIGHT (R9): `S_w = 1`, one row group per shard.
    - WIDTH (R10): `S_r = 1`, `S_w` = shard cores. A ragged last shard is handled by valid `cw` vs physical `cw_phys`.
    - Ports go on the first free grid cores.
    - Stats-ready is multicast per shard-grid rectangle (port kernel: a list of rectangles).
  - **R5 device-local pre-combine (WIDTH).**
    - Each W chunk writes its per-row record into the inbox of the combiner owning that tile row: an L1 sharded tensor, with rows split `lc_per` per chunk. Combiners REDUCE_COL the S_w records (new `cb_lc` → `cb_lc_out`) and publish one record per tile row.
    - Why: flat records at a pinned S_w ≤ 86 would need `G·S_w·Ht·128 B` of lockstep scratch (1 MB at 4096×1024) and quadratic pulls.
  - **New knobs (single source in `_choose_knobs`).**
    - `stat_rows`: the per-row stats CBs stream in batches; WIDTH holds up to 128 tile rows.
    - `gamma_chunk`: gamma streams double-buffered in chunks on the fused path; HEIGHT 32×8192 needs 1 MB of gamma tiles beside 2×512 KB of shards.
    - Both are chosen as the coarsest that fits the CB room the allocator really leaves (L1 block table via `GetMemoryView`, so tensors left alive by earlier OOM-skipped cells are counted).
    - Interleaved defaults are unchanged: the whole block and the whole gamma.
  - The interleaved path keeps its Phase 0 per-core runtime-arg lists: sharded-only args are compile-time gated. Appending them unconditionally added host dispatch time, and so cross-device launch skew, measured as +2–12 µs of exchange wait on 640×1792.
- **Reused / Added**:
  - Reused: the R1 record exchange (port kernel, fence, relay), the sumsq / finalize / normalize helper calls, the gather (now a lambda shared by the stats gather and the R5 local gather), the record-publish code (now a lambda), and the ragged-chunk CB grouping.
  - Added: the sharded layout plan, shard-backed CBs, Strided chains on the sharded path, the R5 inbox + 2 CBs + 1 semaphore, stats batching, and gamma chunking.
- **Accuracy achieved**: same tolerance gates as the golden suite (bf16 PCC ≥ 0.999 class, fp32 tighter), passing on every shape below.
- **Golden test progress**:
  - Supported sharded cells (`*_SHARDED`, fp32/bf16, `fp32_dest_acc_en=True`, tile_aligned / h_non_aligned): 1404 passed, 0 failed, 36 `INFEASIBLE_L1` skips (harness buffer OOMs); the rest are INVALID.
  - Interleaved supported slice: 720/720 passed.
  - Loose 3/3; translated 231 passed, 47 xfailed (bf8b), 0 failed; ring mock 16/16; unit dir all green.
- **Device time** (bf16 + gamma, axis 1, max over 4 devices):
  - Interleaved unchanged: 640×1792 24.3–27.1 µs (baseline on this box 24.3–24.5), 2048×1280 ~45 µs.
  - HEIGHT 640×1792: 19.8 µs, below interleaved because there are no DRAM crossings.
  - WIDTH: 32×1792 9.3 µs, 640×1792 ~81 µs, 2048×2048 ~171 µs.
- **Finding (WIDTH perf)**:
  - WIDTH costs ~2.6 µs per tile row and is compute-bound (TRISC0 is the long pole). Every W-chunk core runs the per-row stats pipeline for all Ht rows: REDUCE_ROW + transpose, then REDUCE_COL + transpose + rsqrt finalize, on 1-tile-wide rows.
  - Next levers:
    - finalize once per row on the combiners and multicast rstd (R4 for the sharded path);
    - batch the per-tile transposes;
    - spread combiners over both links (WIDTH runs `L_eff = 1` since `S_r = 1`).
  - Not done here: this refinement is generality.
- **Issues encountered**:
  - In a long golden session, a cell whose input shard or output OOM-skips leaves its tensors alive. A fixed L1-base budget then let CBs clash with them.
  - Fixed by budgeting against the allocator's block table.
  - Tracy capture crashed on a multi-case profile of the sharded test file; single-case profiles were used instead.
- **Tests added**: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_sharded.py` (12 cases: HEIGHT prescale / h_non_aligned / gamma-chunked, WIDTH one-combiner / many-combiner / stats-batched / ragged last shard, axis 0 with 2 links).

## Refinement 2 — W_local non-tile-aligned (`w_non_aligned`)
- Date: 2026-10-14
- What was done:
  - `"w_non_aligned"` added to `SUPPORTED["alignment"]` on every placement (interleaved, HEIGHT_SHARDED, WIDTH_SHARDED).
  - Verifier option (a): the reader zeroes columns `W_local % 32 .. 31` of the tensor's last W tile in each tile row before `cb_push_back`. On the interleaved path this happens in the CB page after the read barrier; on the sharded paths it happens in the resident shard, on the core whose chunk ends at `tensor_w_tiles`.
    - Whole padding faces (1 and 3 when `w_tail <= 16`) are zeroed with `noc.async_write_zeros`.
    - The partial face rows (16 rows × (16 − c0) elements in 2 faces) are the only RISC stores.
  - The new CT arg `w_tail` is 0 on aligned shapes, which compiles the mask out, so aligned paths are byte-identical in behaviour.
  - Nothing else changes: sumsq, the record layout and count, the exchange, and normalize. The mean already divides by the true `G·W_local`.
  - Gamma: each segment read is clamped to the stick's `buffer_aligned_page_size()`, so it never reads past the page. The offsets stay multiples of the 64/128 B segment, which keeps the DRAM-source alignment. The padding columns of the gamma tile only ever multiply padding columns of x.
  - Reused: the reader `load_block`, the existing `_choose_split` / ragged-chunk sizing, and every compute phase. Added: one reader CT arg, the `mask_w_tail` lambda, and the gamma read clamp.
- Accuracy achieved: within the golden tolerances (fp32: PCC ≥ 0.9999, rel-RMS ≤ 0.005; bf16: PCC ≥ 0.999, rel-RMS ≤ 0.01) on (32,17), (1,1,32,50), (1,1,17,50), (1,1,64,1000), (2,1,128,100), (4,128,47) across all three placements.
- Golden test progress:
  - w_non_aligned slices: 18/18 (axis 1, bf16/bf16, 1 link, all placements) and 111 passed / 108 skipped / 0 failed (axis 0, 2 links, fp32 + bf16, every gamma mode, plus tile-aligned representatives).
  - The full suite is left to the harness.
- Issues encountered:
  - Golden inputs come from `ttnn.from_torch`, which zero-pads the last tile. So the golden suite passed with SUPPORTED widened alone and cannot detect a missing mask.
  - The new unit test injects garbage into the padding. Without the mask it fails (rel-RMS ≈ 0.98 on every placement); with the mask it passes.
  - The sharded mask writes zeros into the padding of the input's resident shard. That region is tile padding (not logical data), so the input's logical value is unchanged.
- Tests added: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_w_non_aligned.py`, 7 cases with garbage in the tile padding: interleaved / HEIGHT / WIDTH, fp32 + bf16, w_tail 4/8/15/17/18 (both the faces-{0,2} and faces-{1,3} branches).

## Refinement 3 — Speed up the perf-flagged profile (overlap the exchange with DRAM)
- Date: 2026-10-14
- What was done:
  - **R2 multi-round regime built** (interleaved). `block_rows < core_rows`. All three worker kernels split each round into:
    - `front(b)`: load / sumsq / publish / prescale.
    - `back(b)`: gather / finalize / normalize / store.

    They run `front(b + ROUND_LOOKAHEAD)` before `back(b)`. Round b's exchange therefore runs under round b+1's DRAM reads and sumsq, and round b+1's exchange under round b's normalize and writes. `ROUND_LOOKAHEAD = 1` is a host knob passed as a CT arg; 0 gives the serial schedule.
  - **Port:** loops the rounds. Each round:
    - waits on round b's local-arrival threshold;
    - sends round b's region of each slot (the slot is round-major; one entry = `round_recs·128 B`);
    - waits for arrivals ≥ `(b+1)·expect_in`;
    - multicasts +1 to its own stats-ready counter.

    The ready fence is still once per call, and the arrival counter is re-armed by `num_rounds·expect_in`.
  - **Synchronization (two races found and fixed):**
    - One shared stats-ready counter (≥ `2L·(round+1)`) would let a fast port's round b+1 stand in for a slow port's round b. Now there is one counter per port, `SEM_STATS_READY_BASE + 2l + d`, and workers wait on both ports of their link.
    - One shared local-arrival counter did let a fast worker's round-1 increment complete a slow worker's round 0. Observed: round-0 rows wrong on 4096×1024 / 2048×1280. Now there are `1 + ROUND_LOOKAHEAD` rotating counters (round b → `b % (1+LA)`). A worker publishes round b+1+LA only after its round-b gather, so no counter can run ahead.
  - **Split** (`_rounds_split`):
    - `EXCHANGE_ROUNDS = 2` caps `S_r` at `ceil(Ht/2)` and relaxes the W-chunk floor so `S_w` may double (the design's "S_r = Ht/2, S_w doubled"). It is taken only if the tiles per worker do not grow.
    - The first try kept the floor at 8 (640×1792 → `S_w` 7, 16 tiles per worker). That measured worse on max-device (25.5 µs median) than the floor-relaxed `S_w` 8 / 14-tile split (21.7 µs).
  - **P_resident lift** (`_fit_rounds`): when the footprint does not fit, `rounds` grows (prescale path preferred). (2, 2048, 4096) bf16 + gamma left the fused fallback for 4 one-row prescale rounds.
  - **cb_x_resident sizing (`x_blocks`):**
    - With prescale (x popped every round): `1 + LA` blocks.
    - Without prescale: the whole assignment. The next round's sumsq reads past the held block at a tile offset that is not wrap-safe; a `1+LA` ring corrupted fp32 4096×1024 at 4 rounds.
  - **Reader:** gamma loads after the lookahead x blocks. Between them it delayed round 1 (fused fallback measured +0.6% until moved).
  - Sharded placements keep one round. The shard is the block, and R5 runs once.
  - **Reused / Added:**
    - Reused: every helper call, the gather / publish lambdas, the record layout per round, the port send loop, and the CB grouping.
    - Added: front/back lambdas plus the lookahead loop, per-round port thresholds, per-port stats-ready counters, rotating local-arrival counters, `_rounds_split` / `_fit_rounds` / `x_blocks`, and the knobs `EXCHANGE_ROUNDS`, `ROUND_LOOKAHEAD`, `GAMMA_PRESCALE`.
- Accuracy achieved: golden tolerances (fp32 PCC ≥ 0.9999 / rel-RMS ≤ 0.005; bf16 PCC ≥ 0.999 / rel-RMS ≤ 0.01). Shapes: 640×1792, 2048×1280, 4096×1024 (ragged rounds), 100×512, (2, 2048, 4096), and every golden interleaved shape.
- **Perf** (median over calls of the max over 4 devices, bf16 + gamma, axis 1, Linear, 2 links unless noted):

  | Shape | Before | After |
  |-------|--------|-------|
  | **640×1792 (flagged)** | **25.1 µs** | **21.9–22.5 µs** (min device 21.7 → 18.8) |
  | 2048×1280 | 45.2 | 37.3–40.7 |
  | 32×1792 | 10.8 | 10.5 |
  | fp32 no-gamma 640×1792 | 38.8 | 34.3–35.4 |
  | (2, 2048, 4096) | 346.5 | 196–204 |
  | HEIGHT 640×1792 | 19.8 | 19.8 |
  | WIDTH 640×1792 | 70.9 | 70.9–71.7 |
  | Ring mock 640×1792 L2 (single calls) | 54.6 / 55.3 | 51.7 / 54.3 |

  Gamma prescale vs fused: fused is ~2× slower (640×1792: 52 µs) because of the reader's RISC row replication, so prescale stays.
- **Finding:** max-over-devices is now set by cross-device launch skew. 2–5 µs typical, with 10+ µs spikes in ~1 of 10 calls. Per-device work is 18.8 µs. Refinement 5 (per-link stats-ready, port placement) is next.
- Golden test progress:
  - Interleaved + sharded slice (fp32/bf16, `fp32_dest_acc_en=True`, `num_links=2`, both axes / axis 1 sharded), plus loose and ring mock: 667 passed, 0 failed (648 skipped as INVALID / topology).
  - Translated: 235 passed, 43 xfailed, 0 failed.
  - Unit directory: 69/69.
- Issues encountered: the two counter races and the CB wrap above, each found by a deterministic debug test before landing.
- Tests added:
  - `test_distributed_rms_norm_rounds.py`: `EXCHANGE_ROUNDS` 2/4 × `ROUND_LOOKAHEAD` 0/1/2 × prescale / fused, ragged rounds, h_non_aligned, two calls each (counter re-arm), plus the P_resident-lift shape.
  - `test_distributed_rms_norm_debug.py::test_r2_round_values`: per-tile-row values that expose a round reading the wrong stats.
  - `test_distributed_rms_norm_perf_bench.py`: 21 calls per shape, plus `test_guard_bench` (the guard set).

## Refinement 4 — Numerical configurability (bfloat8_b + fp32_dest_acc_en=False)
- Date: 2026-10-14
- What was done:
  - `SUPPORTED["dtype"]` += `bfloat8_b`, `SUPPORTED["fp32_dest_acc_en"]` += `False`. The `{float32, fp32_dest_acc_en=False}` EXCLUSION is kept (precision convention). No other EXCLUSIONS were needed: bf8b x w_non_aligned, h_non_aligned, HEIGHT and WIDTH all pass.
  - **Intermediate formats:** `cb_xg` follows the DEST width (`xg_dtype_of`: Float32 / Float16_b). It is threaded through `build_plan` → `_fit_rounds` / `_worker_footprint`, so the rounds / prescale fit sees the narrower page. The stats CBs and wire records stay Float32 (the spec's fp32 exchange).
  - **Compute (the one kernel change):** the reduces' `ReduceFp32Mode` is `Accurate` only with a 32-bit DEST. Reason: the Accurate SFPU reduce takes its format from the input CB (Float32), so on a 16-bit DEST it reads the DEST as fp32 and every output came back inf. With a 16-bit DEST the reduces take the FPU path (Fast). The post-ops then re-arm `reduce_init` instead of `copy_init`, clearing the pack edge mask after it. The fp32-DEST path is call-for-call identical to before.
  - **Reader:** the w_tail mask handles Bfp8_b tiles. Mantissa bytes sit after the 64 B exponent section and are 1 B per element. A zero mantissa byte is 0.0 whatever the shared exponent. New CT arg `x_exp_bytes` (host `BFP8_EXP_BYTES`).
  - **UnpackToDestFp32 knob (`UNPACK_STATS_TO_DEST`, parked = False):** it tags `cb_sq_acc` / `cb_stats_combine` / `cb_lc` (read only by copy_tile on the Accurate SFPU path; `cb_rstd` / `cb_xg` feed FPU multiplies and are never tagged). It is correct: all tests pass with it on. But measured end-to-end it is worse. Fp32 mean ULP goes 15.0k → 19.9k, median got/true 0.9988 → 0.9984; bf16 mean ULP 0.12 → 0.18. The stats' tf32 truncation (which lowers sum(x²) and so raises rstd) was partly cancelling the downward tf32 truncation of x / gamma / rstd on the FPU multiplies. Turn it on together with an fp32-exact multiply path (x unpack-to-DEST + SFPU multiplies). That is the remaining fp32-bias lever, not taken here.
  - Reused: every helper call, the CB inventory, and the program descriptor / kernel files. Added: `xg_dtype_of`, `UNPACK_STATS_TO_DEST`, `STATS_UNPACK_TO_DEST_CBS`, `BFP8_EXP_BYTES`, the `stats_mode` / `rearm` constexpr branch, and the reader's 1-byte mask branch.
- Accuracy achieved: all cells meet the golden gates. These are bf8b PCC ≥ 0.995 / rel-RMS ≤ 0.04 (fp32 acc) or 0.05 (bf16 acc), and bf16 + False PCC ≥ 0.999 / rel-RMS ≤ 0.015. Shapes covered: 32x50 … 4096x1024, 8192-wide, rank 2/3/4, all alignments, interleaved / HEIGHT / WIDTH.
- Precision baseline (re-run, shipped default = fp32 DEST, knob off; identical to the verifier baseline). Fp32: PCC 1.0, rel-RMS 0.00136–0.00146, mean ULP 13.9k–15.2k, median got/true 0.9988. Bf16: rel-RMS 0.00184–0.00189, mean ULP 0.11–0.13, max ULP 1.
- Golden test progress: 1221 / 1221 target cells run in slices, all passed. The slices were cluster_axis=1 / num_links 1 and 2 / Linear across all 36 shapes, plus a cluster_axis=0 / 2-link FLOAT32-gamma sample. test_op_loose, test_regression and test_translated (DeepSeek decode/prefill, bf8b distributed-norm) pass, except 2 translated cells below. The harness's full run is the record (7776 target cells).
- Issues encountered:
  - Accurate SFPU reduce on a 16-bit DEST → inf (fixed as above).
  - **Pre-existing, independent of dtype or acc (finding):** `test_ds_distributed_norm[*-prefill-8192]` hits "static CBs clash with L1 buffers". The interleaved path budgets CBs at a fixed `WORKER_L1_BUDGET` (1 MiB) and never subtracts the per-call exchange scratch (lockstep L1 allocation lowers every core's CB ceiling). It also fails with fp32_dest_acc_en=True for (1,1,8192,3584). `prefill-131072` OOMs allocating the exchange scratch: `G·slot_bytes` grows with all rows, about 3.7 MB per port. Both were failing before this refinement (UnsupportedAxisValue), so they are not a regression. The fix belongs to the interleaved L1-fit / exchange-slot reuse work, not to numerics.
- Tests added: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_numerics.py` (12 cases: bf8b and bf16+False across interleaved prescale / fused gamma / w_tail ≤16 and >16 / HEIGHT / WIDTH).
