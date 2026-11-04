# Changelog: nomic_residual_layer_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-03
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op` program, regime `split3_exchange`.
  - x / residual / output are ND-sharded in L1 over groups of 3 row-major cores.
  - Each core sums its hidden third in fp32 and packs per-row (S, Q) into one tile.
  - The packed stats are all-gathered once per group (Mcast fixed-sender channels, chain unicast). Each core finalizes mean / rstd redundantly in a fixed contributor order, then normalizes its shard in place into the output shard.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16], fp32_dest_acc_en = [True], layout = [TILE_LAYOUT];
  - placement = [op_layout], residual_mode = [residual, no_residual], output_mode = [new];
  - seq_alignment = [tile_aligned, seq_padded];
  - shape gate 32 ≤ M ≤ 128 with a plan of ≤ 4 rows a core.
- **Accuracy achieved**: PCC = 0.9999984, max_abs_err = 0.027–0.033 (≈ 1 bf16 ULP at |out| ≈ 4), relative RMS error = 0.0022–0.0023, max bf16 ULP = 2–4, got/true median 0.9987. Measured on 5 shapes × 2 residual modes via `test_nomic_residual_layer_norm_precision_baseline.py`.
- **Performance at Phase 0** (device kernel, 1350 MHz): 19.04 / 11.71 / 8.26 µs at 8x512 / 4x512 / 2x512 residual, and 17.61 µs at 8x512 no residual. That is 1.45–1.60× the incumbent, **short of the required 1.85×** (ceilings 16.1 / 10.1 / 6.5 / 14.7 µs). The op is compute-bound: removing the gamma/beta reads leaves the time unchanged.
- **Golden suite at Phase 0**: 46 / 154 cells passing. supported_pass 46, xfail_expected 108, supported_fail / xpass_drift / xfail_wrong_mode all 0 (per `verifier_report.json`).
- **Issues encountered (verifier fixes)**:
  1. Replaced the global `NUM_ROW_BLOCKS` (which asserted on any shape it did not divide) with a per-plan `MAX_BLOCK_ROWS` knob: `block_rows` = largest divisor of `rows_per_shard` ≤ the knob. Default unchanged.
  2. The compute kernel's CB indices and packed-stats slot count now come from host CT args, a single source of truth with the descriptor. `stat_slots` is a `Plan` field.
  3. Operand-contract violations in `validate()` (residual / gamma / beta mismatch) now raise `UnsupportedAxisValue` (a `SupportRefusal`) instead of `ValueError`, as the prompt's Validation section requires.
  4. `l1_ledger.md`: symbol table updated for the knob, and a verifier audit section added (including the measured gamma/beta-shadowing ablation and the `MAX_BLOCK_ROWS = 2` measurement).
- **Tests added**: `test_nomic_residual_layer_norm.py` (acceptance, planner), `test_nomic_residual_layer_norm_perf.py` (perf driver, implementer), `test_nomic_residual_layer_norm_precision_baseline.py` (verifier).

## Refinement 1 — Model split layout placement + in-place output
- Date: 2026-11-03
- What was done:
  - **`model_layout` placement** (BLOCK_SHARDED L1, COL_MAJOR, 8 core rows × C columns, shard (32·bh, 96), bh = ceil(M/11)).
    - The op replicates the model's layout formula (`model_memory_config`, `_model_split`) without importing models/;
      `test_model_layout_matches_model` checks equality with `sharded_norm_configs` for every M in 32..128.
    - `validate()` classifies it as `"model_layout"`.
    - `_plan(shape, device, placement)` reads the plan off the layout: a group is one grid column (G = `MODEL_HIDDEN_SPLIT` = 8, member = grid row = hidden slice), hidden_tiles = 3, column-major core order.
    - Mcast channels use `McastCoreOrder.ColumnMajor` groups.
    - x / residual / output are consumed zero-copy (`cb_descriptor_from_sharded_tensor`), and the last column's partial shard is processed at nominal size.
  - **Odd-width row sum**: hidden_tiles = 3 cannot fold in pairwise halves, so the S pass adds every tile to itself (2·S, exact) and the finalize scales S by fl(1/768)/2 (`inv_sum_bits`). The mean is bit-identical.
  - **Row blocking under an L1 budget**:
    - `block_rows` = the largest divisor of rows_per_shard whose CBs fit `CB_L1_BUDGET` = 256 KB (the `MAX_BLOCK_ROWS` knob stays a cap, default no cap).
    - Blocks are uniform. A ragged tail is not wrap-safe for the ring CBs, which several passes re-walk with linear indices.
    - The 8-member exchange is multi-round, one rendezvous per block.
  - **`in_place` output** (both placements): `output_tensor=x` backs `cb_output` on x's own shard and returns x. `"provided"` stays refused.
  - **Perf levers on the new path**, all kept:
    1. *Row-block lookahead* (software pipeline, lamp L2; `LOOKAHEAD` knob, active only when num_row_blocks > 1).
       - Phase A of block b+1 runs before block b's finalize, so block b's exchange overlaps it.
       - Normalize recomputes y = x + r from the resident shards, so cb_y stays one block.
       - The writer retires the output credits once, after the last exchange.
       - Lambdas are forced inline: not inlining them cost the op layout ~2%.
    2. *Pair tiles* (both placements): a packed stats tile carries S and Q of 2 rows (S in slots 0-1, Q in slots 2-3).
       - The finalize extracts every row's mean / rstd in one DEST window.
       - `cb_stats_final` and its fp32 round trip (slots 2-3) are deleted.
       - `cb_row_partials` doubles; op_layout M = 128 lands at exactly 256 KB.
    3. *Batched combine*: contributors are accumulated 3 per SFPU pass (D1..D3), still left to right in member order.
  - Exchange transport measured on the model layout: `serialized_chain` = `serialized_mcast` (identical, because on a 1×8 rectangle the helper multicasts in both). `concurrent_mcast` / `concurrent_unicast` were slower at the 1-block shape (15.4 / 16.0 vs 14.3 µs). Default kept: `MODEL_EXCHANGE_MODE = "serialized_chain"`.
- Accuracy achieved:
  - PCC = 0.9999984, rel RMS = 0.0022–0.0023, max bf16 ULP = 2–4, got/true median 0.99861–0.99868 (rtol / atol as golden: PCC ≥ 0.99999, RMS ≤ 0.005).
  - Shapes: 5 op_layout + 8 model_layout (one per bh class: 3 / 4 / 5 / 6 / 8 / 9 / 11 / 12) × 2 residual modes.
  - Bit-identical before / after the perf levers.
- Performance (device kernel, µs, 1350 MHz; one fresh run per variant):

  | Shape | Refinement 1 first cut | + lookahead | + pair tiles | + batched combine (final) |
  |---|---|---|---|---|
  | model 8x512 residual | 48.7 | 39.5 | 36.1 | **34.4** |
  | model 4x512 residual | 27.8 | 23.1 | 21.3 | **20.6** |
  | model 2x512 residual | 14.3 | 14.3 | 13.6 | **13.4** |
  | model 8x512 no residual | 46.2 | 36.4 | 32.8 | **31.5** |
  | model 7x512 residual (bh 11: 11 blocks of 1 row) | 95.0 | 66.1 | 54.0 | **53.0** |
  | op 8x512 / 4x512 / 2x512 residual, 8x512 no residual (Phase 0: 19.04 / 11.71 / 8.26 / 17.61) | 19.0 / 11.7 / 8.26 / 17.65 | 19.1 / 11.7 / 8.28 / 17.64 | 18.24 / 10.46 / 8.33 / 16.78 | **18.03 / 10.44 / 8.15 / 16.77** |

  - **The model-layout 8x512 bar (beat the incumbent's 29.7 µs) is NOT met: 34.4 µs, 1.16× slower.** No model-layout focus shape beats its incumbent yet (4x512: 20.6 vs 18.7 µs; 2x512: 13.4 vs 12.0 µs). The op-layout path gets faster on every focus shape.
  - Measured breakdown at model 8x512 (zones, TRISC1, 3 blocks): phase A 16.2k cycles (8.8k of it the per-row SFPU collapse at 3 tiles a row), finalize ~17k (bound by TRISC0 unpacking 8 full fp32 gather tiles a pair tile: G = 8 all-gather), normalize 12.5k (incl. the y recompute). The exchange is hidden (gather wait ≈ 1k cycles, against 18k before the lookahead).
- Golden test progress: 154/154 passing (was 46/154): all 108 former xfail_expected cells pass, 0 failures / xpass.
- Issues encountered:
  - The prime bh = 11 (M = 112, 121) only admits 1-row uniform blocks (11 exchanges a call). It is correct, but it is the slowest shape (53 µs at 7x512).
  - An off-by-one in the lookahead block offset slipped past the perf test's loose PCC (0.9999) on 8x512 (rows of different blocks have similar statistics on random data). The acceptance tolerance caught it. Lesson: always gate with the unit tests, never the perf driver.
- Tests added: `test_nomic_residual_layer_norm_placements.py` (model_layout × 8 bh classes × residual × new / in_place; op_layout in place; config equality with the model's; CB budget on both placements for every M); precision baseline extended with 8 model_layout shapes; perf driver gained a placement axis and 7x512.
- Next levers for the model layout (not done here, both scheme changes; Refinement 3 holds the model layout in its guard set):
  - (1) A reduce-scatter combine: member m receives only bytes [512m, 512m+512) of every contributor's pair tile, sums 1/8 of the vectors and all-gathers the finalized slice. That is 2 unpacks a pair tile instead of 8, and 1/8 of the rsqrt work.
  - (2) Lane-compacted statistics (SFPSHFT2 rotations) to pack 8× more rows a tile, shrinking both the gather and the per-row collapse.

## Refinement 2 — Speed up the perf-flagged profile: per-tile compute cost
- Date: 2026-11-03
- What was done:
  - **Measured breakdown first** (8x512 op_layout residual; zones; TRISC0 / 1 / 2 cycles of 24.3k total):

    | Stage | Cycles |
    |---|---|
    | residual add | 1.6k / 1.4k / 1.8k |
    | S | 0.7k / 1.0k / 0.7k |
    | Q | 2.6k / 2.7k / 2.3k |
    | collapse | 2.4k / 3.1k / 2.0k |
    | exchange wait | 3.9k (TRISC0) |
    | finalize | 2.1k / ~2.5k |
    | N1 (sub, col bcast) | 2.8k / 1.5k / 1.7k (unpack-bound) |
    | N2 (mul, col bcast) | 2.8k / 2.9k / 3.0k |
    | N3 (mul, row bcast) | 2.7k / 2.7k / 2.8k |
    | N4 (add, row bcast) | 1.7k / 1.6k / 1.4k |

    The per-tile stages sum to ≈ 15.4k against ≈ 13.5k of pure FPU math for 6.5 FPU passes a tile at HiFi4. That is near the floor of the formulation.
  - **HIDDEN_SPLIT sweep (lamp L1)**, measured with the new `test_nomic_residual_layer_norm_split_sweep.py` over the golden M classes (device µs, G = 2 vs G = 3):

    | M | G = 2 | G = 3 |
    |---|---|---|
    | 32 | 7.2 | 8.2 |
    | 33 | 7.2 | 7.9 |
    | 36 | 7.3 | 9.2 |
    | 39 | 7.8 | 10.3 |
    | 44 | 8.7 | 10.1 |
    | 48 | 9.5 | 10.1 |
    | 64 | 11.8 | 10.3 |
    | 80 | 11.7 | 14.8 |
    | 96 | 11.7 | 14.7 |
    | 128 | 18.4 | 18.2 |

    G = 4: 14.8 / 10.4 at 4x512 / 2x512, and 8x512 has no plan. G = 6: does not build (6 chain-unicast channels exceed NUM_SEMAPHORES), and 8x512 has no plan.
    The op_layout plan now picks G per shape (`HIDDEN_SPLIT_CANDIDATES = (3, 2)`, `SPLIT_COST_US` table keyed by (G, rows_per_shard), `_choose_hidden_split`). The kernels needed no change: G is already a CT arg, and 12 hidden tiles keep the pairwise-halves row sum.
  - **Normalize fusion (lamp L6)**, built and parked: `SEEDED_NORMALIZE` (default False). It is 2 chains of `UnaryBcast` seed + `BinaryFpu Mul`. ELWMUL accumulates into DEST, so the result is seed + a·b.
    - The finalize emits center = −mean·rstd with rstd rounded to tf32, so both terms use one factor.
    - The seed must be fp32 in DEST (cb_center_col on UnpackToDestFp32). Through srcB, constant rows fail (|out − beta| up to 2.9).
    - Measured: fp32-seed variant 23.3 µs; srcB-seed variant 18.28 µs (and incorrect); expanded-to-full-tile seed 24.9k vs 24.3k cycles.
    - The seed costs as much as the ADD stage it replaces, because stages here are unpack/FPU-bound, not L1-round-trip-bound.
  - **Block-size co-tune**: `ELTWISE_BLOCK_TILES` 4 (half-sync max) stays. 2 → 18.87 µs; `dst_full_sync_en` + 8 → 21.8 µs (math/pack overlap lost). Pre-expanding mean so N1 drops its column-broadcast unpack → 18.5 µs (the extra phase costs more than it saves). All reverted.
  - **Rename**: cb_mean_col → cb_center_col (mean, or −mean·rstd under the knob).
  - Reused: the existing plan / descriptor / kernels (G was already a CT arg); the eltwise chain (`UnaryBcast` + `BinaryFpu`) for the seeded path. Added: `_choose_hidden_split` + `_op_layout_rows`, the cost table, the `SEEDED_NORMALIZE` knob (CT arg 22), the sweep driver.
- Accuracy achieved: PCC = 0.9999984, rel RMS = 0.00224–0.00228, max bf16 ULP 2–4, got/true median 0.99861–0.99867 (golden gates PCC ≥ 0.99999, RMS ≤ 0.005). Shapes: precision baseline 5 op_layout (2x512 and 5x512 now G = 2) + 8 model_layout × 2 residual modes. Unchanged from Refinement 1.
- Performance (device kernel µs, 1350 MHz, one fresh run a variant):

  | Shape | Before | After |
  |---|---|---|
  | op 8x512 residual (flagged) | 18.10 | 18.08 |
  | op 4x512 residual | 10.42 | 10.40 |
  | op 2x512 residual | 8.24 | **7.21** |
  | op 8x512 no residual | 16.70 | 16.73 |
  | op 7x512 residual | 18.01 | 17.93 (first run 18.21; re-run 2x: 17.90 / 17.93, noise) |
  | model 8x512 / 4x512 / 2x512 residual | 34.4 / 20.6 / 13.4 | 34.3 / 20.4 / 13.3 |
  | model 8x512 no residual | 31.5 | 31.4 |
  | model 7x512 residual | 53.0 | 52.9 |

  - Off the focus set, golden M classes that switch to G = 2: M = 33–48 −0.7 to −2.5 µs; 5x512 / 6x512 / 8x352 / 8x384 14.8 → 11.7 µs.
  - The flagged 8x512 shape did not move; it stays at 18.1 µs against its 16.1 µs ceiling.
- Golden test progress: 154/154 expected; the op_layout slice was run here (76/76 passed). model_layout code paths are unchanged at the default knobs.
- Issues encountered:
  - The design's 400-cycles-a-tile budget is below the measured HiFi4 FPU floor (ELWMUL ≈ 84 cycles a tile, column-broadcast unpack ≈ 87). The per-tile slope cannot reach it without fewer multiplies.
  - The perf driver's `-k` expressions with spaces silently produce no CSV under `--profile` (the tracy wrapper escapes them). Use node IDs.
- Tests added: `test_nomic_residual_layer_norm_split_sweep.py` (HIDDEN_SPLIT sweep driver; forces one candidate, loose PCC).

## Refinement 3 — Speed up the perf-flagged profile: fixed per-call cost and exchange overlap
- Date: 2026-11-03
- What was done:
  - **Measured first.** Zones at 8x512 (one block of 4 rows): the TRISC0 gather wait was 3.8k cycles, collapse 3.0k, finalize 2.1k. A knob sweep (new `test_nomic_residual_layer_norm_knob_sweep.py`) over the exchange mode and row blocking found:
    - 2-row blocks with the existing lookahead schedule: 8x512 18.1 → 16.9 µs. The +3.5 µs recorded in Phase 0 predates the lookahead.
    - `concurrent_unicast` wins on single-round op plans (2x512 −0.3, 4x512 −0.4 µs).
    - Everything else was flat or worse: 1-row blocks, concurrent_mcast, concurrent on the model layout.
  - **Row-block software pipelining (lamp L2)**: `PIPELINE_BLOCK_ROWS = 2` in `_block_rows()`. A plan the L1 rule leaves as one block is split into 2-row blocks wherever that gives at least two (op_layout 4-row shards, M 109–128; model bh 4, M 34–44). Plans L1 already splits, odd row counts and 1-row results are unchanged. `None` restores the old plan. `SPLIT_COST_US[(3, 4)]` was updated to 16.9.
  - **Exchange transport (lamp L5)**: `EXCHANGE_MODE = "concurrent_unicast"` for the op layout. Multi-round plans already fall back to the serialized chain. `MODEL_EXCHANGE_MODE` stays serialized (re-measured).
  - **Parked lever**: `FUSED_RECOMPUTE` (CT arg 23, default off), the lookahead's y recompute fused into normalize stage 1 as (x − mean) + r (BinaryFpu Sub + DestReuseBinary Add, the library layernorm's pre-add pattern). It is correct, and measured slower at every lookahead shape.
  - Payload compaction (lamp L4) was not built. The single-round wait is rendezvous latency on a 4 KB slot, about 60 cycles of transfer.
  - Reused: the R1 lookahead schedule, the existing exchange modes and the eltwise chain. Added: the `PIPELINE_BLOCK_ROWS` plan rule, the per-plan mode default, the `FUSED_RECOMPUTE` branch, and two test files.
- Accuracy achieved: PCC = 0.9999984, rel RMS = 0.00223–0.00228, max bf16 ULP 2–4, got/true median 0.99861–0.99868 (gates PCC ≥ 0.99999, RMS ≤ 0.005). Precision baseline: 5 op_layout + 8 model_layout shapes × 2 residual modes. Unchanged.
- Performance (device kernel µs, 1350 MHz, second call of a fresh run):

  | Shape | Before | After |
  |---|---|---|
  | op 8x512 residual (flagged) | 18.10 | **16.82** |
  | op 7x512 residual | 18.02 | 16.93 |
  | op 4x512 residual | 10.36 | **9.98** |
  | op 2x512 residual | 7.17 | 7.05 |
  | op 8x512 no residual | 16.74 | **14.15** |
  | model 8x512 / 4x512 / 2x512 residual | 34.3 / 20.3 / 13.3 | 34.3 / 20.3 / 13.3 |
  | model 8x512 no residual / 7x512 | 31.4 / 52.9 | 31.4 / 52.9 |
  | model 4x288 residual (bh 4) | 15.2 | 14.8 |

  - Against the ceilings: 4x512 (10.1) and 8x512 no residual (14.7) now pass. 8x512 residual (16.1) misses by 0.7 µs, and 2x512 (6.5) by 0.55 µs.
  - What binds now: compute (8x512 T1 ≈ 23k zone cycles). Per-tile streams are ≈ 15.7k, collapse 3.3k, finalize 3.5k, and the exchange is hidden. 2x512 still exposes a 1.4k-cycle rendezvous.
- Golden test progress: 154/154 expected. A slice was run here: op_layout cells + 4x288 + regression, 86/86 passed. The model-layout code path is unchanged except bh 4, which is in the slice.
- Issues encountered:
  - `--profile` masks compile failures (a bad include passed as PASS). Check `UP_FRONT_COLLECT_RESULT` for `compile_errors`.
  - pytest node IDs with brackets do not resolve through the profile wrapper, so the sweep driver filters by the `NRLN_VARIANTS` / `NRLN_SHAPES` env vars instead.
- Tests added:
  - `test_nomic_residual_layer_norm_schedule.py`: back-to-back cached calls with different data (pins the single-round concurrent exchange against stale semaphores), and the pipelined-plan rule.
  - `test_nomic_residual_layer_norm_knob_sweep.py`: knob sweep driver with acceptance-level PCC.
