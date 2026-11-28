# Changelog: g6_interleaved_to_sharded

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Owner-fetch on the shard grid:
  - each shard owner reads its tile pages from DRAM straight into its L1 shard;
  - one NoC request per tile, a 544 B valid-row span on partial tile rows;
  - one read barrier per issuer;
  - no CB, no compute kernel, no writer kernel;
  - one `generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], rank=[2], alignment=[h_non_aligned], out_scheme=[width_sharded]
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, rms_err=0, max ULP=0, got/true ratio exactly 1 (measured on 4 shapes via test_g6_interleaved_to_sharded_precision_baseline.py).
- **Golden suite at Phase 0**: 4 supported_pass, 2 xfail_expected (rank 4 / height → Refinement 1), 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verify_results/verifier_report.json`). 9/13 tests pass overall. `test_refuse_uncovered_shard_spec` fails only in the eval L1-profiling harness (`peak_total=0` on a refusal test), not in the op.
- **Perf**: FLUX #58 p1 at 1101 ns profiled device kernel duration (golden harness: 1106 ns; stock ttnn 1.43 µs). Per core: ~233 ns issuing 5 reads and ~326 ns DRAM round trip. On top of that, ~450 ns of kernel-start skew across the 32 cores, which is outside the op.
- **Issues encountered** (verifier fixes):
  - `validate()` now refuses a width shard taller / height shard wider than the tensor with `ValueError`. TensorSpec cannot allocate such a shard; before the fix the call failed with a TT_FATAL after caching a program.
  - The cache key uses `device.id()` instead of the Python wrapper's `id()`.
  - The shard-multiple check derives the tile shape from the tensor.
  - `describe_plan()` returns a deep copy.
  - Argument loading moved inside the `I2S-FETCH` zone.
  - `KNOB_VALUES["issuers_per_core"]` is derived from `ISSUERS`.
- **Tests added**: test_g6_interleaved_to_sharded.py (acceptance), test_g6_interleaved_to_sharded_knobs.py, test_g6_interleaved_to_sharded_perf.py, test_g6_interleaved_to_sharded_precision_baseline.py, test_g6_interleaved_to_sharded_validation.py

## Refinement 1 — Height-sharded placement and rank-4 inputs
- **Date**: 2026-11-27
- **What was done**: Knob-turn, as planned. `SUPPORTED` now holds `rank: [2, 4]` and `out_scheme: ["width_sharded", "height_sharded"]`. No kernel or descriptor change was needed:
  - `owner_block()` already gives height shard `s` tile rows `[s*sh, min((s+1)*sh, rows))` with all tile columns;
  - `compute_geometry()` already folds leading dims into tile rows at `ceil(H/32)` rows per image;
  - the reader picks the partial-row span by `r mod image_tile_rows`.

  The height path holds the Phase 0 perf bar: one block per issuer covering the whole shard, every read in flight, one barrier.
- **Accuracy achieved**: bit-exact (PCC=1.0, relative RMS=0, NaN padding poisoned in the knob cases) on:
  - height: `(1,1,1,128)` p2 on 1 core; `(2,1,33,128)` on 4 cores; `(2,3,33,128)` on a 4x2 COL_MAJOR grid (6 owners, 2 idle); `(3,1,1,256)` whose last shard is partial;
  - rank-4 width: `(1,1,1,5120)` on 8x4; `(2,1,33,256)` on 4 cores.
- **Golden test progress**: the two former xfail_expected cells pass (`test_op[1x1x1x128-height_sharded…]`, loose `flux57_p2`), as does regression `test_height_multi_shard`. Golden: 6/6 cells supported_pass. I ran only a `-k` slice: the 2 height cells plus `flux58_p1` as the width representative.
- **Issues encountered**: None.
- **Tests added**:
  - `test_g6_interleaved_to_sharded.py`: 4 `test_placement` cases (multi-image COL_MAJOR height, partial last height shard, 2 rank-4 width) and `test_plan_report_height_map` (shard `s` on `G[s]`, idle cores past the last shard, short last block);
  - `test_g6_interleaved_to_sharded_knobs.py`: the 4-core `(2,1,33,128)` height scenario, bit-exact on all 6 `issuers_per_core x partial_tile_read` values.

## Refinement 2 — Speed up the perf-flagged FLUX #58 profile
- **Date**: 2026-11-28
- **What was done**: Perf refinement on the issue path. One kernel source, still one dispatch.
  - **Diagnosis** (p1, timing-only experiments, not committed):
    - The issue loop is CPU-bound at ~44 ns per read: ~20 ns of interleaved address math, the rest NIU register writes. Dropping the command-buffer poll saved only 27 ns.
    - Kernel-start times on BRISC and NCRISC are equal, so moving the single issuer to the other RISC buys nothing.
    - The stateful one-packet API saves nothing here: every page sits on a different DRAM bank, so every read must rewrite the coordinate.
  - **Lever 1, `split_reader`: default `issuers_per_core` 1 → 2.**
  - **Lever 2, `noc_placement`:**
    - New knob `issuer_placement`: four (RISC, NoC) orders, both issuers always on different RISCs and NoCs.
    - New knob `issuer_split`: `even` or `noc_balanced` (default). Measured:
      - NOC0 read responses share a grid column. 8 owners stacked in one column: 1 issuer on NOC0 takes 1334 ns.
      - NOC1 read responses share a grid row. 8 owners in one row: an even NOC0/NOC1 split takes 1093 ns, worse than 1 issuer at 979 ns.
    - `noc_balanced` therefore gives NOC0 `C/(R+C)` of each block (R / C = most owners in one grid column / row). p1 → 3/2, the 8x1 row → 18/2, the 1x8 column → 2/18.
  - **Lever 3, `bank_stagger`:** new knob (CT switch + per-issuer `col_rotate` RT arg). It rotates each tile row's column walk so the first page sits on bank `shard mod n_banks`. Measured flat/worse (on (32, 640): 979 → 1023 ns), so it stays parked at the byte-identical default False.
  - **Reused:** the reader kernel, `issuer_blocks()`, `owner_block()`, the cache-keyed `KNOBS` dict.
  - **Added:** the `ISSUER_PLACEMENTS` table, `noc_weights()`, a weighted `_split()`, and the `bank_stagger` CT switch and RT arg.
- **Perf** (profiled `DEVICE KERNEL DURATION`, 5-call medians, same session; this session launches faster than Phase 0's, so p1 old default reads 804 ns here vs 1101 ns then):

  | Placement | Old default (1 issuer) | New default (2 issuers, `noc_balanced`) | Change |
  |---|---|---|---|
  | p1 ROW_MAJOR | 804 ns | 758 ns | −5.7% (even split, same 3/2: 741 ns) |
  | (32, 640) 8x1 | 979 ns | 953 ns | −2.7% (no regression; even split would be +12%) |
  | (32, 640) 1x8 | 1334 ns | 985 ns | −26% |
  | (33, 5120) regime pin | 1770 ns | 1486 ns | −16% |

  Unprofiled trace replay, p1, 200 calls captured: 2708/2717 ns per call (1 issuer) vs 2719/2721 ns (new default). That is the same within 0.4%: back-to-back replay is dispatch-bound and hides the ~50 ns kernel gain, but shows no RB-053 regression (`probes/probe_012.py`).
- **Accuracy achieved**: bit-exact (PCC=1.0, relative RMS=0) with NaN-poisoned padding, on every `issuer_placement x issuer_split x bank_stagger` combination. Shapes: p1 `(1,5120)`, `(33,5120)`, `(1,5120)` on 8 cores `(32,640)`, height `(2,1,33,128)` on 4 cores.
- **Golden test progress**: 13/13 passing (6 test_op / test_op_loose cells + 7 regression). Unit dir: 207 passed.
- **Issues encountered**:
  - Under `--profile`, `-k` with spaces and `::node[id]` selectors are mangled by the tracy wrapper, so run the whole perf file.
  - One tracy capture crashed and dropped the first test's rows, shifting every label by one variant. Always check the row count (#tests x 5) before reading medians.
- **Remaining headroom (finding, not a follow-up)**:
  - On p1 the per-core wall is now ~2-3 reads of issue (~44 ns each) + one DRAM round trip (~320 ns) + launch skew. The read-ablated floor is ~250 ns.
  - `noc_balanced` over-weights contention on the 1x8 column (985 ns vs 912–944 ns for even), because issue cost per RISC also binds there. A blend of the even and contention weights is the next thing to try; it does not touch p1, whose split is 3/2 either way.
- **Tests added**:
  - `test_g6_interleaved_to_sharded_knobs.py`: `test_issue_path_knobs_bit_exact` (4 placements x 2 splits x 2 stagger x 4 scenarios), `test_one_issuer_bank_stagger_bit_exact`, `test_issuer_blocks_weighted`, `test_noc_weights`;
  - `test_g6_interleaved_to_sharded_perf.py`: scenarios p1 / p1_640 / p1_640_col / h33 and the new knob variants;
  - `probes/probe_012.py`: unprofiled trace-replay timing.
