# Changelog: g6_pad

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Two regimes, one device program per call:
  - `extension_band`: an in-place band write over both DM RISCs, with a metadata-only view as the output.
  - `new_storage_tiles`: an NCRISC reader plus a BRISC writer that assemble each output page once in L1.
- **SUPPORTED at Phase 0** (after the verifier drift fix):
  - dtype = [bfloat16, float32, int32, uint32], layout = [TILE], memory_config = [dram, l1];
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned], rank = [2, 3, 4], padded_dims = [leading, h, w, hw].
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0, rms_err = 0, max ULP = 0. This is bit-exact, measured on 4 shapes (bf16 extension ×2, bf16 new_storage with a −2^127 fill, fp32 p1) via `test_g6_pad_precision_baseline.py`.
- **Golden suite at Phase 0**:
  - 78 / 85 tagged cells pass, with 7 `xfail_expected` (all ROW_MAJOR) and 0 loud categories, with `EVAL_CAPTURE_L1=0`.
  - The default runner shows 74 pass plus 4 `supported_fail`. Those 4 are a harness L1-profiling artifact on host-side refusals (see verification_report.md).
- **Perf at Phase 0** (BH, device ns): p1 28,327 (goal < 29,910), p2 11,894, p3 5,050, p4 10,189.
- **Issues encountered / fixed by the verifier**:
  - Three stale test files from an earlier attempt broke collection of the test directory. Two were rewritten against the current API and the redundant profile test was removed.
  - The L1 cap hardcoded the buffer depth `2`; it now uses the `input_depth_blocks` knob (DRY).
  - SUPPORTED drift was fixed (`xpass_drift` on the sharded-refusal cell). Probe evidence showed 17/17 gated TILE cells passing, plus int32 TILE.
  - `l1_ledger.md` was brought up to date (the 32-bit cap is 164, the cap expression depends on the depth knob, and the page formats cover all dtypes).
- **Tests added**:
  - `test_g6_pad.py` (acceptance, by the planner);
  - `test_g6_pad_extra_cases.py` and `test_g6_pad_knobs.py` (rewritten; knob test covers the multi-block depth-2 path);
  - `test_g6_pad_dtypes.py` (fp32 / int32 / uint32 / rank 2 / L1);
  - `test_g6_pad_precision_baseline.py`;
  - `test_g6_pad_perf_ab.py` (p1 added).

## Refinement 1 — ROW_MAJOR layout (`new_storage_rows` regime)
- Date: 2026-11-27
- What was done: built the design's `new_storage_rows` regime natively, with no host-side layout conversion and one device program per call. `SUPPORTED["layout"]` now holds `[TILE, ROW_MAJOR]`.
  - Reused: `select_regime`, `make_plan`, the descriptor cache, `build_new_storage_descriptor`, the NCRISC reader / BRISC writer kernels, `OutTileWalker`, the `cb_input_tiles` nominal-push block discipline and `cb_fill_tile`.
  - Added: a `rows_mode` CT flag in the shared reader/writer. A unit is (output row, row chunk): the walker's tile row is a row and its tile column is a 64 B-aligned row chunk, and `source_row()` / `out_row()` were added to the walker.
  - Added knobs `row_chunk_bytes` (0 = auto) and `min_row_chunk_bytes` (1024). In auto mode each row is split into enough chunks that rows × chunks covers the grid, with the chunk L1-capped. This puts `[5,8192]` on 112 units and `[5,3,2560]` on 120 units, both spread over 110 cores.
  - Reader: reads only the input part of each source-bearing chunk into the start of its slot. Fill-only rows and chunks are never read.
  - Writer: fill-only units are written from the fill page before the input wait. The row tail `[d_w, D_w)` is filled in place: RISC stores up to the next 16 B boundary, then one local NoC copy from the fill page. Every output page is written once.
  - The fill page larger than 4 KB is RISC-seeded and grown by local NoC doubling. Tile pages are ≤ 4 KB, so the tiles path is unchanged.
- Accuracy achieved: bit-exact on all cells (exact gate, PCC = 1.0, RMS = 0). Shapes: the 6 golden RM cases, plus `[3,7,45]` bf16 hw, `[2,3,33,1000]` fp32 multi-chunk + tail, `[4,600]` fp32 L1, `[1,1,2,4097]` i32, `[200,64]` bf16, and a 9-way chunk/block knob sweep.
- Golden test progress: all 7 former `xfail_expected` ROW_MAJOR cells pass, including the loose `special_values` cell on 262. In the targeted slice (RM + TILE representatives), 17/17 pass. The unit dir passes 93/93.
- Perf (BH, device ns, 110 cores): `[5,8192]` fp32 RM 2,856; `[5,3,2560]` bf16 RM 1,590; `[1,1,196,196]` bf16 RM 1,485.
- Issues encountered: none in the op. The Tracy wrapper splits a multi-word `-k` expression, so use single-token filters with `--profile`.
- Tests added: `test_g6_pad_row_major.py` (11 regime-pinned cases plus a knob sweep). `test_g6_pad_perf_ab.py` gained 3 ROW_MAJOR guard cases (`-k rm`).

## Refinement 2 — Speed up the perf-flagged profile p1 (pipeline knobs)
- Date: 2026-11-27
- What was done: tuned and added pipeline knobs for `new_storage_tiles` / `new_storage_rows`. Every knob has one source (`KNOBS`) and is carried through `make_plan`. There is no restructure and no new CB.
  - Reused: the existing `block_tiles` / `input_depth_blocks` knobs and the multi-block ring, the writer's NoC-doubling fill-page path (Refinement 1), and `OutTileWalker`.
  - `block_tiles` 0 → **2** at depth 2, so the read of block k+1 overlaps the store of block k. It is path-gated by the new knob `pipeline_min_units` = 8: a core with fewer units runs its range as one block, as before.
  - New knob `bank_stagger` = **1**: `RotatedCursor` in `g6_pad_common.hpp` walks core i's range starting at the unit on bank `i mod n_banks` (n_banks from `ttnn.get_memory_view`) and wraps. The reader and writer get the same rotation as an RT arg, so slot k matches on both sides. 0 = the in-order walk.
  - New knob `fill_seed_bytes` = **256**: only the first 256 B of the fill page are RISC stores, and the rest is grown by local NoC doubling. Zones showed the whole-tile RISC fill cost 2.3 µs on every writer (4 KB fp32), and it now costs 0.5 µs.
  - New knobs `reader_noc` / `writer_noc` (defaults 0 / 1, unchanged). Measured on p1: writer on NoC0 40.4 µs, reader on NoC1 29.6 µs, both swapped 38.1 µs, against 26.7 µs for the defaults.
- Perf (BH, 110 cores, device kernel ns):
  - p1 `[5,12,128,128]` fp32: **29,300 → 26,780** (median of 3; −8.6%, 1.21× the 22.19 µs roofline, under the 29.91 µs limit).
  - Guard set, before → after:

    | Case | Before (ns) | After (ns) | Change |
    |---|---|---|---|
    | p2 (extension path, untouched) | 11,896 | 11,810 | noise |
    | p3 (extension path, untouched) | 5,036 | 5,064 | noise |
    | p4 (extension path, untouched) | 9,767 | 9,907 | noise |
    | `[1,3844,1536]` (extension path, untouched) | 1,981 | 1,971 | noise |
    | 299 L1 (extension path, untouched) | 2,019 | 2,015 | noise |
    | `[1,196,128]` | 2,041 | 1,289 | faster |
    | `[1,1,196,196]` hw | 3,030 | 2,270 | faster |
    | case 63 `[1,196]` u32 | 4,413 | 2,708 | faster |
    | RM `[5,8192]` | 2,867 | 2,070 | faster |
    | RM `[5,3,2560]` | 1,613 | 1,368 | faster |
    | RM `[1,1,196,196]` | 1,487 | 1,46–1,54 | noise |

  - The RM `[1,1,196,196]` case read 1.63 µs at block 2 before the `pipeline_min_units` gate was added; the gate fixed it.
  - Most of the small-shape wins come from `fill_seed_bytes`.
- Remaining bottleneck (finding): per-core end times rise steeply with logical x inside every grid row, from about 8 µs at x=0 to about 26 µs at x=10. This holds for copy cores and fill-only cores alike. The wall is NoC/DRAM placement skew, not the copy/fill imbalance. `bank_stagger` alone measured within noise on p1 (block 3: 27.0 vs 27.1 µs). It is kept on because it is correct and does not regress. Next lever: a cost- and x-aware split that gives east-side cores fewer units (Refinement 3's load-balance scope).
- Accuracy achieved: bit-exact on every case (exact gate, PCC = 1.0, RMS = 0). Shapes: p1, the guard set, and every knob set in `test_g6_pad_knobs.py`.
- Golden test progress: targeted slice 18/18 pass (p1, case 63, hw 196/185/207, `[1,196,128]`, all ROW_MAJOR cells). The unit dir passes 106/106.
- Issues encountered: the Tracy capture tool sometimes exits with code 3 before writing a trace. Re-running is enough; the test itself had passed.
- Tests added: `test_g6_pad_knobs.py` gained `stagger_block3` (rotated walk wrapping across blocks, 64 B fill seed) and `phase0_schedule`. The existing multi-block sets now pin `pipeline_min_units=1`. `test_g6_pad_perf_ab.py` gained guard cases 63 (uint32) and 299 (L1).

## Refinement 3 — Speed up the perf-flagged profile p1 (copy/fill load balance)
- Date: 2026-11-27
- What was done: built the design's "load balance (new_storage_tiles)" lever as live knobs. The defaults keep the Refinement 2 schedule.
  - Reused: `OutTileWalker`, `RotatedCursor` (bank_stagger), the block ring, the N3 fill pass and the descriptor plumbing.
  - Added (host): `split_mode` (0 = equal count, `split_work_to_cores`, the default). Mode 1 is a cost-weighted contiguous split: `_source_prefix` counts the kernels' source-bearing predicate in closed form, and `_cost_split` places the boundaries by binary search, weighted by `copy_cost` / `fill_cost`. Mode 2 gives each core a cost-weighted slice of the region holding the source units (output dim-0 index < input dim 0) and an even slice of the new dim-0 (fill-only) region. The core set is the same single rectangular row-wise range in every mode (BB-17).
  - Added (kernels): `MixCursor` / `SegmentCursor` in `g6_pad_common.hpp`. A `two_segments` CT flag selects it, so mode 0/1 compile the old `RotatedCursor` walk and read no extra RT args. Two rotated segments are interleaved with Bresenham, and reader and writer share the same walk.
  - Added (writer): `fill_upfront` CT knob. It hoists N3 into one pass over the whole range before block 0.
- Perf (BH, 110 cores, device kernel ns, median of 3 where noted):
  - p1: 26,830 at the defaults, against 26,780 after Refinement 2 (noise).
  - Rebalance variants on p1 (all slower):

    | Variant | Device ns |
    |---|---|
    | mode 1 at copy/fill cost 2/1 | 28,480 |
    | mode 1 at cost 1/2 | 27,605 |
    | mode 1 at cost 2/3 | 27,500 |
    | mode 2 | 27,610 |
    | mode 2 with `fill_upfront` | 28,107 |
    | mode 0 with `fill_upfront` | 27,114 |

  - Ablation:

    | Ablation | Device ns | Note |
    |---|---|---|
    | Write-only (reads stubbed) | ~22,500 | does not move under cost or position weights (22,700–22,900) |
    | Read-only (writes stubbed) | ~11,400 | |
    | Write-only with writes on NoC0 | ~38,000 | mirror-image skew |

  - Guard set (defaults; extension path untouched):

    | Case | Device ns | Change |
    |---|---|---|
    | `[1,196,128]` (≤ 32 tiles) | 1,384–1,399 | 1,361–1,376 on the same day without this change: noise band |
    | `[1,1,196,196]` hw | 2,241 | no regression |
    | case 63 | 2,667 | no regression |
    | RM `[5,8192]` | 2,052 | no regression |
    | RM `[5,3,2560]` | 1,403 | no regression |
    | RM `[1,1,196,196]` | 1,507 | no regression |

- Finding (remaining headroom): the copy/fill imbalance is not the bound. The op is bound by the DRAM write stream, and its write-only floor (22.5 µs) is already roughly the 22.19 µs roofline. The ~4 µs left is read/write interference.
  - The per-core end-time skew (west/north early, east/south late) comes from NoC1 arbitration order, not per-core work. Giving east cores fewer units with 1/T position weights does not move the wall.
  - Spreading reads over every core (mode 2) is slower than concentrating them on ~69 cores.
  - The levers stay as tunables at their byte-identical defaults.
- Accuracy achieved: bit-exact (exact gate, PCC = 1.0, RMS = 0). Shapes: p1 in modes 0 and 2, every knob set in `test_g6_pad_knobs.py` (including the new dim-0 + hw shape `[3,2,50,70]`), and the RM split sweep.
- Golden test progress: targeted slice 11/11 pass (p1 strict + loose, `[1,1,196,196]`, `[5,3,2560]` RM, `1x196*`). The unit dir passes 147/147.
- Issues encountered: run-to-run noise on p1 is about ±0.7 µs, so every comparison above is a median of 3. Posted writes were rejected without a measurement: they give no completion guarantee at the op boundary.
- Tests added: `test_g6_pad_knobs.py` gained the `split_cost`, `split_cost_fill_heavy`, `split_mix`, `split_mix_block3`, `fill_upfront` and `split_mix_fill_upfront` knob sets and the `lead_hw_3x2x50x70` shape. `test_g6_pad_row_major.py` gained `test_g6_pad_row_major_split` (modes 1, 2 and 2 + `fill_upfront` on the rows regime).
