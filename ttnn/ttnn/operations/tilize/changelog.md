# Changelog: tilize

## Phase 0 — Core Implementation
- **Date**: 2026-09-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - The `row_split_interleaved` regime is one `ttnn.generic_op` dispatch:
    - a custom stick reader (NCRISC / NoC0);
    - one `compute_kernel_lib::tilize` helper call per kernel (fast tilize);
    - a custom TILE-page writer (BRISC / NoC1);
    - work split over the runtime grid with `split_work_to_cores(row_wise=True)`.
  - Implementer perf pass: a per-core DRAM-bank traversal rotation (a win). A transaction-id read-ahead and a split reader were measured flat or slower and parked as live knobs.
  - Verifier: exposed the `tile_row` streaming window as a knob (`QUANTUM_MIN_TILES`, below).
- **SUPPORTED at Phase 0**:
  - `dtype=[bfloat16]`, `output_dtype=[bfloat16]`, `low_l1=[False]`
  - `shard_api=["none"]`, `out_scheme=["interleaved"]`, `buffer=["dram_to_dram"]`, `orientation=["none"]`
  - `rank=[4]`, `pad_mode=["none"]`, `pad_value=["none"]`, `alignment=[tile_aligned]`
  - `tile_height=[32]`, `in_tile_height=["none"]`
  - `tile_grid=[single_tile, small, tall_narrow]`
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0, mean_abs_err = 0, rms_err = 0, max ULP = 0, 0 bit mismatches. Measured on 5 cases (4 shapes, including one wide-exponent bf16 input) by `test_tilize_precision_baseline.py`.
- **Golden suite at Phase 0**: 30 supported_pass, 1746 xfail_expected, 2310 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode, out of 4390 rows (per `verifier_report.json`). `test_regression.py` has 10 tracked failures (fp32 / uint16 / int32 dtypes, not yet in SUPPORTED); Refinement 7 closes them.
- **Perf at Phase 0** (WH B0, 64 Tensix cores, median device-kernel ns): perf-focus [1,1,16384,64] ≈ 25.4–26.1 µs, against the LOOSE_CASES reference of 25998 ns.
- **Issues encountered (verifier fixes)**:
  1. The `tile_row` streaming window was collapsed to one tile-row per CB handshake, read barrier and write flush. It is now the `QUANTUM_MIN_TILES` → `rows_per_quantum` knob, which gives multi-tile-row quanta only where a tile-row is under 8 tiles, with at least `DEPTH_IN` quanta per core. Measured −3 % on [1,1,16384,64] and −5 to −8 % on [1,1,16384,32]; other paths unchanged.
  2. `validate()` mis-tagged `shard_api` at runtime, because an allocated legacy-sharded tensor also reports an `nd_shard_spec`. It now reads `created_with_nd_shard_spec`.
  3. DRY: removed the duplicated `TILE_WIDTH` and tile-grid `(R, C)` code from `tilize.py`, the derived `in_tile_bytes` CT arg, and the duplicated per-column CB byte formula.
  - The ledger and op_design.md block schedule were updated for the new window.
  - Harness issues reported but not edited: `axes.py:_spec_of` has the same ND mis-tag, and the `low_l1` A/B capture records the first leg. Missing INVALID entry: fp8_e4m3 × retile.
- **Tests added**:
  - `test_tilize.py` (acceptance, planner)
  - `test_tilize_knobs.py` (knob matrix; extended with quantum configs and a ragged multi-row shape)
  - `test_tilize_perf_shapes.py`
  - `test_tilize_precision_baseline.py`
  - `test_tilize_registry.py` (runtime shard tagging)
  - `test_tilize_perf_knob_sweep.py` (device-ns A/B harness, for `--profile`)

## Refinement 1 — Sharded and L1 placement (legacy 2-D, ND, crossovers, DRAM-sharded) + rank widening
- **Date**: 2026-09-23
- **What was done**:
  - Built the `sharded_resident` and `sharded_accessor` regimes on the existing reader / compute / writer (no new kernel file, no second descriptor branch).
  - **Core assignment and residency.** `tilize_program_descriptor._core_assignment` sets the core assignment from the first of:
    - a `resident_ok` L1-sharded output;
    - a `resident_ok` L1-sharded input;
    - the interleaved row split.

    Each side is then resident iff its shard rectangles (`_shard_rects`, all three schemes, both orientations, ragged final shards, ND specs with a 2-D equivalent) equal the assignment. A resident input backs `cb_input_sticks` on the shard, and the reader only publishes pages. A resident output backs `cb_output_tiles`: compute packs into it and the writer issues no NoC write. With the same spec on both sides the op moves zero NoC bytes.
  - **Streamed sides** (interleaved DRAM/L1, DRAM-sharded, cross-spec remote shards, ND without a 2-D equivalent) go through `TensorAccessor`. WIDTH / BLOCK / ND Layout::ROW_MAJOR inputs have shard-width pages, so the stick-segment reads split at page boundaries (new `page_bytes` / `pages_per_stick` CT args; the one-page-per-stick fast path is unchanged).
  - **Rotation.** A resident side pins the tile-row walk to shard order, so the per-core rotation is now two RT args, `row_rotation` and `stick_rotation`. The writer rotates its tile order inside a tile-row by `stick_rotation` (the write twin of the reader's stick rotation): HEIGHT-sharded in → DRAM [1,1,2048,512] went from 18746 to 16502 ns, and the interleaved paths are within noise.
  - **SUPPORTED** gains:
    - `shard_api` legacy_2d / nd;
    - `out_scheme` HEIGHT / WIDTH / BLOCK / nd;
    - `orientation` ROW_MAJOR / COL_MAJOR;
    - `buffer` dram_to_l1 / l1_to_l1 / l1_to_dram;
    - `rank` 2 / 3 / 5 / 6;
    - `tile_grid` short_wide / square_large, with EXCLUSIONS refusing both wherever the row split would be the core assignment (Refinement 5 lifts those).
  - Reused: Walker, StickProducer, store_rows, the tilize helper call, CB slots 0 / 1, `balanced_width` / `rows_per_quantum`. Added: shard-rectangle mapping, CB backing via `cb_descriptor_from_sharded_tensor`, per-page segment reads, `input_resident` / `output_resident` CT modes, the write-order rotation.
- **Accuracy achieved**: bit-exact (`torch.equal`, PCC = 1.0, atol = rtol = 0) at bf16 on every sharded / L1 / rank shape tested. That covers 26 unit cases (HEIGHT / WIDTH / BLOCK × ROW / COL, ragged shards, cross-spec, DRAM-sharded, ND 3-D, ranks 2–6) plus the golden and translated cells below.
- **Golden test progress**:
  - `test_golden.py`: 32 passed / 758 xfailed / 2310 skipped / 0 failed / 0 XPASS; 24 passes are new (Phase 0: 8 in this file).
  - Translated sharded / ND / L1 slice: 191 passed / 1 failed (below).
  - Sharded LOOSE_CASES, WH device-kernel ns (Tensix cores):

    | Case | Measured | Reference |
    |---|---|---|
    | HEIGHT in → DRAM | 16502 (64) | 16852 |
    | DRAM → HEIGHT out | 12269 (64) | 12142 |
    | Same spec | 1914 (64) | 1891 |
    | BLOCK COL_MAJOR | 1971 (16) | 1832 |

  - Perf focus [1,1,16384,64]: 25369 ns on 64 Tensix cores (unchanged).
- **Issues encountered**:
  - `eval/golden_tests/tilize/test_translated.py::test_tilize_program_cache_addr_change[sharded_width_l1]` asserts the first call builds exactly one program. In module order, `test_tilize_col_major_orientation`'s WIDTH case (COL_MAJOR on a 1×4 row grid, which places bytes identically to ROW_MAJOR) has already built the byte-identical program, so the first call is a correct hit (entries = 0). The test passes when run alone. I did not add an artificial CT arg to force a miss.
  - The translated square_large ND cases ([23,96,160]) and the fp32 / padded variants stay refused until Refinements 4, 5 and 7.
  - `test_tilize_registry.py` refusal assertions for sharded inputs became tag assertions, since sharding is now supported.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_sharded.py` (19 regime cases with residency assertions, a same-spec zero-traffic assignment check, rank 2/3/5/6 + L1 interleaved, L1 crossovers).

## Refinement 2 — Tile geometry: tiny output tiles + retile of a TILE input
- **Date**: 2026-09-23
- **What was done**:
  - **Tiny output tiles** (`tile_height` 16 / 8 / 4 / 2 / 1) on every existing placement. The kernels needed no change: both CBs already carried `TileDescriptor(tile_h, 32)`, the tilize helper leaves the fast path on its own, and the reader groups `tile_h` sticks per tile-row.
    - Host: `tilize._allocate_output` allocates the output through a `ttnn.TensorSpec` carrying `ttnn.Tile([tile_h, 32])`, with one overload per placement (interleaved, legacy 2-D shard, ND shard). `allocate_tensor_on_device` would lay out 32×32 tiles; the 32×32 path still calls it unchanged.
    - The `QUANTUM_MIN_TILES` floor now counts full 32-row tile equivalents (`× 32 / tile_h`). Otherwise a `tile_h = 1` quantum would be 8 single-stick tiles.
  - **Retile** (`in_tile_height` 32 / 16 / 8 / 4 / 2 / 1): the `retile_l1_facewalk` regime, a new reader block operation `read_retile` on the shared kernels. Compute and writer are unchanged.
    - **Unit.** A unit is `row_align` output tile-rows fed by `unit_in_rows = max(1, tile_h / in_tile_h)` input tile-rows of one image.
    - **Reads.** Input tiles are read whole, one page-sized NoC read each, into the reader-private `cb_retile_staging` ring (CB 3). `RETILE_STAGE_DEPTH - 1` units are prefetched under transaction ids.
    - **Resident input.** A resident Layout::TILE input shard backs `cb_retile_staging` and is read in place, with no NoC read into L1.
    - **Row split.** The row split and the walk rotation work in units of `row_align`, so each input tile is read once per Tensix core.
    - **Fallback.** H-padded TILE inputs and output shards that cut input tile-rows fall back to `row_align = 1`. That stays correct, with `in_tile_h / tile_h` read amplification.
    - No ROW_MAJOR tensor is materialized. The rejected designs (`retile_compute_untilize`, `retile_dram_facerow_reads`) were not built.
  - **Face-walk mover** (perf, measured). The RISC-V L1 word copy the design names measured ~8 cycles per word. On [1,1,16384,64] retile 32→16 it cost 93.5 µs, against 23.4 µs with the copy stubbed (ablation).
    - Face rows (16 elements) are now moved by NoC loopback reads on the same Tensix core (`noc_async_read_one_packet_with_state`), under their own transaction id. That id is drained before every `cb_input_sticks` push and before a staging slot is refilled.
    - `RETILE_FACEWALK_NOC` defaults on; the RISC-V copy stays a live, tested knob.
  - Reused: Walker (rotation), the CB slots, compute, writer, `_core_assignment`, `balanced_width` / `rows_per_quantum`. Added: `FaceWalk`, `RowSlotWriter`, `retile_source_of`, `read_retile`, `cb_retile_staging`, the `RETILE_STAGE_DEPTH` / `RETILE_FACEWALK_NOC` knobs, and `_allocate_output`.
  - Ledger: `cb_retile_staging` rows and the retile data-movement budget. op_design.md: the regime is marked built, with an implementer note.
- **Accuracy achieved**: bit-exact (`torch.equal`, PCC = 1.0, atol = rtol = 0) at bf16 on every case tested. That covers:
  - tiny tiles at all 5 heights on [1,1,64,128], [2,3,64,64] and [1,1,4096,64] (64 Tensix cores), plus HEIGHT / WIDTH / BLOCK same-spec shards;
  - retile on 9 height pairs × 3 shapes, H-padded TILE inputs ([2,1,48,64], [3,2,48,96]), L1-interleaved, same-spec resident shards (3 schemes × 4 pairs), and crossovers (a resident TILE shard in → DRAM out, and DRAM in → 16-row output shards that cut 32-row input tiles);
  - every retile knob setting.
- **Golden test progress**:
  - `test_golden.py` targeted slice: 14 passed / 0 failed / 0 XPASS (117 xfailed are non-bf16 dtype cells, Refinement 7). That covers all 3 `tile_geometry_tiny` cells, all 6 `tile_geometry_retile` cells including `retile_1_to_32` (the required `in_tile_height: 1 × tile_height: 32` cross), and `PROGRAM_CACHE_CASES` `tiny_tile_16`, `retile_32_to_16`, `baseline_dram_dram`, `tall_narrow_grid_scale` and `height_sharded_same_spec`.
  - Translated tile-geometry tests: 18 passed, 22 xfailed. The xfails are fp32 (Refinement 7) and the [1,1,128,256] tiny-tile cells that tag `square_large` (Refinement 5). `test_tilize_retile` is `skip_for_wormhole_b0` upstream, so the unit file covers those shapes on WH instead.
- **Perf** (WH B0, device-kernel ns, [1,1,16384,64] bf16, DRAM interleaved, 64 Tensix cores):

  | Path | Before | After |
  |---|---|---|
  | Perf-focus 32×32 | 25369 | 25498 (median of 3, no regression) |
  | Tiny 16 | — | 23980 |
  | Tiny 8 | — | 24917 |
  | Tiny 1 | — | 62725 |
  | Retile 32→32 | 93423 (RISC-V copy) | 37906 (NoC face walk) |
  | Retile 32→16 | 93523 | 38311 |
  | Retile 16→32 | 93822 | 35659 |
  | Retile 1→32 | 116201 | 59457 |

  - Tiny 1 is writer-bound (64-byte output pages).
  - Remaining retile headroom: the NoC face walk still issues one 32-byte loopback read per face row (1024 per Tensix core here). NCRISC takes 31 µs, against 17 µs for the reads alone. Two ways to cut it further: split the face-walk issue across both data-movement RISC-Vs (that needs a second input CB, as the split reader does), or re-lay face-to-face with runs of `min(face_h_in, face_h_out)` rows. The second would bypass the tilize compute, which is outside this regime.
  - 1→32 is bound by the 64-byte input tile reads (39 µs even with the face walk stubbed).
- **Issues encountered**: none blocking.
  - The RISC-V copy's cost was the only surprise, and ablation (payload stubbed, sync kept) attributed it before the NoC mover replaced it.
  - The `--profile` wrapper splits a `-k` expression on spaces, so use single-token filters.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_tile_geometry.py`, 102 cases:
  - tiny tile DRAM and sharded;
  - retile DRAM, H-padded, L1, resident shards (with residency assertions), crossovers;
  - a retile knob matrix, program-cache cases, and a perf-shape case for `--profile`.

## Refinement 3 — Speed up the perf-flagged profile (block-quantum / depth / NoC co-tune)
- **Date**: 2026-09-23
- **What was done**: I built and measured every lever the refinement names, plus two the measurements pointed to. The perf-focus shape, [1,1,16384,64] bf16 DRAM interleaved on 64 Tensix cores (WH B0), did not get faster. Every lever is correct (bit-exact), parked at a default that leaves the default path's kernel code and CB sizes unchanged, and still a live knob in `tilize_program_descriptor.py`, which records its numbers. All figures below are median device-kernel ns over 5–10 dispatches.
  - **Classification (ablations, payload stubbed, synchronization kept).** Full 25.4 µs; reads only 16.1 µs; writes only 19.2 µs; no transfers 8.1 µs. The no-transfer floor is mostly NCRISC address generation: 6.5 µs for 256 stick addresses per Tensix core. Reads and writes nearly add up, on separate NoCs, so they share a resource. On 32 Tensix cores the op takes 30.0 µs, only 17 % slower than on 64 (48 cores: 28.7 µs; 56: 25.3 µs). The shape is bound by aggregate DRAM throughput for its transaction mix: 128-byte stick reads plus 2 KiB tile writes, about 165 GB/s including the fixed costs. The per-core schedule does not bind.
  - **Zones.** The reader issues about 42 cycles per 128-byte read uncontended ([1,1,256,64], 8 cores) and about 53 at 64 cores; its barrier waits total only 1.8 µs. The writer's first write waits on the first CB quantum. `/perf-ceiling-dm`: the DRAM floor is 4 MiB / 288 GB/s = 14.6 µs (15.8 µs at 92 %). The `noc_estimate` binary is not built on this box, so the contended DRAM keys came straight from `noc_latencies.yaml`: `ALL_FROM_ALL` for 256 × 128-byte reads is about 18.8k cycles and `ALL_TO_ALL` for 16 × 2 KiB writes about 29k cycles, both slower than measured. The practical reference is the `double_buffer` example's 190.8 GB/s for a 64-core copy (reads plus writes of 2 KiB tiles), which puts this shape at about 88 % of what its transaction mix can reach.
  - **Quantum × depth × read-ahead co-tune** (new knobs `READ_WINDOW_MIN_TILES` / `WRITE_WINDOW_MIN_TILES`). Each window becomes `read_ahead` / `write_ahead` CB quanta; `depth_in` / `depth_out` grow to hold them, and a budget loop keeps them inside `CB_BUDGET_BYTES`. Sweep on the perf shape:

    | Tile-rows per quantum | Depth | Read-ahead | Median |
    |---|---|---|---|
    | 1 | 4 | 4 | 24.7–25.1 µs |
    | 1 | 8 | 8 | 27.5 µs |
    | 2 | 4 | 4 | 27.1 µs |
    | 4 (baseline) | 2 | 1 | 25.0–25.9 µs |

    The best windowed rule (1 tile-row per quantum, windows at 8 full tiles) measured 24.7–25.0 µs against a 25.3–25.5 µs baseline in four runs, then 25.8 vs 25.5 µs in a fifth. On other shapes it regressed: [1,1,16384,32] 18.0 → 19.7 µs, [1,1,32768,64] 51.7 → 54.4 µs, [1,1,4096,64] 7.7 → 8.0 µs. Only [4,3,256,96] won clearly (9.0 → 8.1 µs). Parked at 0.
  - **Eager publish** (`EAGER_PUBLISH`). The reader pushes slots whose reads have landed with a non-blocking transaction-id poll. It fixes a real pathology: `StickProducer` completes slots only when forced, so a read-ahead equal to the core's slot count serializes the core (all reads, then tilize, then writes). That case goes from 27.4 to 25.1 µs. At the default schedule it is flat (25.5 vs 25.3 µs). Parked.
  - **Write-side batching** (the write-ahead window, `TileStorer`). Up to `write_ahead` quanta of tile writes are in flight, each tagged with its own NoC transaction id and flushed oldest-first. It is the write twin of the read-ahead and was measured together with it: 24.7–25.3 µs vs 25.3–25.6 µs, flat. I also tried write bank spreading (rotating the write order across the quantum), which cut the simulated worst-case per-bank load from 12 to 6 Tensix cores: flat (25.2 vs 25.5 µs), so I removed it.
  - **NoC / stream placement: one stream split across both NoCs** (`READ_NOC_SPLIT` / `WRITE_NOC_SPLIT`, with the data-movement kernels in `DM_DYNAMIC_NOC`). Dynamic mode alone is neutral (26.4 vs 25.9 µs) once each kernel's NoC is set explicitly; `NOC.RISCV_1_default` is NoC1, which first put the reader on the wrong NoC at 55 µs. Reads on NoC1 run against the DRAM geometry: 1 in 2 sticks 36.0 µs, 1 in 8 33.4 µs. Writes on NoC0: 1 in 2 tiles 45.4 µs, 1 in 8 26.2 µs. Parked at 0.
  - **Read issue cost** (`BANK_STRIDE`). Stick `p + NB` sits in stick `p`'s bank, one aligned page further, so each Tensix core needs only NB accessor calls; mode 2 also issues each tile-row bank by bank. The no-transfer floor dropped 8.1 → 7.2 µs, but reads only got slower (16.1 → 18.9 µs; 19.5 µs bank-major), because faster request issue congests the DRAM banks. The full op stayed flat (25.2 / 25.8 vs 25.4 µs). Parked; it is the addressing a Refinement 6 bank-coalesced read can build on.
  - **Correctness guard.** One NoC transaction id counts at most 255 outstanding reads, so the stick reader now caps `rows_per_quantum` at `255 // (tile_h × reads per stick segment)`. The cap does not bind on any measured shape.
  - Reused: `StickProducer`, `store_rows`, `Walker`, the CB slots, `rows_per_quantum` / `balanced_width`. Added: the window / depth derivation, `TileStorer`, `publish_landed`, `issue_row_bank_stride`, the NoC-split paths, and five knobs.
- **Accuracy achieved**: bit-exact (`torch.equal`, PCC = 1.0, atol = rtol = 0) on every knob setting across the 7 knob-matrix shapes (168 cases), and at defaults on every unit test.
- **Golden test progress**: `test_golden.py -k test_op_loose` 8 passed / 3 xfailed / 0 failed, covering the perf-focus cell and the sharded / fp32 LOOSE_CASES. The unit regression net is all green: `test_tilize.py` 22, `test_tilize_knobs.py` 168, `test_tilize_sharded.py` 26, `test_tilize_tile_geometry.py` 102, registry 3, precision 5, perf shapes 5.
- **Perf, no regression at defaults** (WH B0, device-kernel ns, median of 7; before → after):

  | Guard path | Before | After |
  |---|---|---|
  | Narrow DRAM [1,1,16384,64] | 25186 | 24963 |
  | Wide DRAM [1,1,8192,256] | 44410 | 44294 |
  | L1 interleaved [1,1,4096,64] | 5651 | 5691 |
  | Sharded resident, HEIGHT same spec [1,1,2048,512] | 1908 | 1930 |
  | Sharded accessor, HEIGHT in → DRAM | 16761 | 16350 |
  | Tiny tile 16 [1,1,16384,64] | 24351 | 24209 |
  | Retile 32→16 [1,1,16384,64] | 38026 | 38110 |

  All on 64 Tensix cores, all within noise. The perf-focus shape stays at about 25.0–25.5 µs, below the 25998 ns WH reference.
- **Issues encountered**:
  - Device-kernel time on this shape varies ±3 % from run to run. Single-run "wins" of 2–3 % did not survive repeats, so every decision above rests on 10-rep medians in both orders.
  - `ttnn.NOC.RISCV_1_default` is NoC1, not the reader's NoC0.
  - Retile's `read_retile` issues `unit_in_rows × valid_width` reads under one transaction id, which can exceed 255 in principle (for example 1→32 on a wide row). It is not a failure seen so far; it is noted for a later refinement.
- **Tests added**:
  - `test_tilize_knobs.py`: 12 Refinement 3 configs (windows, write-ahead, budget shrink, eager publish, NoC splits, bank stride / bank-major).
  - `test_tilize_r3_perf.py`: the device-ns harness. Variants and shapes come from env vars, plus a `_GRID` core-count probe and the 7-path guard set `test_r3_guard`.

## Refinement 4 — Padding: auto / explicit pad, all fill signs, non-aligned H / W, rank 0 / 1
- **Date**: 2026-09-23
- **What was done**:
  - **SUPPORTED.** Added `pad_mode` auto / explicit, `pad_value` zero / positive / negative, `alignment` w / h / hw_non_aligned, and `rank` 0 / 1.
  - **Host (`tilize.py`).**
    - `_resolve_padding` computes the padded shape `P`: auto rounds the last two dims to (`tile_h`, 32), and ranks 0 / 1 synthesize the tile dims (`[] → [32, 32]`, `[W] → [32, round_up(W, 32)]`); explicit takes `output_padded_shape`.
    - The output is allocated at `P`, and `R`, `C` come from `P`.
    - The returned tensor is a zero-copy view at the input's logical shape (`_logical_view`, i.e. `ttnn.reshape(out, logical, padded)` → `tt::tt_metal::view`). The buffer address is checked, so it can never become a second dispatch.
    - The fill value is packed per input dtype (`_fill_bits`) into RT arg 10, so fills of different values share one cached program.
  - **Reader (`tilize_stick_reads.hpp`, `tilize_reader.cpp`).**
    - `PadMap` is the per-image stick map, with a mixed-radix leading-dim decode.
    - `StickProducer::issue_row_padded` reads only the existing sticks' data bytes. It then fills the W-tail band and the trailing whole pad sticks: the H tail, and whole pad tile-rows / images.
    - `PadFill` stores the fills. Short ranges use `fill_l1_range`, or a band split computed once per tile-row. Long ranges are NoC loopback copies from a 1 KiB reader-private `cb_pad_source` that is filled once, under their own transaction id and drained before every push.
    - With one column block per core, the W-tail band is filled only on the first pass through the CB ring (`PAD_W_TAIL_PERSIST`).
    - Compute and writer are unchanged. The unpadded path compiles to the same code: `issue_row` now goes through `open_row` / `close_row` with a no-op fill.
  - **Placements.**
    - A padded input always streams.
    - A resident output shard over `P` works as before.
    - A legacy-sharded output `MemoryConfig` with no shard spec now derives it from the input's shard grid and orientation over `P` (`_resolve_output_memory_config`, needed by the translated `tilize_with_val_padding` sharded cases).
    - Tiny tiles pad against `tile_h`.
  - **Refused.**
    - Retile combined with a pad that has something to fill is an EXCLUSION (`_retile_pad_exclusions`): the face walk has no fill yet. An auto pad of an aligned TILE input fills nothing and runs.
    - Growing an inner leading dim ([2, 3, …] → [3, 4, …]) raises `NotImplementedError`. TTNN's logical view maps logical image k to padded image k, so no buffer satisfies both the logical and the `F.pad` padded readback; I verified this on host with a correct `F.pad` buffer. Growing the outermost indexed leading dim, and growing the rank, both work.
  - **Reused / added.** Reused: `StickProducer`, `Walker`, the CB slots, `rows_per_quantum` / `balanced_width`, `_allocate_output`, the whole writer and compute. Added: `PadMap`, `PadFill`, `issue_row_padded`, `cb_pad_source`, `PadSpec`, and three knobs (`PAD_SOURCE_BYTES`, `PAD_NOC_MIN_BYTES`, `PAD_W_TAIL_PERSIST`).
- **Accuracy achieved**: bit-exact at bf16 (`torch.equal`, PCC = 1.0, atol = rtol = 0), on both the logical readback and the padded readback against `F.pad`. Covered shapes: [1,1,32,50], [1,1,50,64], [1,1,50,50], [1,1,30,32], [50,50], [3,50,64], [1,2,1,50,50], [64], [50], [], [2,3,70,100], [4,1,1000,72], explicit growth up to [1,1,128,128] and [3,1,32,64] / [2,64,64], L1, HEIGHT-sharded output, WIDTH-sharded input, and tiny tiles 16 / 8 / 1.
- **Golden test progress**:
  - `test_golden.py`: 15 of the 19 bf16 → bf16 padding / degenerate-rank cells pass. That is all of `padding_auto` (7), `padding_explicit` (4), `padded_to_height_sharded`, `padded_l1_to_l1`, `rank1` and `rank0_scalar`, plus `test_program_cache_reuse[auto_hw_tails_negative_fill]`; 0 failed, 0 XPASS. The 4 still xfail on Refinement 5 axes: `padded_low_l1`, `short_wide_single_stick`, `short_wide_w_tail`, `square_large_from_leading_dims`.
  - `test_translated.py`: 723 passed; the only failure is `test_tilize_program_cache_addr_change[sharded_width_l1]`, the known module-order case recorded in Refinement 1.
  - `test_regression.py`: the 10 tracked fp32 / integer failures, unchanged (Refinement 7).
- **Perf** (WH B0, 64 Tensix cores, device-kernel ns):
  - The unpadded guard set is unchanged within noise (5-rep medians): narrow DRAM [1,1,16384,64] 25621, wide DRAM 43783, tiny tile 16 24614. Single runs: L1 interleaved 5838, sharded resident 1918, sharded accessor 16317, retile 32→16 38045.
  - Padded path: [1,1,16370,50] auto (H and W tails) 36.6 µs vs 25.3 µs aligned [1,1,16384,64]. Stubbing the W-tail fill gives 27.6 µs, so the band stores (~27 cycles per stick on NCRISC) are what is left.
  - [1,1,65520,50] 118.8 µs vs aligned [1,1,65536,64] 108.6 µs, and 129.7 µs with persistence off.
  - [1,1,8192,64] → [1,1,16384,64] (half the tile-rows whole pad): 22.0 µs, vs 49.5 µs with CPU-only fills.
  - [1,1,16384,40] → [1,1,16384,128]: 48–49 µs vs aligned [1,1,16384,128] 46–47 µs.
- **Issues encountered**:
  - A non-dependent `static_assert` inside a discarded `if constexpr` branch of `kernel_main` is still evaluated; it broke the split-reader knob build. Moved to top-level conditional asserts.
  - A first-pass whole-pad tile-row band-filled its whole segment (32 redundant loopback reads per row, 22 → 38 µs on the half-pad shape). The band is now only for existing sticks.
  - An upfront W-tail prefill serialized ahead of the first reads (34 → 38 µs), so it was replaced by the lazy first-pass fill.
  - The output shard spec derivation for spec-less sharded outputs was missing entirely. It surfaced here because those translated cases were previously refused on `pad_mode`.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_padding.py` (60 cases):
  - auto / explicit / rank 0 / 1 / leading-dim and rank growth / L1 / sharded out / width-sharded in / tiny tiles;
  - W-tail alignments;
  - a fill-knob matrix;
  - program cache across fill values;
  - the two refusals (inner leading-dim growth, retile × fill) and retile × an auto pad with nothing to fill;
  - perf shapes and A/B tests for `PAD_W_TAIL_PERSIST` and the fill mover.
