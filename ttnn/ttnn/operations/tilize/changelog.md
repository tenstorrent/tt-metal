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

## Refinement 5 — 2-D grid split (short_wide, square_large) + low_l1
- Date: 2026-09-23
- What was done:
  - **SUPPORTED.** `low_l1` is now `[False, True]`. The `short_wide` / `square_large` row-split EXCLUSIONS (`_row_split_wide_exclusions`) are deleted, so both values run on every placement.
  - **`grid_2d_split` regime (host only, kernels unchanged).** They already took `(row_start, core_row_tiles, col_start, core_col_tiles)`. `tilize_program_descriptor.grid_2d_split(R, C, N)` transcribes the pinned rule and replaces `NUM_COL_GROUPS`: minimize the busiest core's cost; tie-break 1 wider column groups; tie-break 2 fewer Tensix cores; every column start a multiple of `col_align_tiles`; row groups in units of `row_align` (retile).
    - One measured amendment: the cost is `rows * (cols + ROW_COST_TILES)` with `ROW_COST_TILES = 1.5` (0 = the pinned tile-count rule, a live knob). Every tile-row costs `tile_h` stick reads whatever its width. The pure tile count put 1-column groups on shapes the row split already filled: [4,3,256,96] 8.9 → 11.0 µs, [1,1,2080,2048] 92 → 181 µs.
    - `g_c == 1` falls through to the unchanged `split_work_to_cores` row split, so the perf-focus path compiles to the same program.
    - `g_c > 1` puts row group `k // g_c` × column group `k % g_c` on the first `g_r * g_c` Tensix cores, row-wise.
  - **Transaction-size lamp** (`MIN_GROUP_COL_TILES`, a column-group floor): measured 1 / 2 / 4 / 8 / 16 tile-columns. Maximum participation wins, so it is parked at 1 (live knob).
  - **Quantum re-measure.** On short_wide every Tensix core owns one walk position, so `rows_per_quantum` is pinned to 1 and `QUANTUM_MIN_TILES` cannot engage.
  - **New lever: `PIPELINE_MIN_POSITIONS` (2) + `PIPELINE_MIN_SEGMENT_BYTES` (2048).** It targets the one-position serialization: a core with one walk position cuts its columns into up to 2 blocks, so read / tilize / write overlap in the depth-2 CBs, but only while each block keeps at least 2 KiB stick segments.
- Accuracy achieved: bit-exact (bf16 → bf16, `torch.equal`; PCC = 1.0, rtol = atol = 0) on [1,1,32,2048], [1,1,64,4096], [1,1,32,8192], [1,1,1,2048] (auto pad), [1,1,32,4090] (auto pad), [1,1,2048,2048], [8,1,249,2048] (auto pad), [2,3,64,1024], [1,1,96,64], [1,1,2048,64], L1-interleaved [1,1,32,2048]. `low_l1=True` vs `False` is bit-identical on [1,1,64,256], [1,1,32,4096], [1,1,32,8192], [1,1,50,50] (padded) and [1,1,4096,1024].
- Golden test progress:
  - `test_golden.py`: 74 passed, 0 failed, 0 XPASS, 716 xfailed (all dtype / output_dtype, Refinement 7). Every `work_geometry` and `low_l1` cell passes, including `short_wide_l1_forcing` and `low_l1_forcing_width` at both settings with no OOM, and `PROGRAM_CACHE_CASES` `short_wide_canonical`, `tall_narrow_grid_scale` and `square_large`.
  - `test_translated.py`: 2 failures, both module-order program-cache hits that pass when run alone:
    - the known `test_tilize_program_cache_addr_change[sharded_width_l1]` (Refinement 1);
    - `test_tilize_with_val_padding_block_per_node_cb_size[1.0-input_shape5]`, newly admitted (short_wide). (2,1,60,7328) folds to the same R = 4, C = 229 program as the earlier (1,1,100,7328) case, and its pad map rides on RT args, so its "first" call is a correct cache hit.
  - `test_regression.py`: the 10 tracked dtype failures, unchanged (Refinement 7).
- Perf (WH B0, device-kernel ns, Tensix cores in parentheses):
  - short_wide_canonical [1,1,32,2048]: 14273 (1) → 3640–3828 (64); ref 3486.
  - [1,1,32,8192]: 30847 (1) → 7105–7505 (64); ref 7142.
  - [1,1,64,4096]: 20149 (2) → 7292–7342 (64).
  - square_large [1,1,2048,2048]: row split kept on 64 cores (tie on makespan); the pipeline lever takes it from 92293 to 87033 (−5.7 %, medians of 3).
  - [1,1,96,64]: 3133 (3) → 2927 (6).
  - [1,1,128,64]: flat, 3118 (4) → 2894–3141 (8).
  - Guard set unchanged within noise: narrow DRAM 25191, wide DRAM 43131, L1 interleaved 5683, sharded resident 1921, sharded accessor 17270, tiny tile 16 25103, retile 38324.
- Issues encountered:
  - The pinned tile-count rule regressed shapes the row split already filled (see above), fixed by `ROW_COST_TILES`.
  - An ungated column pipeline lost 30–50 % on short_wide (narrower reads), fixed by the 2 KiB segment floor.
- Tests added: `tests/ttnn/unit_tests/operations/tilize/test_tilize_grid_2d.py` (32 cases):
  - the rule, pinned and with row cost, plus row_align;
  - bit-exact work-geometry shapes, with a Tensix-core-count assertion (≥ 3/4 of the grid on short_wide / square_large / tall_narrow);
  - padded short_wide / square_large;
  - an L1-interleaved 2-D split;
  - the `low_l1` A/B bit-identity and the CB-budget bound;
  - the 2-D program cache.

## Refinement 6 — Speed up the perf-flagged profile (bank-coalesced stick reads)
- **Date**: 2026-09-23
- **What was done**:
  - **Reader.** New `bank_coalesced` load_block (`read_bank_coalesced`, `tilize_reader.cpp`), the lever the refinement names. Stick page `p` lives in DRAM bank `p mod NB` at offset `(p div NB) * page`, so a run of consecutive tile-rows' sticks in one bank are contiguous there.
    - Per CB quantum (a unit), each run of consecutive tile-rows is read with one NoC read per bank into a reader-private staging ring (bank-major), up to `depth - 1` units ahead under per-slot transaction ids. Rotated walks wrap at most once inside a unit.
    - Each core rotates its starting bank by its core index.
    - Once a unit lands, NoC loopback reads (one packet per stick, `set_state` once) move each stick to its tilize position in the reserved `cb_input_sticks` slot. After their barrier the slot is pushed.
    - Compute and the writer are unchanged.
  - **Host (`tilize_program_descriptor.py`).** The path is gated: DRAM `TensorMemoryLayout::INTERLEAVED` input, one page per stick, no pad / retile / split reader / resident input, and every Tensix core reading whole sticks in one column block (`block_width == C`). Sticks must be at most `BANK_COALESCE_MAX_STICK_BYTES` = 256 bytes, the StickProducer-only levers must be at their defaults, and one tile-row of staging must still fit `CB_BUDGET_BYTES[low_l1]`.
    - The staging CB reuses the retile staging slot (index 3; the regimes are disjoint).
    - The staging bytes are counted in the `rows_per_quantum` budget cap and in the window loop.
    - On this path the quantum is `BANK_COALESCE_QUANTUM_ROWS` = 2 tile-rows (full 32-row equivalents).
  - **Knobs** (all live): `BANK_COALESCE_STAGE_DEPTH` (2; 0 = off, i.e. StickProducer), `BANK_COALESCE_QUANTUM_ROWS` (2), `BANK_COALESCE_MAX_STICK_BYTES` (256), `BANK_COALESCE_SCATTER_WRITE` (False, parked).
  - **Measurements** (WH B0, 64 of 64 Tensix cores, device-kernel ns, median of 3; off → on):
    - Perf shape [1,1,16384,64]: 25967 → 23303.
    - [1,1,16384,32]: 18170 → 13347.
    - [1,1,32768,64]: 51477 → 46344.
    - Flat: [1,1,4096,64], [4,3,256,96], [1,1,16384,128], [1,1,8192,256].
    - [1,1,2048,1024] (2 KiB sticks): +7.6 %, hence the stick-bytes gate.
  - **Quantum sweep, in tile-rows.** [1,1,16384,64] 1 / 2 / 4 rows: 24449 / 23303 / 24304. [1,1,16384,32] 2 / 4 / 8 rows: 13347 / 14459 / 14287. [1,1,32768,64] 1 / 2 / 4 rows: 49436 / 46344 / 46774. Staging depth 3 measured flat-to-worse vs 2.
  - **Ablations on [1,1,16384,64]** (payload stubbed, synchronization kept):
    - The no-transfer floor fell 8.1 → 2.3 µs: `NB` accessor calls per run instead of one per stick.
    - Reads only: 16.1 → 12.0 µs, flat across per-bank read sizes of 3 to 21 sticks.
    - Scatter only: 8.1 µs, ~22 cycles per stick whatever the stick bytes. It costs ~1.3 µs of the wall (full 23.9 vs 22.6 µs with the scatter stubbed).
    - Writes only: 17.8 µs, and 17.2 µs on 32 Tensix cores vs 17.0 µs on 64.
    - Reads and writes still roughly add up, so the op is bound by aggregate DRAM throughput for its read + write mix.
  - **Writer twin (measured, not built).** A timing-only ablation wrote each tile-row's 2 tiles as one 4 KiB bank-contiguous transaction. Writes-only went 17.0 → 28.3 µs. Larger writes do not help, and the writes do not scale with core count, so bank-coalesced writes have no headroom here. Loopback *writes* for the scatter also measured slower (scatter-only 8.8 vs 8.1 µs), so they are parked behind `BANK_COALESCE_SCATTER_WRITE`.
  - **Reused / added.** Reused: `Walker`, the CB slot scheme, the retile staging slot and CT arg, the NoC-loopback `set_state` / `with_state` pattern from `FaceWalk`, the transaction-id prefetch loop shape from `read_retile`, and the `rows_per_quantum` caps. Added: `read_bank_coalesced`, reader CT args 26 / 27, the gate, the staging CB, 4 knobs.
- **Accuracy achieved**: bit-exact (`torch.equal`, PCC = 1.0, atol = rtol = 0) on every shape and knob setting tested. That covers the 7 knob-matrix shapes × 6 new coalesce configs (depth 1 / 3, quantum 1 / 3 / 4 tile-rows, scatter-write, the wide-stick gate opened, a tiny budget) and the 11-path guard set at both settings.
- **Golden test progress**:
  - `test_golden.py -k "test_op_loose or test_program_cache_reuse"`: 19 passed / 1 xfailed / 0 failed.
  - `-k dram_to_dram` bf16 → bf16: 45 passed / 0 failed / 0 XPASS.
  - Unit net all green: `test_tilize.py` 22, `test_tilize_knobs.py` 210, `test_tilize_padding.py` 60, `test_tilize_grid_2d.py` 32, `test_tilize_tile_geometry.py` 102, `test_tilize_sharded.py` 26, registry 3, precision 5, perf shapes 5.
- **Perf guard set** (`test_tilize_r3_perf.py::test_r3_guard`, now A/B-able via `TILIZE_R3_VARIANTS`; WH B0, 64 Tensix cores, median of 3; off → on):

  | Guard | Off (ns) | On (ns) |
  |---|---|---|
  | narrow DRAM [1,1,16384,64] | 25670 | 23509 |
  | narrow W=32 [1,1,16384,32] | 18488 | 13285 |
  | tiny tile 16 | 24706 | 23184 |
  | `low_l1` narrow | 24624 | 23736 |
  | 2-D split [1,1,32,2048] | 3783 | 3678 |
  | L1 interleaved | 5763 | 5692 |
  | sharded resident | 1917 | 1914 |
  | retile 32→16 | 37973 | 38254 |

  Four guards run identical kernels at both settings because the gate does not select the coalesced path for them: wide DRAM [1,1,8192,256] (43851 / 45058), sharded accessor (16031 / 16519), wide 2 KiB sticks (45955 / 46231) and retile. Their spread is run-to-run noise. No regression.
- **Issues encountered**: the new path's `static_assert` first sat inside a discarded `if constexpr` branch of the non-template `kernel_main`, where it is still evaluated. It fired on split-reader configs; I hoisted it as a conditional assert. Pytest `-k` is case-insensitive, so `not INT` also deselects "interleaved".
- **Tests added**: `test_tilize_knobs.py` gains 6 coalesce configs (42 cases, including `coalesce_off`, which keeps StickProducer covered on the narrow DRAM shapes the coalesced path now takes by default). `test_tilize_r3_perf.py`: the guard test takes `TILIZE_R3_VARIANTS` and gains the 2-D split, `low_l1`, W=32 and 2 KiB-stick guards.

## Refinement 7 — Numerical formats: fp32 / fp8 / integer inputs, block-float and integer outputs
- **Date**: 2026-09-23
- **What was done**:
  - `SUPPORTED["dtype"]` gains `float32`, `fp8_e4m3`, `uint32`, `int32`, `uint16` and `uint8`. `SUPPORTED["output_dtype"]` gains `float32`, `bfloat8_b`, `bfloat4_b`, `uint32`, `int32`, `uint16` and `uint8`. fp8_e4m3 is declared unconditionally (Blackhole-only, per the feature spec) and is not verified on this Wormhole box.
  - `cb_input_sticks` carries the input dtype and `cb_output_tiles` the output dtype, as before. The value-preserving cast happens at pack.
  - New `tilize_program_descriptor.NumericConfig` is the one source of the compute config for a dtype pair:
    - `fp32_dest_acc_en = True` whenever either page is Float32, Int32, UInt32 or UInt8. Measured: UInt8 through a 16-bit DEST packs every datum as 0.
    - `fp32_dest_acc_en` is never set for UInt16. Measured: fp32 DEST scrambles UInt16, PCC ~0.
    - A 32-bit input tags the compute input CBs (`cb_input_sticks`, plus `cb_input_sticks_odd` under the split reader) `UnpackToDestFp32`, and compute CT arg 5 selects `Fp32Mode::Lossless`. Tilize is the final consumer, so the fast path's fp32 → tf32 truncation would corrupt the output.
    - bf16 → bf16 (the perf-focus path) keeps `fp32_dest_acc_en=False`, fast tilize and the default unpack modes.
  - `compute_kernel_config` (optional `ttnn.ComputeKernelConfig`) is exposed on `tilize()` and threaded to the descriptor. It can only add fp32 DEST. math_fidelity / math_approx_mode pass through and have no effect (test-pinned).
  - The pad fill gains an fp8_e4m3 encoding. Integer fills already bit_cast signed → unsigned at the element width.
  - Nothing else changed: `col_align_tiles` (uint8 → 32-byte tile-columns), `per_col_tile_bytes` and the `block_width` cap (fp32 → fp32: 32 tiles) already derive from the element and tile sizes.
  - `BFP_PACK_PRECISE` (bfp8_pack_precise for block-float outputs) was built, measured and parked off. It makes fp32 → bfp8 bit-identical to host quantization on randn. But a rank-0 datum that bfp8 represents exactly came back off by 0.0078, and a bfp4 pad cell fell to PCC 0.9799.
  - **Reused / added.** Reused: every kernel, CB, regime and knob; the one tilize helper call (now with an explicit `Fp32Mode` template argument); `_fill_bits`; `validate()`. Added: `NumericConfig` (plus the `WIDE_DTYPES` / `FP32_DEST_DTYPES` / `FP32_DEST_FORBIDDEN_DTYPES` / `BFP_PACK_PRECISE` constants), compute CT arg 5, the `compute_kernel_config` kwarg, and 6 EXCLUSIONS.
- **EXCLUSIONS (each measured on WH B0, with its category)**:
  - Packer / LLK: block-float output at tile [16, 32] (`{output_dtype: bfloat8_b | bfloat4_b, tile_height: 16}`). Each mantissa row comes back paired with the next row's exponent (PCC ~0). Tile heights 32 / 8 / 4 / 2 / 1 match host quantization. bfp8_pack_precise and fp32 DEST do not change it, and bf16 at [16, 32] is exact.
  - Precision (format): `{output_dtype: bfloat4_b, rank: 0 | 1}`. The packer truncates to bfp4, giving PCC ~0.984 on randn versus ~0.993 with the host's rounding. A single mostly-pad tile lands at or below the 0.98 floor, and rank 0 is a single datum with no PCC.
  - Oracle (unrepresentable): `{dtype: uint16 | uint8, pad_value: negative}`.
    - uint16: the device writes the bit_cast (65536 − n), but uint16 reads back zero-extended into int32.
    - uint8: `F.pad` raises on a negative fill for a uint8 tensor.
- **Accuracy achieved**:
  - Bit-exact (`torch.equal`) for every same-dtype pair (fp32, bf16, uint32, int32, uint16, uint8), for bf16 → fp32 and for int32 ↔ uint32.
  - fp32 → bf16 is also bit-exact (lossless tilize, then round at pack).
  - bf16 / fp32 → bfp8: PCC ≥ 0.99994.
  - bf16 / fp32 → bfp4: PCC ≥ 0.9818.
  - Shapes: the 8 precision-matrix shapes (32x32 … 256x2048, plus W / H / both non-aligned with pad), randint over the full bit-pattern space for the integers, and every golden scenario.
- **Golden test progress**:
  - Every non-bf16→bf16 `test_op` cell: 3025 selected → 717 passed, 18 xfailed (the new EXCLUSIONS), 2290 skipped as INVALID / fp8-on-WH, 0 failed, 0 XPASS.
  - `test_op_loose` + `test_program_cache_reuse`: all pass.
  - bf16 → bf16 `test_op` (55): all pass.
  - `test_regression.py`: 10/10 (were the 10 tracked failures: integer passthrough, extreme magnitudes, pad-value extremes).
- **Perf** (WH B0, 64 Tensix cores, device-kernel ns, 3 fresh runs, median):
  - bf16 perf focus [1,1,16384,64]: 23644 (23206 / 23703 / 23644). Refinement 6 recorded 23303, so this is within the ~2 % noise band. The compiled bf16 kernels are unchanged apart from one unused compute CT arg.
  - fp32 LOOSE reference [1,1,8192,32]: 14055, against the 15064 reference.
  - fp32 [1,1,16384,64]: 47474, 2.0× the bf16 time for 2× the bytes, so DRAM-bound as before.
  - int32 [1,1,16384,64]: 48275.
  - bf16 → bfp8 [1,1,16384,64]: 19126 (fewer output bytes).
  - uint8 [1,1,16384,64]: 13937.
- **Issues encountered**:
  - uint8 needed fp32 DEST.
  - uint16 must not have fp32 DEST (found by the precision matrix's requested-fp32-acc axis).
  - The tiny-tile block-float packer mismatch (above).
  - The harness's up-front-collect precompile pass prints fake-device metrics before the real run. Only the second half of a `-s` log is real.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_numeric_formats.py` (604 cases: `test_tilize_precision_matrix`, `test_tilize_fidelity_is_noop`, `test_tilize_integer_extremes`, `test_tilize_numeric_pad_fill`, `test_tilize_numeric_perf_shape`). Also `precision_matrix_results.md`.

## Refinement 8 — Speed up the perf-flagged profile (post-generality re-tune)
- **Date**: 2026-09-23
- **What was done**:
  - **Re-measured the flagged shape: no regression, and it is at the DRAM roofline.**
    - [1,1,16384,64] bf16 DRAM interleaved runs on 64 Tensix cores. A same-session A/B of this run's Refinement 6 commit against HEAD, 5 fresh runs each, gives medians of 23642 ns (R6) and 23489 ns (HEAD). The compiled path is the same. R6's recorded 23303 sat at the low end of a ±3 % noise band.
    - Empirical roofline (new `test_tilize_r8_roofline.py`): a native `ttnn.clone` of the same tensor as `Layout::TILE`, DRAM → DRAM on 64 Tensix cores, moves the same bytes (2 MiB read + 2 MiB written) in 26644 ns (157 GB/s). The same clone as `Layout::ROW_MAJOR` takes 121214 ns.
    - Tilize is therefore ~12 % faster than the native tile copy. It moves ~180 GB/s, against the 190.8 GB/s the `double_buffer` example measured on a larger copy, so its ceiling is ≈ 22.0 µs (≤ 5 % headroom).
    - `noc_estimate` is not built in this checkout (it needs `--build-tests`), so this empirical copy is the ceiling used.
    - Writes remain the binding stage (writes-only 17.8 µs, Refinement 6). The parked split reader would load BRISC with reads on top of those writes, so it was not revisited.
  - **Took the lever the roofline leaves: the tiny-work lamp.**
    - Device zones on [1,1,128,64] (8 Tensix cores, one tile-row × one tile each, ~2990 ns; reference 2294) show a serial chain:
      - NCRISC's issue loop for the 32 stick reads takes 1462 cycles (~45 cycles per `noc_async_read`: five command-buffer register writes plus the ready poll);
      - the read barrier takes ~300 cycles, tilize ~200, and the write + barrier ~480;
      - BRISC idles in `cb_wait_front` for ~2000 cycles.
    - Ablations: the no-transfer floor is 929 ns, writes-only 1440, reads-only 2498.
    - Cheaper address math doesn't help: `BANK_STRIDE` 1 / 2 measured 3102 / 3075 ns, still 37 cycles per read.
    - The parked `READ_NOC_SPLIT` doesn't help either: 2 → 4381 ns, 4 → 4543 ns, because DM_DYNAMIC_NOC counters cost ~76 cycles per read.
  - **The co-read lever (new).**
    - On a walk of exactly one position per Tensix core, BRISC (NoC1) reads the last `CO_READ_SHARE` (0.5) of the tile-row's sticks straight into `cb_input_sticks`' first slot. It then raises a core-local program semaphore after its read barrier.
    - NCRISC reads the rest and waits for that flag before its push (`CoReadLanded`, passed as `StickProducer`'s Fill). NCRISC stays the CB's only producer.
    - Both halves use one new helper, `read_tile_row_sticks`, in the same per-core rotated stick order. `StickProducer::issue_row` now calls it too, and `read_paged_segment` moved to a free function.
    - Gate: `max_positions == 1`, the plain stick walk (not padded / retile / resident / split reader / bank-stride / NoC split), and stick segments within `CO_READ_SEGMENT_BYTES[input BufferType]`: DRAM ≤ 128 bytes, L1 unbounded. Co-read takes precedence over `bank_coalesced`, which has nothing to overlap its scatter with on one position (4-core row split of [1,1,128,64]: 3.6 µs coalesced vs 3.1 µs plain).
  - **Why the gate.** BRISC's NoC1 share of DRAM reads loses once reads stop being issue-bound:

    | DRAM stick segment | Co-read vs off (medians of 3) |
    |---|---|
    | 192 bytes | +4.6 % |
    | 256 bytes | +3 to +6 % |
    | 512 bytes to 2 KiB | +26 to +31 % |

    An L1-interleaved source wins at every width. On [1,1,2048,W], 64 Tensix cores: 128 bytes −14 %, 512 bytes −24 %, 1 KiB −17 %, 2 KiB −28 %.
  - **Share sweep** on [1,1,128,64], `CO_READ_SHARE` 0.375 / 0.4375 / 0.5 / 0.5625 / 0.625: 2591 / 2402 / 2297 / 2521 / 2447 ns. 0.5 is best on all three probe shapes.
  - **Knobs** (live): `CO_READ_SHARE` (0.5; 0 = off, byte-identical kernels), `CO_READ_SEGMENT_BYTES`, `CO_READ_SEM`.
  - **Measurements** (WH B0, device-kernel ns, median of 3 fresh runs; co-read off → on):

    | Shape (bf16, DRAM interleaved unless noted) | Tensix cores | Off | On | Change |
    |---|---|---|---|---|
    | [1,1,128,64] | 8 | 3122 | 2307 | −26 % (reference 2294) |
    | [1,1,2048,32] | 64 | 4307 | 3413 | −21 % |
    | [1,1,2048,64] | 64 | 6087 | 5552 | −9 % |
    | [1,1,64,2048] | 64 | 5442 | 4972 | −9 % |
    | [1,1,32,2048] | 64 | 3659 | 3445 | −6 % (reference 3486) |
    | [1,1,2048,256], L1 interleaved → DRAM | 64 | 14282 | 10844 | −24 % |

  - **Reused / added.** Reused: `StickProducer` (its Fill hook carries the landed wait), `Walker`, the per-core stick rotation, the writer's input accessor CT args (already there for the split reader), and the r3 perf harness. Added: `read_tile_row_sticks`, `CoReadLanded`, reader CT args 28 / 29, writer CT args 17–19, one program semaphore (only when co-read engages), 3 knobs, and the gate.
- **Accuracy achieved**: bit-exact (`torch.equal`; PCC = 1.0, rtol = atol = 0) on every co-read shape and share (the knob matrix: 9 shapes × 4 new co-read configs). HEAD and R7 outputs are bit-identical on 60 seeded float32 → bfloat4_b cases (20 seeds × [1,1,50,50] padded, [1,1,128,64], [1,1,32,2048]).
- **Golden test progress**:
  - `test_op_loose` + `test_program_cache_reuse`: 20/20 passed.
  - `test_op -k "single_tile or short_wide or small"`: 681 passed, 18 xfailed (EXCLUSIONS), 2100 skipped, 1 failed.
  - The failure is `1x1x50x50-pad_auto` float32 → bfloat4_b, pad negative: PCC 0.9789–0.9799 against the 0.98 floor. It fails on ~1 run in 3 because the golden input is unseeded randn. It is pre-existing: that cell is padded, so co-read never engages, and HEAD matches R7 bit for bit on seeded inputs. It is the bfloat4_b truncation near-miss Refinement 7 recorded. Not silenced.
- **Perf guard set** (`test_r3_guard`, co-read off → on, median of 3): `tiny_one_position` 3045 → 2329, `l1_one_position` 14282 → 10844, `grid_2d_short_wide` 3711 → 3472. The other 10 guards run identical kernels at both settings and sit within ±3 % noise (e.g. narrow_dram 23538 / 23902, wide1024_dram 46702 / 46501).
- **Issues encountered**:
  - The first gate was ungated by segment size: [1,1,2048,1024] DRAM went 46186 → 60423 ns. Fixed by the DRAM ≤ 128-byte gate.
  - A height-sharded L1 input going to an interleaved output is consumed resident, so co-read never runs there. My first "height-sharded" sweep was therefore pure noise (±7 %, now documented as the control).
  - The probe's `import ttnn.operations.tilize.tilize` resolves to the function, not the module (use `sys.modules`).
- **Tests added**:
  - `test_tilize_r8_roofline.py`: `test_dram_copy_roofline` (native clone ceiling) and `test_co_read_l1_source` (the L1 gate sweep, with a height-sharded resident control).
  - `test_tilize_knobs.py`: 2 one-position shapes and 4 co-read configs (306 cases).
  - `test_tilize_r3_perf.py`: guards `tiny_one_position` and `l1_one_position`.
