# Changelog: pad

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier), for Quasar
  on the ttsim simulator: one `ttnn.ProgramSpec`, one `generic_op` dispatch, reader + writer DM kernels, no compute
  kernel; data rows and pad rows split as two lists over the full grid; thread plan (R, C, W) = (1, 0, 1).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR], memory_config=[DRAM], sharding=[interleaved],
  pad_value=[0.0], padding_side=[after], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[4]
  (= the specification's PHASE0)
- **Accuracy achieved**: exact on 4 shapes (`test_pad_precision_baseline.py`): PCC 1.0, max_abs_err 0.0,
  rel_rms_err 0.0, 0 ULP, got/true ratio 1.0 everywhere.
- **Golden suite at Phase 0**: 14 / 14 supported cells pass, 4 / 4 rejections pass, 242 `xfail_expected`, 0 in every
  loud category (`verifier_report.json`). With SUPPORTED temporarily set to TARGET: 260 / 261 (all 256 cells), the
  evidence for Refinement 1.
- **Issues encountered** (verifier):
  - **Fixed — rows wider than one NoC packet.** Quasar's overlay packetizes at 8 KiB.
    Reachable inside Phase 0 SUPPORTED (bf16
    rank 4, `W_o > 4096`, >= 4 rows/core). The writer now drains `dfb_out_rows` with explicit sync when
    `entry_bytes > NOC_PACKET_BYTES` (host CT flag `explicit_out_rows`; plans with R > W refused there); implicit
    otherwise, unchanged.
  - Removed the unbound placeholder `kernels/pad_compute.cpp`.
  - `max_core_data_rows` robust to an empty data split.
- **Tests added**: `test_pad_plan.py` (implementer: coverage at value 1.0 under four plans, rows_streamed regime,
  pad barriers, device session, rejections; verifier: wide rows on both sides of the packet boundary, R6
  sharded-input refusal), `test_pad_precision_baseline.py` (exactness + R4 output placement).

### Implementer notes (Phase 0)


Built as op_design.md specifies: one `ttnn.ProgramSpec`, one `ttnn.generic_op` dispatch, a reader and a writer
DM kernel, no compute kernel (`kernels/pad_compute.cpp` is a documented, unbound placeholder).

- Work distribution: data rows (`D = N*C*H`) and pad rows (`P = O - D`) split as two lists over the full grid
  (`split_work_to_cores(row_wise=True)`); `pad_start_out_row` per core from the O(rank) host count
  (checked against brute force on 3,000 random geometries).
- Reader: explicit sync on `dfb_out_rows` and `dfb_pad_row`, every transfer `trid 0`; one read per input row
  landing at `bw*e` in its entry; head/tail margins filled with `fill_l1_range` on the first lap of each tile
  counter only, in the shadow of the reads; one barrier + one push per block.
- Writer: pad rows from the template, one write barrier per `pad_rows_per_barrier` (default: the core's
  whole pad assignment); data rows by implicit-sync writes; `noc_async_write_set_trid(0)` last.
- Knobs (single source, `pad_program_artifacts.py`): `PAD_THREADS = (1, 1)`, `BUFFER_DEPTH`, `block_rows` /
  `ring_blocks` from `choose_blocking` (rows_resident / rows_streamed), `pad_rows_per_barrier`.

Deviations from the design (advisory parts), and one finding:
- **Thread dealing under a plan with R != W.** A STRIDED ring's thread rotates its tile counters per *op*
  (reserve/push), and an op's entries sit `stride_size` apart (`dataflow_buffer.inl`). The design dealt rows
  `e ≡ w (mod W)` to writer threads, which hangs for W > R with block pushes (found with plan (1,0,2): only
  cores with 2 data rows hung). Fixed: with W = m*R a writer takes whole reader ops (`k ≡ w/R mod m` of reader
  `w mod R`); with R > W it takes rows `w + i*W`. Entry addresses use `get_stride_size()`. The resident block is
  `ceil(max_core_data_rows / max(R, W))` so each (reader, writer) counter gets one op. Template copies: one per
  writer thread, copy k pushed by reader `k mod R`.
- No regression at the Phase 0 plan (1,0,1): stride = entry size, one tile counter, behaviour unchanged.

Tests: acceptance 4/4; golden suite 18 passed (14 PHASE0 cells + 4 rejections), 242 strict xfail, 0 failed;
`test_pad_plan.py` 41 cases: value-1.0 coverage under plans (1,0,1) (1,0,2) (2,0,1) (2,0,2), rows_streamed
(block_rows 1 and 3, ring wraps, ragged last block) under three plans, several pad barriers per core, one
device session with tt-metal's tilize/untilize before and after pad, rejections X1-X4, refusals outside PHASE0.

Structural counts (simulator: no timing claim): input crosses DRAM once (one read per input row), output
once (one write per output row), 0 B cross-core, one reserve/barrier/push per core in rows_resident on every
INPUTS entry, one pad-write barrier per core, all 32 clusters engaged whenever D or P >= 32.

## Refinement 0 — Tile layout and rank 2 (Qwen)
- Date: 2026-11-21
- What was done: SUPPORTED["layout"] += TILE, SUPPORTED["rank"] += 2 (at bf16, DRAM, value 0, after-padding).
  - Reused: the whole program. Same `ProgramSpec`, same two rings, same split of data pages and pad pages over
    the full grid, same thread plan and block knobs (`choose_blocking`, BUFFER_DEPTH, `pad_rows_per_barrier`),
    same template mechanism for pad pages. Rank 2 was already folded to rank 4 on the host, so for row-major
    it is a SUPPORTED widening only.
  - Added:
    - `page_geometry` (host): both layouts become a 4-dim page grid. In tile layout a page is a tile, and with
      after-padding an input tile maps whole to the output tile at the same (n, c, ht, wt).
    - Writer: the page walk gains an inner dim `T` / `T_o` / `bt`. Row-major passes 1 / 1 / 0, so its behaviour
      is unchanged, including the whole-run skip in the pad walk.
    - Reader: a `fill_tile_padding_of_block` step after the read barrier. In tiles of the last tile row / column
      it writes `value` over the input's tile padding, face by face, via `fill_l1_range`. Compiled in only under
      the `tile_layout` CT flag.
    - `_run` allocates the output in the input's layout.
- Accuracy achieved: exact (0 differing elements; PCC 1.0, relative RMS 0) on every tile and rank-2 shape in
  INPUTS. That includes inputs whose tile padding holds 7.0 (it does not leak) and value 1.0 through the program
  entry.
- Golden test progress:
  - The 28 newly supported cells pass (14 rank-4 TILE, 7 rank-2 TILE, 7 rank-2 ROW_MAJOR), plus 3 prior-passing
    representatives (targeted slice; the harness's full run is the record).
  - With the 14 Phase 0 cells, 42 SUPPORTED cells in all.
  - Unit dir: 110 passed (prior suites unchanged, plus the new file).
- Issues encountered: None on the supported surface. Finding for Refinement 1: in tile layout, an H or W
  before-amount is an element shift across tile boundaries, not a page copy. The program refuses it with
  `NotImplementedError`, so "SUPPORTED = TARGET with zero kernel change" no longer holds for
  `layout=TILE, padding_side=before` cells with an H/W before-amount. N/C before-amounts in tile layout are a
  page shift and map through the same walk.
- Tests added: `tests/ttnn/unit_tests/operations/pad/test_pad_tile.py`:
  - the Qwen model call, checking every added element, on an input with non-zero tile padding;
  - tile coverage on 6 shapes × plans (1,0,1) (1,0,2) (2,0,2) × value 0 / 1;
  - the tile streamed regime (`block_rows` 1 and 3, the ring wraps, the fill runs every lap);
  - rank-2 row-major at value 0 / 1;
  - a tile session between tt-metal programs (`to_layout` before and after);
  - refusal of tile before-padding in H.
