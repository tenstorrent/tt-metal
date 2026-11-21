# pad changelog

## Phase 0 (ttnn-implementer, 2026-11-21) — quasar on ttsim

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
