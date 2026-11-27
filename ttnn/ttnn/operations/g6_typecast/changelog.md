# Changelog: g6_typecast

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  R1 `tile_stream`: interleaved TILE in → eltwise-chain typecast → interleaved TILE out, contiguous tile ranges
  over the full compute grid, one CB handshake + one reader barrier per block.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], output_dtype=[float32, bfloat8_b, int32], layout=[TILE],
  buffer=[dram, l1], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3, 4]
- **Accuracy achieved**: float32 / int32 destinations bit-exact (PCC=1.0, max_abs_err=0, 0 ULP); bfloat8_b
  PCC=0.999975, max_abs_err=0.03125, rel_rms_err=0.0076, ≤ 1 quantization step vs host rendering (measured on
  6 production shapes via test_g6_typecast_precision_baseline.py)
- **Golden suite at Phase 0**: 42 supported_pass / 27 xfail_expected / 0 supported_fail / 0 xpass_drift /
  0 xfail_wrong_mode (44/71 incl. 2 untagged regression tests; per `verifier_report.json`)
- **Issues encountered**: verifier fix — program-cache key now includes the block knobs (`BUFFER_DEPTH`,
  `BLOCK_TILES_CAP`, `L1_OP_BUDGET_BYTES`, `L1_SLACK_BYTES`) so a turned knob never replays a stale program (BB-07).
- **Tests added**: test_g6_typecast.py, test_g6_typecast_blocking.py, test_g6_typecast_precision_baseline.py

## Refinement 1 — ROW_MAJOR regime (R2 `rm_pseudo_tile_stream`) with its conversion arms
- Date: 2026-11-27
- What was done: Regime R2 built natively (no to_layout / tilize wrapper; one program). Selected by
  `input_tensor.layout` and reported in `last_plan["regime"]`. Contiguous row ranges over the full grid
  (`split_work_to_cores(row_wise=True)`); per block of `rows_per_block` rows the reader issues ONE NoC read
  per bank run (rows r, r+nb, ... are adjacent in a bank at the aligned-page stride; nb =
  `TensorAccessor::contiguous_page_stride()`) into a 64 B-aligned staging CB, one barrier, then compacts the
  rows densely into `cb_input_tiles` as 1024-value pseudo-tiles; compute runs the unchanged R1 chain; the
  writer expands dense rows back into bank-run order and issues ONE write per bank run (flush before
  staging reuse). `block_tiles` stays the single block knob (`plan_rm_blocks` derives rows_per_block from
  it, `BLOCK_TILES_CAP` caps it). New arms: bf16→uint8 (SFPU truncation), uint32→int32 and int32→uint32
  (unpack-to-DEST, no SFPU, all 32 bits). 32-bit sources tag `cb_input_tiles` `UnpackToDestFp32`;
  `fp32_dest_acc_en` also for uint8 destinations and 32-bit sources (Phase 0 pairs unchanged).
  SUPPORTED: dtype += uint32, int32; output_dtype += uint8, uint32; layout += ROW_MAJOR; rank += 1, 2.
  `validate()` refuses pairs with no arm (`CONVERSION_ARMS`) and bf8b + ROW_MAJOR with ValueError;
  EXCLUSIONS stays empty.
  Reused: compute kernel (unchanged), reader/writer files (rm_mode `if constexpr` branch), split/plan/
  cache. Added: `_create_rm_program_descriptor`, `plan_rm_blocks`, 2 staging CBs, `_compute_config`.
- Accuracy achieved: bit-exact (0 differing values) on [1,3844], [1,31], [1], [3,77,33], [2,50,3] (L1),
  [1,1024,1024,3], [1,2048,2048,3], TILE [1,1,32,128] u32→i32 and [1,1,64,96] i32→u32.
- Golden test progress: targeted slice (all 14 ROW_MAJOR cells + loose F322/F317 + Q2230 + TILE
  representatives Q2185/P39/Q2229): 23 passed, 1 xfailed (Q2228, int32→bf16 = Refinement 2). Unit dir:
  test_g6_typecast.py + test_g6_typecast_blocking.py 30/30.
- Perf baseline (device kernel ns, one fresh run): F322 2 330 848 ns (stock 14 563 310; target 110 590),
  F317 952 224 ns; BRISC (writer) is the longest RISC (F322: writer 2.33 ms, reader 1.71 ms) — the
  per-row RISC-V expansion/compaction (byte copies of 3 B / 6 B rows) binds, not the NoC. Refinement 3
  levers: wider copy units / fewer per-row ops, overlap staging reuse (double staging), bank-local core
  assignment. TILE shapes unchanged (Q2185 36.1 µs, P39 74.7 µs).
- Issues encountered: None (first run green).
- Tests added: tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_rm.py (regime pins for TILE and
  ROW_MAJOR, new arms, forced multi-block / ragged RM via BLOCK_TILES_CAP, L1 RM, host refusals).
