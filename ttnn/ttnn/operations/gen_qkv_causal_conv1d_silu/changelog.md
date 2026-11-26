# Changelog: gen_qkv_causal_conv1d_silu

## Phase 0 — Core Implementation
- **Date**: 2026-11-26
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op` program per call, with two regimes: ROW_MAJOR sticks read directly, or TILE untilized in-kernel.
  - Windowed tilize does the token shift as exact data movement. Four row-broadcast MACs accumulate in one DEST slot, SiLU runs on the packer thread, and there is one pack.
  - The writer routes each tile to q/k/v.
  - Work is split across channel groups × token ranges, using the full device grid.
  - Taps are multicast once per channel group through the kernel_lib mcast helper.
- **SUPPORTED at Phase 0**:
  - dtype=[bfloat16], layout=[ROW_MAJOR, TILE]
  - in/history/taps buffer=[dram, l1], output_memory=[default, dram_interleaved, l1_interleaved]
  - fp32_dest_acc_en=[False, True], history_layout=[token_paged], input_window=[exact]
  - width_split / channel_count / seq_tiles / chunk_blocks = every value; EXCLUSIONS=[]
- **Accuracy achieved** (HiFi4, `test_gen_qkv_causal_conv1d_silu_precision_baseline.py`, 4 shapes T 32…4096 + the fp32 cancelling-taps cell):
  - 16-bit DEST: PCC=0.999987, max_abs_err≤0.181, mean_abs_err≈0.0033, rel_rms_err≈0.0056, ULP p99 17–18
  - fp32 DEST: PCC=0.999998, max_abs_err=0.0078, rel_rms_err=0.0018, ULP p99 0 (max 1); FP32_DEST_BOUND holds
- **Golden suite at Phase 0**: 17 / 21 passing, 4 xfail_expected (012, 013, 018, 019: channel_paged / column_window), 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`).
- **Perf (flagged targets, BH 11×10)**: 001 8.7 µs, 002 10.4 µs, 003 13.2 µs, 004 128–131 µs. All are under native (30.5 / 43.1 / 48.1 / 346.9) and under target (17.0 / 18.5 / 22.5 / 164); 004 stretch is 117.
- **Issues encountered** (implementer, recorded in `l1_ledger.md` and breadcrumbs):
  - The tap fetch was DRAM-request-bound, so tap multicast was built (004: 185.9 → 152.9 µs).
  - SiLU was moved to the packer thread (197.7 → 185.9 µs).
  - Default `block_rows` was capped at 1 for overlap (153 → 129 µs).
- **Verifier fixes**:
  - Fixed the acceptance-test harness bug in `test_repeat_is_bit_identical` (`case.values[0]` on a dict → `TypeError`).
  - CB capacities now have one source, `cb_layout()` (it previously had to be kept in sync by hand between `footprint()` and the descriptor).
  - Added `page_channels` to the program-cache key (latent BB-07 hazard).
  - `HISTORY_ROWS` is now imported from `HALO_ROWS`.
  - Added a pinned TILE multi-row-block test.
  - Added an as-built note to `op_design.md` and a single-source note to `l1_ledger.md`.
- **Tests added**: `test_gen_qkv_causal_conv1d_silu.py` (acceptance, planner), `test_gen_qkv_causal_conv1d_silu_debug.py` (BB-07 cache-hit correctness + pinned TILE multi-row block), `test_perf_knobs.py` (implementer knob sweep), `test_gen_qkv_causal_conv1d_silu_precision_baseline.py` (verifier).

## Refinement 1 — Reader addressing generality: input_window column_window + history_layout channel_paged
- **Date**: 2026-11-26
- **What was done** (reader-only knob-turns; no CB, block, schedule or L1 change):
  - **column_window**: the existing `input_col_offset_tiles` addressing already covered both regimes (RM byte offset `2·(offset + c0)` in the token-row page; TILE page `r·W_t + offset/32 + c`, including the t0 > 0 halo rows / extra untilized tile-row). Opened `SUPPORTED["input_window"]`. The TILE page stride `W_t` is now `ceil(W/32)` (the padded width), so a W that is not a multiple of 32 is **supported** natively in both layouts rather than refused (RM: the read offset stays 64 B aligned inside the 2W-byte page).
  - **channel_paged**: new reader helper `read_history_halo` (reader CT arg `history_page_channels`, 0 = token-paged; token-paged path unchanged via `if constexpr`). History `[1, C/P, 3P]`: row r of channel c is page `c / P`, element `r·P + c mod P`; a group whose channel slice straddles pages issues one read per contiguous run per row (every run 32-channel / 64 B aligned). Only the `t0 = 0` core of each group reads history.
  - `page_channels` is threaded op → `create_program_descriptor` (it was already in the program-cache key).
  - Reused: the reader's halo/body addressing, the plan, every CB, the cache key. Added: `read_history_halo`, one CT arg, the padded-width stride.
- **Accuracy achieved** (golden operands vs float64 reference, q / k / v):
  - conv1d-012 (RM, P 160, 16-bit DEST HiFi4): PCC=0.999987, rel RMS=0.0056
  - conv1d-013 (RM, offset 32, HiFi2): PCC=0.999983, rel RMS=0.0061
  - conv1d-018 (TILE, L1 in/history, P 32, offset 64, fp32 DEST HiFi2): PCC=0.999992, rel RMS=0.0058
  - conv1d-019 (TILE, L1 in, P 32, W 448 > C 384, fp32 DEST HiFi4): PCC=0.999998, rel RMS=0.0018, ULP p99 0; FP32_DEST_BOUND holds
- **Golden test progress**: 21 / 21 expected (012, 013, 018, 019 move from xfail_expected to passing; confirmed on a `-k` slice of those four + 001, 002, 005).
- **Perf (hot-path no-regression, BH 11×10, device kernel ns, same session, before → after)**: 001 8.39 → 8.48 µs, 002 9.67 → 9.62, 003 12.86 → 12.92, 004 128.97 → 127.73, 020 74.49 → 73.30, 005 6.88 → 6.89 — all within noise.
- **Issues encountered**: `--profile` runs end with "tracy-capture exited with code 3" on this box (no `ops_perf_results*.csv`); the device profiler's own `generated/profiler/.logs/cpp_device_perf_report.csv` still has one row per program and was used instead.
- **Tests added**: `test_reader_addressing.py` — the four golden cells; pinned plans covering every halo source under a column window + channel-paged history (RM later-block L1 halo, RM DRAM halo at t0 > 0 with a page-straddling group, TILE later-block, TILE extra-untilized-row at t0 > 0, TILE multi-row block); W = 2200 (not a multiple of 32) in both layouts; C = 3P (token- and channel-paged histories with the same shape, back to back).
