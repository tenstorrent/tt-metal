# Changelog: qkv_proj_heads_rope

## Phase 0 - Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1: 2D token_rows x head_cols output-stationary grid (108/110 cores on p100a), unicast reads, matmul with the bias folded as the 25th K tile into Float32 L1-accumulated partials, rotary as two DEST-accumulated FPU multiplies per output tile, one `generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED], buffer_type=[L1], out_buffer_type=[L1], table_buffer_type=[L1]
- **Accuracy achieved**: PCC >= 0.9999984, max_abs_err <= 0.0212, rel_rms <= 1.91e-3 (gate 4e-3), ULP p99 <= 4.9 (measured on 4 shapes via test_qkv_proj_heads_rope_precision_baseline.py); q/k got/true median 0.99925 (truncation bias of the rotary FPU multiply, inside tolerance), v 1.0000
- **Golden suite at Phase 0**: 25 supported_pass / 175 xfail_expected / 0 loud of 210 (per `verifier_results/verifier_report.json`, run with a local, reverted patch for two golden-harness defects; see verification_report.md)
- **Perf at Phase 0**: 8x512 279-281 us (stock 226.3, target 90); every flagged row slower than stock
- **Issues encountered**: verifier DRY fix: the reader derived the table row size and page index from literals; `TILES_PER_HEAD` is now a reader CT arg. Golden harness defects (`packer_l1_acc` kwarg, tuple tolerance) reported, not edited.
- **Tests added**: test_qkv_proj_heads_rope.py, test_qkv_proj_heads_rope_perf.py, test_qkv_proj_heads_rope_precision_baseline.py

## Refinement 1 - x multicast within each M-group (regime R3) + `buffer_type = DRAM`
- **Date**: 2026-11-02
- **What was done**: Built regime R3. Logical cores are reordered to `l = m_group*grid_n + n_group`, so each M-group is a consecutive run that `ttnn.Mcast` can partition. One `Mcast` channel on the reader (NOC_0, handshake on, fixed sender with `Staggered` placement) delivers each x K-block: every core reserves its `cb_activation_block` slot, the group's sender reads the x tiles plus the ones column into it and multicasts the leading `rows*k_block` tiles to the same slot on the other `grid_n - 1` cores. Removed the per-core x read stagger (one reader per K-block now). Compute kernel, writer, CB contracts and selection function are unchanged. Added `ttnn.BufferType.DRAM` to `SUPPORTED["buffer_type"]`.
- **Deviation**: the verifier asked for a rotating sender. `McastRotatingSenderConfig` requires each receiver group to be a rectangle (`mcast.cpp:89`), and `grid_n` consecutive cores on the 11-wide grid usually are not, so the sender is fixed per group and staggered across groups. Because the kernel serializes rounds (read, multicast, next round), rotation would not overlap reads anyway.
- **Perf (device us, x in L1)**: 8x512 279 -> 265, 16x256 278 -> 267, 32x128 275 -> 270, 8x384 252 -> 252, 4x512 215 -> 145, 8x256 213 -> 147, 2x512 206 -> 93, 4x256 205 -> 97. x in DRAM: 8x512 272 (+2.5% vs L1, was 317), 4x512 147, 2x512 92 (was 221).
- **Bottleneck now**: at 8x512, stubbing W reads gives 134 us and stubbing the x path gives 277 us. The W unicast re-read (12 DRAM crossings) binds, which is Refinement 3 (R2). The M-group-consecutive order means R2 needs per-N-group `Mcast` channels or a rectangular-grid constraint (see `op_design.md`).
- **Accuracy achieved**: unchanged data path; unit suite gates PCC >= 0.9999, rel RMS <= 4e-3 per output on 1x32, 1x128, 2x64, 3x96, 4x256, 8x256, 32x128, 8x512 (x in L1) and 1x32, 2x512, 4x256, 8x384, 8x512 (x in DRAM).
- **Golden test progress**: 42/42 tile_aligned cells with out/table in L1 pass (17 of them with x in DRAM, `test_op` + `test_op_loose`). `test_regression.py`: 6 passed, 4 failed (all `h_non_aligned`, Refinement 2 scope, failing before this change).
- **Issues encountered**: the `-k` filter in `run_safe_pytest.sh` cannot take parentheses or `=`, so golden slices were run by explicit node id.
- **Tests added**: `test_qkv_proj_heads_rope.py::test_x_in_dram` (5 shapes, one per grid class), `test_qkv_proj_heads_rope_perf.py::test_qkv_proj_heads_rope_perf_x_dram` (8x512, 4x512, 2x512).

## Refinement 2 - Non-aligned S (`h_non_aligned`) + DRAM outputs and tables
- Date: 2026-11-02
- What was done: SUPPORTED widened: alignment += h_non_aligned, out_buffer_type += DRAM, table_buffer_type += DRAM. Writer gets CT arg `last_valid_rows = S % 32`; on the last tile-row of each image it zeroes rows `[S % 32, 32)` of every output tile in cb_out_tiles via `noc.async_write_zeros` (partial rows of faces 2*(r/16)+{0,1}, whole lower faces when r < 16) before the NoC write. `if constexpr` removes it for aligned S. DRAM placements needed no kernel change (TensorAccessor). Reused: existing per-row output path, CB, zeroing primitive used for the bias tile. Added: one CT arg and the pad-zero block.
- Accuracy achieved: unit + golden tolerance (PCC >= 0.9999, rel RMS <= 4e-3) on [2x37, 16x8, 13x300, 7x511, 4096x1 (multi-block), 2x512, 8x512, 3x100]; pad rows exactly 0 with inf in x and table padding.
- Golden test progress: 210/210 passing (test_golden.py 200 + test_regression.py 10).
- Issues encountered: `async_write_zeros` on a CircularBuffer addresses from the write pointer; the pad-zero offsets are taken relative to it from the read pointer. 4096x1 cannot be L1-resident (tile padding makes x ~200 MB), so the unit case places x and outputs in DRAM.
- Perf: 8x512 aligned all-L1 268 us (post-R1 265 us; noise band).
- Tests added: `test_non_aligned_padding_zero` (5 shapes, inf padding, pad rows == 0), `test_dram_outputs_and_tables` (3 cases) in test_qkv_proj_heads_rope.py.
