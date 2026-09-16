# affine_lane_form — Perf round 2: lane-form pass-2 affine build + broadcast-row apply

Isolated bake-off (perf-lab style, single core, sharded L1, pure compute) of the op's whole post-finalize region for
one image / column group: `c_stats_bcast + c_affine + c_apply`. Baseline = the op today (`batched`, Perf 1),
reproduced verbatim from `kernels/groupnorm_sc_N_1_HW_C_compute.cpp`.

Files: `affine_lane_form_bench.py` (inline compute kernel, 6 methods via CT arg + `ttnn.ProgramDescriptor`),
`test_affine_lane_form.py` (correctness gate vs torch fp32 and vs the baseline variant; in-process device profiler),
`report_perf_sweep.md` (raw sweep table written by the test, 21 cells x 6 variants).

Run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/affine_lane_form/test_affine_lane_form.py`
(env `ALF_ITERS`, `ALF_REPORT`, `ALF_VARIANTS`, `ALF_SWEEP=focus|gb|affine|apply|all`, `ALF_AFFINE_KG`, `ALF_APPLY`).
Do NOT run with `--dev`: watcher + device profiler together overflow the BRISC firmware region (not a kernel issue).

Precision contract, identical for every variant (never tuned): `fp32_dest_acc_en=True`, `HiFi4`, `math_approx_mode=False`,
`dst_full_sync_en=True`; stats / E / stats_T / a / b pages Float32, x / y / gamma / beta bf16.

## Variants

| method | build | apply (y = x*a + b per tile) | raw LLK |
|---|---|---|---|
| `batched` (op today) | `unary_bcast<Row>` stats -> full; matmul on full tiles; a chain (gamma Row-bcast); `unary_bcast<Row>` beta -> full; b chain (DestReuse Sub) | chain: `BinaryFpu Mul(x, a_full)` + `DestReuseBinary Add(b_full, DEST_TO_SRCB)` + Pack, block 1 | no |
| `batched_blk` | as above | same chain with `.block_size(cols)` | no |
| `lane_fullab` | lane: matmul straight on the lane-form stats; a_row = rstd_T_row * gamma_row (plain eltwise); b_row = beta_row - mean_T_row * a_row (DestReuse Sub on the row tile) | `unary_bcast<Row>` a_row / b_row -> full, then the `batched` chain | no |
| `lane_d2a` | lane | per tile-row: `mul_tiles_bcast_rows(x, a_row)` into DEST, then dest-reuse `ELWADD<ROW, DEST_TO_SRCA>` of `bcast_row(b_row)` | yes |
| `lane_accadd` | lane | per tile-row: `mul_tiles_bcast_rows(x, a_row)` into DEST, then standard `ELWADD<ROW>` with `acc_to_dest = 1`: DEST += 0 (zero srcA tile) + `bcast_row(b_row)` | yes |
| `lane_l1` | lane | chain `x * bcast_row(a_row)` -> fp32 `cb_interm`; chain `interm + bcast_row(b_row)` -> y | no |

The lane form needs NO property of rows 1..31 of the totals: every consumer reads row 0 only (the expansion matmul's
row 0 depends on in0's row 0 only; ROW-broadcast unpacks and `unary_bcast<Row>` read one row). The test's
`dirty_rows=True` mode fills rows 1..31 of the stats with random finite junk and every variant still passes.

## Result (box=bh-qb-11-special-mstaletovic-for-reservation-93463, BLACKHOLE, 1 core, DEVICE KERNEL DURATION [ns])

Focus geometry cols=2, Kg=1, chunk_rows=2, 2 chunks, second chunk ragged to 1 row (Ht_core=3), gamma+beta.
`per-region ns` = slope between iters=1 and iters=6 launches (launch-independent cost of the region).

| variant | per-region ns | speedup | y vs baseline | max / mean err vs torch fp32 |
|---|---:|---:|---|---:|
| batched | 2986 | 1.00x | — | 8.2e-2 / 3.58e-3 |
| batched_blk | 3006 | 0.99x | bit-identical | same |
| lane_fullab | 3071 | 0.97x | bit-identical | same |
| **lane_d2a** | **2456** | **1.22x** | **bit-identical** | same |
| lane_accadd | 2294 | 1.30x | 1 bf16 output ulp on some elements (6.2e-2) | 8.2e-2 / 3.53e-3 (more precise) |
| lane_l1 | 2529 | 1.18x | bit-identical | same |

Sweep (all correct; `report_perf_sweep.md`): affine cols {1,2,4,5,8} x Kg {1,2}: lane_d2a 1.20-1.26x, lane_accadd
1.27-1.35x, lane_l1 1.11-1.23x, lane_fullab 0.94-1.07x. apply chunk_rows x cols {1x1,2x2,4x4,8x1,1x8,4x8}: lane_d2a
1.09-1.30x, lane_accadd 1.24-1.46x, lane_l1 1.07-1.18x, lane_fullab 0.93-1.01x. gamma/beta (1,0) and (0,0): lane_d2a
1.14-1.15x, lane_accadd 1.23x, lane_l1 1.12x, lane_fullab 0.92-0.93x (measured regression: it always pays 2*cols
broadcasts where the baseline pays only 2*Kg without beta).

Mechanism: the baseline apply unpacks a full 4 KB fp32 a_full and b_full tile per 2 KB x tile (10 KB/tile through the
unpackers); the lane apply unpacks one row of each (~2.25 KB/tile). Blocking alone (`batched_blk`) is flat, so the win
is the deleted broadcasts + the row-form operands, not the MOP init cadence. lane_accadd additionally drops the
per-face DEST->src move of the dest-reuse path.

## Integration recipe (real op, `kernels/groupnorm_sc_N_1_HW_C_compute.cpp`)

1. Delete zone `c_stats_bcast`; matmul in0 = `CircularBuffer stats_row_buf(cb_stats_row)` (`WaitAndRetainOnLastBlock`
   as today); at image end `cb_pop_front(cb_stats_row, num_stats)` replaces the `cb_stats_g_full` pop.
2. a chain: `input(cb_gamma_row, BroadcastDim::None, WaitPolicy::PerTile, PopPolicy::PerTile)` (was `Row`). Output
   `cb_a_full` becomes row-form (same Float32 pages, same count) — rename `cb_a_row`.
3. b chain: drop the `unary_bcast<Row>(beta)` + `cb_beta_full`; `DestReuseBinary<Sub, input(cb_beta_row, PerTile, PerTile),
   DEST_TO_SRCB, D0>` directly. `has_gamma=false` (CopyTile) / `has_beta=false` (Negative) branches unchanged.
4. apply (`apply_chunk`): replace the chain by the method-3 (`lane_d2a`, bit-identical) or method-4 (`lane_accadd`)
   raw loop of `affine_lane_form_bench.py`, with `cols -> valid_cols`, x index `x_base + r * valid_cols + c`
   (the dense ragged layout the chain's Block mapping uses today), `pack_reconfig_data_format(cb_out)` once per chunk,
   `cb_reserve_back/pack_tile/cb_push_back` per tile-row, then the existing `pad_push`. Method 4 needs a 1-page bf16
   zero CB (reader zero-fills one page next to the pass-2 constants).
5. CBs (`groupnorm_sc_N_1_HW_C_program_descriptor.py`, `l1_ledger.md`): delete `CB_STATS_G_FULL` (2*Kg fp32 pages) and
   `CB_BETA_FULL` (cols fp32 pages); `CB_STATS_T`, `CB_A_FULL`, `CB_B_FULL` unchanged in size/format (row-form
   content); +1 bf16 page for method 4. Reader: no change (E already k-major, gamma/beta rows unchanged).
6. Kernel-head justification comment for the raw LLK: see the header of `_KERNEL` in `affine_lane_form_bench.py`.
