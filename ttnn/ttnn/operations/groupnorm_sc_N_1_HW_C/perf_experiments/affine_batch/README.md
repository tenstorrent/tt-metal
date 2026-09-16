# affine_batch — idea E1: batch the pass-2 affine build per column group

Isolated bake-off (perf-lab style, single core, sharded L1, pure compute) of the op's pass-2 `build_affine_block`:
`[mean;rstd]_full x E_T -> a_T = rstd_T*gamma_T ; b_T = beta_T - mean_T*a_T` for the `cols` channel tiles of one
column group. Baseline = the op's current per-tile sequence (4 helper calls per tile), reproduced verbatim from
`kernels/groupnorm_sc_N_1_HW_C_compute.cpp` (zone `c_affine`).

Files: `affine_batch_bench.py` (inline compute kernel, 5 methods via CT arg + `ttnn.ProgramDescriptor`),
`test_affine_batch.py` (correctness gate vs torch and vs the baseline variant; in-process device profiler),
`report_perf_sweep.md` (raw sweep table written by the test).

Run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/affine_batch/test_affine_batch.py`
(env `AB_COLS`, `AB_KG`, `AB_ITERS`, `AB_REPORT`).

Precision contract, identical for every variant (never tuned): `fp32_dest_acc_en=True`, `HiFi4`, `math_approx_mode=False`,
`dst_full_sync_en=True`; stats / E pages Float32, gamma / beta bf16 row-0 tiles, a / b Float32 full tiles.

## Variants

| method | what | helper calls per column group |
|---|---|---|
| `baseline` | op today: per tile T `matmul_block(2x Kg x 1)` -> a chain -> `unary_bcast(beta)` -> b chain (FPU Mul + DestReuse Sub) | 4 x cols |
| `batched` | ONE `matmul_block` with N = cols (`MatmulBlockShape::of(2, 1, 1, cols, Kg, 1)`, SubblockMajor: tiles 0..cols-1 = mean_T, cols..2cols-1 = rstd_T), ONE a chain over `tiles(cols)`, ONE `unary_bcast` over cols beta rows, ONE b chain over `tiles(cols)` | 4 |
| `batched_fold` | `batched` minus the `cb_beta_full` round trip: b chain = FPU Mul -> `UnaryBcast<Row>(beta) -> D1` -> SFPU `SubBinary<D1,D0,D0>` | 3 |
| `fused` | one matmul + ONE chain for a and b (a stays in DEST; `DestReuseBinary<Mul>` for mean*a, SFPU sub) | 2 |
| `fused_sfpu` | as `fused` but product + sub on the SFPU (`CopyTile(mean)`, `MulBinary`, `SubBinary`) | 2 |

## Measured (box=bh-qb-11-special-mstaletovic-for-reservation-93463, arch=BLACKHOLE, 1 core, DEVICE KERNEL DURATION [ns], one fresh run per cell)

`per-group ns` = slope between an iters=1 and an iters=11 launch of the same kernel (launch-independent cost of one
column group's affine build). Full table incl. iters=1 latencies: `report_perf_sweep.md`.

Focus config of the tournament shape (1,1,1024,640) G=32 on the 11x10 grid: **cols = 2, Kg = 1**

| variant | per-group ns | per-tile ns | speedup | a vs baseline | b vs baseline | b max abs err vs torch fp32 |
|---|---:|---:|---:|---|---|---:|
| baseline | 1964 | 982 | 1.00x | — | — | 6.8e-3 |
| **batched** | **1574** | **787** | **1.25x** | bit-identical | bit-identical | 6.8e-3 |
| batched_fold | 1633 | 817 | 1.20x | bit-identical | 1.6e-3 | 5.5e-3 (more precise) |
| fused | 1751 | 875 | 1.12x | bit-identical | 6.2e-3 | 1.29e-2 (LESS precise) |
| fused_sfpu | 1762 | 881 | 1.11x | bit-identical | 4.5e-3 | 3.3e-3 (more precise) |

Domain sweep, per-group ns (speedup vs baseline), `batched` / `batched_fold`:

| cols | Kg=1 baseline | Kg=1 batched | Kg=1 batched_fold | Kg=2 baseline | Kg=2 batched | Kg=2 batched_fold |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1025 | 1002 (1.02x, flat) | 1021 (1.00x) | 1146 | 1140 (1.01x, flat) | 1141 (1.00x) |
| 2 | 1964 | 1574 (1.25x) | 1633 (1.20x) | 2200 | 1792 (1.23x) | 1845 (1.19x) |
| 4 | 3736 | 2651 (1.41x) | 2755 (1.36x) | 4197 | 3047 (1.38x) | 3154 (1.33x) |
| 5 | 4630 | 3158 (1.47x) | 3324 (1.39x) | 5192 | 3655 (1.42x) | 3823 (1.36x) |
| 8 | 7291 | 4744 (1.54x) | 4998 (1.46x) | 8226 | 5538 (1.49x) | 5786 (1.42x) |

`fused` / `fused_sfpu`: 0.98x at cols=1, 1.11-1.25x elsewhere — always slower than `batched` (an SFPU op on a full
fp32 tile costs more than the L1 round trip it replaces once the per-call overhead is already amortised).

Numerics (all 10 (cols, Kg) cases): `batched` a and b bit-identical to baseline. `batched_fold` a identical, b differs
by <= 2.8e-3 abs (|b| up to ~8) and is closer to the fp32 torch reference than the baseline in every case.
`fused` b is 2x further from torch than the baseline (the DEST->srcB re-route of a rounds harder) — a precision cost.
The baseline's own distance from torch (a 3.4e-3, b 6.8e-3 abs) is the op's documented tf32-class FPU operand precision.

## Domain

Applies everywhere: correct on every regime run (cols 1,2,4,5,8 x Kg 1,2 with straddling groups and padded lanes),
flat at cols = 1 (the flagship's Ct_core = 1 regime; the batched calls degenerate to the per-tile ones), wins from
cols = 2 up. No `incorrect` / `inexpressible` / `measured-regression` exceptions. Untested (same helper calls, shape
`tiles(cols)` instead of `one_tile()`): the `has_gamma`/`has_beta` = false branches, ragged `valid_cols < cols`.

## Integration notes (real op)

* Compute: replace the `for (tl...)` loop of zone `c_affine` by the `batched` sequence in `affine_batch_bench.py`
  (`method == 1`) with `cols -> valid_cols` at runtime; keep the pad-pop / pad-push of the ragged group.
  Ordering hazard: `cb_wait_front(cb_a_full, valid_cols)` between the a chain and the b chain (CB credit is the
  only pack->unpack ordering). Pop `cb_stats_T` by `2 * valid_cols`.
* CB sizes (`groupnorm_sc_N_1_HW_C_program_descriptor.py`, `l1_ledger.md`): `CB_STATS_T` 2 -> `2 * cols` pages,
  `CB_BETA_FULL` 1 -> `cols` pages (Float32; +cols*12 KB, 24 KB on the focus shape, 84 KB at cols = 8). Or take
  `batched_fold` (no `CB_BETA_FULL` at all, 60 ns/group slower, slightly more precise).
* Reader pass 2 (`write_membership_lanes`, transposed=false): E tile order `kg * valid_cols + tl` (the matmul helper
  indexes in1 as `k * N + n`). Identical to today's `tl * Kg + kg` whenever Kg == 1 (every G <= 32 shape); Kg = 2
  measured correct with the new order. Pass 1 (`transposed=true`, E^T as in1 with N = Kg) is unchanged.
* DEST: out subblock 1 x cols <= 8 fp32 tiles — guaranteed by the host `cols = _balanced_block(Ct_core_max, dest_limit)`.
* No raw LLK anywhere; helper gap (capability): `DestReuseBinary` has no `BroadcastDim` and the LLK
  `binary_dest_reuse_tiles` path is `BroadcastType::NONE` only, so an FPU `beta_row(Row) - DEST` is inexpressible —
  that is why the beta broadcast either round-trips L1 (`batched`) or the subtraction moves to the SFPU (`batched_fold`).
