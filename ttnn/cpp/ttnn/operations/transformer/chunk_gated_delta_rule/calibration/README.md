# Calibrating the fused GDN geometry model

The fused `chunk_gated_delta_rule` program picks its core split (receivers and producers per head, the producer
pool and the extras' share, the hand-off depth) from a cost model in `device/chunk_gdn_device_operation.cpp`,
mirrored in `tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_fused_geometry.py`. The model's constants
are measured values of the kernels, so a kernel change moves them. These scripts measure the quantities the
constants stand for, check the model against the measurements, refit what needs refitting and confirm that the
model still ranks the candidates it decides between in the right order.

The model only ever decides between a few near-equal candidates per head count. It does not need to be exact; it
needs the ranking right where it matters, and its constants need to follow the kernels.

## The model

```
T      = fill(BH, P) + max(C, H + start, X) * (1 + pen) + tail

fill   = kFillAUs + min(BH, 24) * (kFillBUs + kFillCUs * P)          first scan step done; P = producers in total
C      = (NC - 1) * pace + kSkewAUs + kSkewBUs * BH [+ kPoolSkewUs]   the chain after the fill, plus the head skew
pace   = t_step_us(Vtl) + kChainSlopeUs * BH [+ kPoolCreditD2Us * BH: pool, depth 2, Vtl <= 2]
         row-major and not production-bound: pace = max(pace, kRowMajorPaceUs)
H      = (n_home - 1) * kWpUs     n_home = items of the busiest home producer (the shared item map)
X      = (n_extra - 1) * kWpUs    n_extra = items of the busiest extra
pen    = max(bump(C, H), g * bump(H, X))   at Vtl <= 2; the bumps of kBalance* / kPoolBalance*
start  = kPoolStartUs * min(1, share / kPoolStartShare)   pool, Vtl <= 2
tail   = kTailUs
```

Three regimes: chain-bound (H, X well below C: the op is the fill plus NC - 1 steps, producers wait for credits),
production-bound (H or X well above C: the op is the busiest producer's items, receivers wait for input), balanced
(two bounds within 25 % of each other at Vtl <= 2: each side's jitter is exposed to the other).

## What is measured

One capture per geometry: five launches on flat inputs with the WY inverse pinned (`calib_capture.py`), the op's
device time as the median of launches 2..5, and the device zones of the last launch. The zones are named in the
kernels and read per (core, RISC, zone):

| zone | RISC, kernel | gives |
|---|---|---|
| `tx_valid` | BRISC, fused writer | the item period (VALID to VALID over the run), first VALID, producer finish; BRISC holds every item |
| `prep_item` | TRISC_1, prep compute | the item's compute time (cross-check of the period) |
| `tx_wait_cb`, `tx_wait_credit` | BRISC, fused writer | the regime: credit waits = chain-bound, compute waits = production-bound; per class in pools |
| `tx_issue`, `tx_barrier` | BRISC, fused writer | the hand-off's own cost per item |
| `scan_step` | TRISC_1, scan compute | the step; its first end per receiver is the fill |
| `scan_wait_in` | TRISC_0, scan compute | the compute's real input wait per step |
| `rx_wait_valid`, `rx_reserve` | NCRISC, fused receiver | the last chunk's end (+ one step) = the chain end, hence pace, skew and tail |

The profiler keeps ~125 zones per RISC per run: TRISC holds ~21 steps / ~10 items per core, so steady-state medians
come from TRISC and anything spanning the run (periods, chain ends, finish times) from BRISC and NCRISC.

From the zones `calib_collect.py` derives per capture: item and period (the producer's two views of the item time),
fill, step and pace (the receiver's zone time and the chain's actual rate), chain end, skew and tail, the waits on
both sides, the regime, and for pools the items, period, waits and first VALID per class (home / extra).

| constant | measured as | on which rows |
|---|---|---|
| `kWpUs` | the VALID period on production-bound rows (`per`); `prep_item` as the cross-check | pools at BH 32 / 48, per-head NV 2 NP 3 at BH 16 |
| `t_step_us(Vtl)` | chain pace on chain-bound per-head rows at depth 3, `scan_step` as the floor | one NV per Vtl, three BH values each |
| `kChainSlopeUs` | slope of pace vs BH at one Vtl | the same rows |
| `kFill*` | median first `scan_step` end, least squares on (1, min(BH,24), min(BH,24) * P) | all rows |
| `kSkew*`, `kTailUs` | skew = max - median chain end; tail = op - median chain end | chain-bound rows |
| `kRowMajorPaceUs` | chain pace with `row_local=False` | BH 16 |
| `kBalance*`, `kPoolBalance*`, `kPoolStartUs`, `kPoolCreditD2Us` | fitted on the residual of the rows that exercise them, never read off a zone | per-head NP 5 / 6 at BH 8 / 12; the BH 16 share sweep; depth 2 / 3 pairs at BH <= 12 |
| `kPhasedUs[]` | prep + scan op medians of `--phased` captures | BH 4, 8, 12, 16, 32, 48 |

Which change touches what: prep compute -> `kWpUs` (and the share choice, NP 6 vs 7 at BH 12, depth at BH <= 12);
scan compute -> `t_step_us`, `kChainSlopeUs` (and NV 1 vs 2 at BH 16 / 24); the hand-off dataflow -> the depth
terms and the pool Vtl 1 step; the prep reader or kickoff -> `kFill*`, `kPoolStartUs`; placement or routing ->
`kRowMajorPaceUs`, `kSkew*`; the phased prims -> `kPhasedUs[]`.

## The scripts

| file | does | device |
|---|---|---|
| `calib_env.sh` | `TT_METAL_HOME` (default: this repository), `CALIB_OUT` (default `generated/gdn_calib`), the device lock, a private JIT cache; export to override, then `source` | no |
| `calib_capture.py` | the launches of one capture: `--hv BH`, pinned per-head / pool / phased geometry or the op's own pick | yes |
| `calib_rows.py` | the measurement matrix as a rows file (`label script args`) from the tree's own picks; `--sets phased,auto,perhead,pool,share,rowmajor` | no |
| `calib_batch.sh` | one profiler capture per row under the lock; skips rows already done; 8-minute run timeout, board reset, one retry | yes |
| `calib_ops.py` | per-op device medians of a capture, launches 2..n | no |
| `calib_collect.py` | per capture: the op median and the derived quantities above, as JSON rows | no |
| `calib_model.py` | the model port: `--check` (port == the tree's C++ through the binding), `--terms` (direct estimates), `k=v` overrides, `--fit a,b` (coordinate descent on the residual), `--emit` (the C++ and oracle constant blocks) | no |
| `calib_picks.py` | per BH: the pick, the candidates within a band of it, their measurements, `--emit-rows` for the unmeasured ones | no |

The geometry functions come from the ttnn binding (`chunk_gdn_fused_geometry`, `chunk_gdn_fused_placement`,
`chunk_gdn_fused_item_map`, `chunk_gdn_fused_pool_home_producers`), so every number the scripts print about a
geometry is the tree's own.

## The procedure

1. Build the tree with the device profiler and install it into its python_env (`./build_metal.sh --enable-ccache`,
   then `cmake --build build_Release --target install`). Without the install step the venv keeps serving the old
   library, which looks like a kernel hang.
2. `source calib_env.sh`, then `python calib_rows.py --prefix c1 --bh 4,8,12,16,24,32,48 > c1_rows.txt`. Choose
   `--sets` from the change: a prep-only change needs `auto,pool,share` plus two production-bound per-head rows; a
   scan change needs `perhead` at every NV; a full recalibration takes everything (~190 rows, ~10 s each once the
   JIT cache is warm).
3. `./calib_batch.sh c1_rows.txt | tee c1_batch.log`. Re-run it to fill in failures.
4. `python calib_collect.py c1_rows.txt --json c1.json`, on the tree that captured (the op's own picks are resolved
   through the binding).
5. `python calib_model.py c1.json --check` must print a zero maximum difference; otherwise the constants in
   `calib_model.py` are behind the tree and must be updated first. Then `--terms` gives the item period, the step
   per Vtl, the fill fit and the skew directly: set those by hand (`python calib_model.py c1.json W=16.9 step2=2.72`).
6. Fit the residual terms only (the bumps, the start term, the depth-2 credit term), one or two at a time on the
   rows that exercise them: `--fit peak,widthD2 --bh 8,12`, `--fit pool_peak,pool_start --only bh16_pool`. Read
   the per-row errors, not only the mean. `--emit` prints the constant blocks.
7. Paste the constants into `device/chunk_gdn_device_operation.cpp` and the oracle at the top of
   `test_chunk_gdn_fused_geometry.py`; replace the measurement anchors there (`_FUSED_MEASUREMENTS`,
   `_POOL_MEASUREMENTS`, the phased table test, the operating-point tests, the share-choice test) with the new rows
   and picks. Rebuild, install, run the geometry test (host-only) and `test_chunk_gdn_fused.py`.
8. Confirm the picks against their runner-ups: `python calib_picks.py --collected c1.json --emit-rows c2 >
   c2_rows.txt`, batch and collect those, then `python calib_picks.py --collected c1.json c2.json`. A pick within
   run-to-run noise (1-2 %) of its runner-up is right; a runner-up that measures clearly faster means a constant is
   still off, and the pair says which.

Good enough: mean absolute error under 3 % and no row the model decides between above 6 % (rows it never picks, such
as NV 4 at the balance, may sit at 15 % with a wider test tolerance); the fitted item within 0.3 us of the measured
period and each step within 0.1 us of its chain-bound pace; every pick within noise of its runner-ups, including the
share at BH 16 and the depth at BH <= 12; the geometry test green with the mirrored constants.

## Pitfalls

- Medians of launches 2..5, never single samples; run-to-run spread is 1-4 us. Repeat the headline rows.
- Pin the depth. A row without `--nbuf` runs at the model's pick of that day and later analysis cannot tell depth 2
  from 3; `calib_rows.py` always pins it, and the collector takes an unpinned row's depth from a `_d<N>` label
  suffix, else from the current model with a warning.
- Keep the private JIT cache and the device lock: another tree's programs must never be served, and another user may
  hold the device; the batch runner waits for the lock before its run timeout starts.
- After a kernel assert (watcher builds, injected faults) the next run hangs holding the lock; reset the board first.
- The profiler's post-processing occasionally fails with "Start and end marker IDs do not match": rerun the row.
- A source file restored with `cp -p` keeps its old mtime and is not recompiled; restore with `cp` and `touch`.
