# whole_k_rowstream_v3 (Perf 20): the whole-K row stream on the round-20 tree

This is Perf 19's whole-K row stream (`../whole_k_rowstream_v2/`), ported onto the current working tree. It coexists
with the whole-stream fill split (`RM_SPLIT`), band-first kh reuse and the routed store (`STORE_ROUTE`).

The core's valid output tile-rows run flat, in batches of `rows`. For each batch, compute:
- tilizes the batch's units;
- accumulates all of K in DEST against the resident weights;
- packs the fp32 sums once;
- adds the bias one row group at a time.

There are no M-blocks, no per-K-block packs or inits, and no padding rows. The output leaves at the compute rate.

## Change (`integration.patch` = `diff -ruN op_base op_copy`; op_base = the working tree at round start)

### Host
`rowstream_plan` re-spends the L1 that the fp32 partials ([bm, bn]) and in0 ([bm, bk]) used. The new uses are:
- in0 [rows, K];
- the batch's fp32 sums;
- the unit FIFO (lag rows);
- a ring grown to what the lag needs;
- a deeper output CB.

The blocking solve does not change with `ROWSTREAM_KHR = "keep"`, the default.

Domain rule: `not output_stream_bound(g, b, fidelity)`. This is the graduated function that already gates `RM_SPLIT`
and `STORE_ROUTE`.

Compute receives `actual_m_tiles` as runtime argument 3.

### Reader
The flat loop replaces the M-block loop. For each tile-row:
1. `rm.release(lowest live pixel)`. The FIFO depth bounds which entries compute has popped.
2. `rm.need_whole(x)`. Under `RM_SPLIT` this is `need_split`, which raises `hungry`, so the writer's `serve()` takes
   back claims.
3. `rm.keep_busy()` until the staging is full.
4. One unit per K-block, with `base_for(row lo)`.

There is no new fill API: v2's `need_stream` and the startup-window wait are gone.

### Compute
`conv2d_3x3_rowstream.hpp`, `walk` with `BlockIn0`. It uses the batch bias only: v2 measured the K-step bias slower
everywhere.

### Writer
Every weight round is issued up front. Store windows of `ROWSTREAM_STORE_GROUPS` row groups go through
`out_store.write_page`, so the routed store applies where `STORE_ROUTE` is on. The writer runs `rm_w.serve()` while it
waits for a window.

### Reach
Only the DRAM-interleaved ROW_MAJOR ring with resident weights, `sb_h == 1`, and neither kh reuse nor the C_out fold.
The row stream does not reach:
- TILE input;
- sharded programs (Panoptic);
- the one-M-block programs (idx 31 / 32 / 34);
- C_in 32 (excluded by the rule).

## Results
Setup: Blackhole p150b, bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1). Values are device kernel us, base in the
same run, one call per run.

| idx | case | calls | base | row stream | delta |
|---|---|---|---|---|---|
| 47 P13 | 514x1024 128->512 | 1 | 5334.6 | 4794.7 | -10.1% |
| 45 P12 | 514x1024 128->256 | 1 | 2750.4 | 2469.4 | -10.2% |
| 46 P6 | 1026x2048 64->256 | 1 | 5284.5 | 4862.3 | -8.0% |
| 44 P5 | 1026x2048 64->128 | 1 | 2894.5 | 2673.7 / 2673.8 | -7.6% |
| 43 P7 | 258x512 128->512 | 8 | 1293.6 | 1247.8 | -3.5% |
| 40 P14 | 258x512 128->256 | 1 | 732.6 (median of 4) | 676.3 (median of 4) | -7.7% |
| 41 P1, rule off | 1026x2048 32->128 | 5 | 1779.1 | 2619.6 | **+47%** |
| 36 P4, rule off | 514x1024 32->128 | 5 | 544.5 | 638.3 | **+17%** |
| P1 / P4 rule on, P2 / P8 / P10 / P11 keep | | | | | same program, bit-identical |

Call-weighted over the 17 cases: 58.90 -> 57.01 ms (-3.2%).

Valid rows only: the K-blocked walk multiplies padding rows on the last M-block, and the flat walk does not. This
accounts for part of the P14 and P7 wins:
- P14: 42 -> 38 rows (-9.5% of matmul);
- P7: 77 -> 75 rows (-2.6%).

v2 (nominal rows) measured P14 +2.0% and P7 -0.4%, so once the K-blocked walk stops multiplying padding rows, the row
stream is about flat on P14 and P7. P5 / P6 / P12 / P13 have at most 0.7% padding rows, so their wins come from the
row stream itself.

## Menu
- **`replace`: the row stream instead of band-first kh reuse.**
  - P8 397.3 -> 373.6 (-6.0%).
  - P2 1268.6 -> 1280.7 and 1270.1 -> 1281.7 (+1.0% / +0.9%, flat).
  - P10 358.5 -> 356.1 (flat).
  - P11 729.7 -> 733.2 (flat).
  - Call-weighted: a further -84 us.
  - `select_blocking` has no fidelity, so this mode assumes HiFi4 for the rule there.
- **Batch of 8 rows:** P5 2647.7 against 2673.8 at 4 rows (-1.0%, flat). The P6 run was lost to the full disk.
- **Extra regimes, `replace`, base -> row stream:**
  - 258x512 64->256, padding 1, no bias, ReLU: 401.1 -> 347.4.
  - The same with bias and ReLU: 397.5 -> 373.4.
  - 258x512 128->256, padding 1, no bias, ReLU: 698.7 -> 657.6.
  - The same with bias and ReLU: 734.8 -> 677.8.

## Precision
With bias:
- PCC vs torch is 0.9999986, equal to base on every case.
- 0.004-0.016% of outputs differ from base.
- Where |ref| >= mean|ref| / 16, they differ by at most 1 bf16 ulp. With padding 1 + ReLU, 1-2 outputs differ by 2 ulp.
- The larger distances are near-zero values, with 0-14 sign flips (|value| <= 1.2e-4).
- The worst error in reference ulps equals base's.

Without bias:
- About 4.5% of outputs differ from base, by up to 8 ulp.
- The row stream is closer to fp32 than base: worst error 0.863 vs 0.867 and 0.774 vs 0.890 reference ulps. Base
  reloads its partials through srcA (tf32).

`--dev` (watcher, LLK asserts) passes on P14, P12 (the tightest ring: rows 1, lag 1) and the no-bias ReLU regime.

## Remaining overhead (zones, P14)
The first batch waits about 87 us (`compute_tilize` 0+87). Every core fetches two full image rows at once before any
tile-row has all three kernel rows. After that each batch takes 31 us, at the matmul rate.

The K-blocked walk hides part of this startup by multiplying kernel row 0 while bands 1 and 2 arrive. A fill that fetches
the first batch's three slivers first would remove most of it. That would also turn `replace` into a win on P2.

## Re-run
- `run.sh <label> <cases> <variants> [ENV=...]` runs in the foreground and deletes its profiler report and trace copy.
- `python3 -I summ.py <label>` summarizes PCC, ulp and the plan.
- `python3 -I rs_plan.py op_copy [KNOB=value ...]` prints the host plan per focus case.
- Variants are defined in `test_rs.py`:
  - `rs` is the default: keep kh reuse, rule on.
  - `rsa`: rule off.
  - `rsx`: replace kh reuse.
  - `rs8`: batches of 8 rows.
  - `off`: the row stream disabled (equals base).
