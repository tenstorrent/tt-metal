# AccumulateViaAdd vs ReduceTile — perf across math fidelity x reduce dim x tiles/output (single core)

box=bh-50-special-mstaletovic-for-reservation-48793  arch=BH  cores=1  N=3 (median)  kernel-iters=100  input bf16, output fp32, fp32 accum. R = reduced tiles per (single) output tile.
ns = median device-kernel ns per whole-block reduce. (×) = reduce_tile_ns / acc_via_add_ns (>1 = fast path wins).
acc = max_abs error vs fp64 (fp32 accum).

## LoFi

| dim | R | reduce_tile ns | acc_via_add ns | × | rt acc | ava acc |
|---|---:|---:|---:|---:|---|---|
| row | 2 | 234 | 447 | 0.52× | 1.8e-03 | 2.8e-04 |
| row | 8 | 388 | 507 | 0.77× | 1.5e-03 | 1.4e-04 |
| row | 32 | 1013 | 720 | 1.41× | 1.4e-03 | 5.9e-05 |
| col | 2 | 206 | 356 | 0.58× | 1.1e-02 | 2.4e-04 |
| col | 8 | 316 | 414 | 0.76× | 9.9e-03 | 1.4e-04 |
| col | 32 | 726 | 634 | 1.15× | 9.5e-03 | 6.4e-05 |
| scalar | 2 | 276 | 527 | 0.52× | 2.4e-02 | 1.5e-05 |
| scalar | 8 | 568 | 586 | 0.97× | 2.6e-02 | 7.1e-06 |
| scalar | 32 | 1729 | 801 | 2.16× | 2.6e-02 | 8.3e-06 |

## HiFi2

| dim | R | reduce_tile ns | acc_via_add ns | × | rt acc | ava acc |
|---|---:|---:|---:|---:|---|---|
| row | 2 | 275 | 447 | 0.61× | 1.8e-03 | 2.8e-04 |
| row | 8 | 540 | 507 | 1.07× | 1.5e-03 | 1.4e-04 |
| row | 32 | 1609 | 720 | 2.23× | 1.4e-03 | 5.9e-05 |
| col | 2 | 232 | 357 | 0.65× | 2.3e-04 | 2.4e-04 |
| col | 8 | 386 | 414 | 0.93× | 1.4e-04 | 1.4e-04 |
| col | 32 | 1005 | 633 | 1.59× | 6.1e-05 | 6.4e-05 |
| scalar | 2 | 327 | 527 | 0.62× | 6.1e-03 | 1.5e-05 |
| scalar | 8 | 764 | 586 | 1.30× | 5.9e-03 | 7.1e-06 |
| scalar | 32 | 2509 | 800 | 3.13× | 5.9e-03 | 8.3e-06 |

## HiFi3

| dim | R | reduce_tile ns | acc_via_add ns | × | rt acc | ava acc |
|---|---:|---:|---:|---:|---|---|
| row | 2 | 347 | 447 | 0.77× | 2.8e-04 | 2.8e-04 |
| row | 8 | 827 | 507 | 1.63× | 1.3e-04 | 1.4e-04 |
| row | 32 | 2746 | 720 | 3.81× | 5.9e-05 | 5.9e-05 |
| col | 2 | 266 | 357 | 0.75× | 2.3e-04 | 2.4e-04 |
| col | 8 | 525 | 414 | 1.27× | 1.4e-04 | 1.4e-04 |
| col | 32 | 1575 | 634 | 2.49× | 6.1e-05 | 6.4e-05 |
| scalar | 2 | 368 | 527 | 0.70× | 5.2e-04 | 1.5e-05 |
| scalar | 8 | 940 | 587 | 1.60× | 5.4e-04 | 7.1e-06 |
| scalar | 32 | 3221 | 800 | 4.02× | 5.3e-04 | 8.3e-06 |

## HiFi4

| dim | R | reduce_tile ns | acc_via_add ns | × | rt acc | ava acc |
|---|---:|---:|---:|---:|---|---|
| row | 2 | 418 | 447 | 0.93× | 2.8e-04 | 2.8e-04 |
| row | 8 | 1109 | 507 | 2.19× | 1.3e-04 | 1.4e-04 |
| row | 32 | 3884 | 720 | 5.39× | 5.9e-05 | 5.9e-05 |
| col | 2 | 300 | 356 | 0.84× | 2.3e-04 | 2.4e-04 |
| col | 8 | 667 | 414 | 1.61× | 1.4e-04 | 1.4e-04 |
| col | 32 | 2141 | 634 | 3.38× | 6.1e-05 | 6.4e-05 |
| scalar | 2 | 415 | 527 | 0.79× | 4.1e-04 | 1.5e-05 |
| scalar | 8 | 1118 | 586 | 1.91× | 4.0e-04 | 7.1e-06 |
| scalar | 32 | 3929 | 800 | 4.91× | 4.2e-04 | 8.3e-06 |

## Speedup pivot (reduce_tile ns / acc_via_add ns)

| dim | R | LoFi | HiFi2 | HiFi3 | HiFi4 |
|---|---:|---:|---:|---:|---:|
| row | 2 | 0.52× | 0.61× | 0.77× | 0.93× |
| row | 8 | 0.77× | 1.07× | 1.63× | 2.19× |
| row | 32 | 1.41× | 2.23× | 3.81× | 5.39× |
| col | 2 | 0.58× | 0.65× | 0.75× | 0.84× |
| col | 8 | 0.76× | 0.93× | 1.27× | 1.61× |
| col | 32 | 1.15× | 1.59× | 2.49× | 3.38× |
| scalar | 2 | 0.52× | 0.62× | 0.70× | 0.79× |
| scalar | 8 | 0.97× | 1.30× | 1.60× | 1.91× |
| scalar | 32 | 2.16× | 3.13× | 4.02× | 4.91× |
