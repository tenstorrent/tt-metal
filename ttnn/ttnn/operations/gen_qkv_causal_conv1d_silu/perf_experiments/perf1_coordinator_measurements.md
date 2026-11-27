# Perf 1 — coordinator measurements (raw)

Blackhole 11×10 (110 worker cores), device kernel ns (`DEVICE KERNEL DURATION [ns]`, op CSV column 20, or
`generated/profiler/.logs/cpp_device_perf_report.csv` column 15 when tracy-capture crashed). Harness:
`tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/test_perf_t1.py` (GQKV_PERF_T1=1); zone tables:
`zone_stages.py` (same dir); spatial maps: `coordinator_coremap.py` (this dir). All runs profile the
instrumented kernels (permanent stage zones ON) unless the variant says `nozones*`.

## Breakdown — cumulative ablation (one fresh-cache run per variant)

| variant | 004 | 020 | 003 | 001 |
|---|---|---|---|---|
| full | 130 959 | 60 782 | 9 691 | 6 050 |
| no SiLU | 122 862 | | | |
| no SiLU, no MAC | 110 734 | | | |
| no compute (DM only) | 111 011 | 51 838 | 6 745 | 4 734 |
| no compute, no reads (writes only) | 75 059 | 33 326 | 3 855 | 2 336 |
| sync only | 8 419 | 6 150 | 2 624 | 2 134 |
| no reads (compute + writes) | 114 647 | | | |
| no writes (compute + reads) | 123 041 | | | |
| no DM (compute only) | 108 618 | 52 538 | 6 050 | 3 266 |
| no DM, no SiLU | 40 393 | | | |
| no DM, no SiLU, no MAC (tilize only) | 18 262 | | | |
| no tilize | 149 216 | | | |
| no MAC | 151 216 | | | |
| quartic body pinned | 148 779 | 60 270 | 9 621 (default) | |
| quartic, no DM | 91 871 | | | |

Core balance (`*-KERNEL` span max / p50): 004 full 1.13, DM-only 1.42 (per-core ends 29 → 111 us), writes-only
1.60 (12.8 → 75 us, bottom-right last), quartic 1.47; 020 DM-only 1.34, writes-only 1.57.

## split_writes — whole-op A/B in the real op (WRITE_SHARE 4 vs 0), two sessions, alternating order

| cell | pass 1 (ws0 → S4) | pass 2 (S4, ws0) | verdict |
|---|---|---|---|
| 001 | 6 052 → 6 279 | 6 223 vs 6 056 | +3.3% regression |
| 002 | 6 865 → 7 046 | 7 000 vs 6 885 | +2.2% (noise edge) |
| 003 | 9 750 → 9 022 | 9 102 vs 9 556 | −6.1% win |
| 004 | 130 545 → 123 204 | 122 833 vs 123 908 | flat (baseline bimodal 123.8 / 130.5) |
| 004 quartic | 151 526 → 128 308 | 129 024 vs 147 100 | −14% (still > library 123) |
| 005 | 6 661 → 6 796 | 6 710 vs 6 681 | flat |
| 006 | 5 617 → 5 307 | 5 311 vs 5 564 | −5.0% win |
| 010 | 7 333 → 7 479 | 7 543 vs 7 265 | +2.9% regression (edge) |
| 011 | 11 721 → 11 219 | 11 204 vs 11 739 | −4.5% win |
| 012 | 14 827 → 15 575 | 15 504 vs 14 700 | +5.2% regression |
| 013 | 6 237 → 6 284 | 6 282 vs 6 179 | flat |
| 014 | 8 368 → 8 640 | 8 594 vs 8 479 | +2.3% (noise edge) |
| 016 | 14 636 → 15 599 | 15 526 vs 14 541 | +6.7% regression |
| 019 | 19 528 → 19 250 | 19 331 vs 19 606 | −1.4% flat |
| 020 | 60 537 → 60 166 | 60 385 vs 60 607 | flat |

016 with S4: `reader_out_issue` (NCRISC NoC0 writes + flush, 4 tiles) is 3–5 us on the top four core rows
(y 2–5) and 0.4 us elsewhere; 003 with S4: 0.2 us on every core. Position-dependent NoC0 write congestion.

## Stage-zone overhead under the profiler (full vs MAYBE_DEVICE_ZONE_SCOPE_OFF)

| cell | zones on | all off | compute off | reader off | writer off |
|---|---|---|---|---|---|
| 001 | 6 034 / 6 012 | 5 724 / 5 699 | 5 919 | 5 907 | 5 988 |
| 002 | 6 895 | 6 490 | | | |
| 003 | 9 687 / 9 710 | 9 137 / 9 154 | 9 517 | 9 499 | 9 484 |
| 004 | 123 547 | 131 374 | | | |
| 005 | 6 532 | 6 277 | | | |
| 006 | 5 527 / 5 557 | 5 121 / 5 070 | 5 467 | 5 306 | 5 418 |
| 010 | 7 271 | 6 903 | | | |
| 013 | 6 119 | 5 834 | | | |
| 016 | 14 728 / 14 361 | 14 310 / 14 472 | 14 921 | 14 379 | 14 422 |
| 020 | 60 753 | 59 476 | | | |
