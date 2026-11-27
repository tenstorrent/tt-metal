# Perf 2 — coordinator measurements (raw)

Blackhole 11×10 (110 worker cores), device kernel ns (`DEVICE KERNEL DURATION [ns]`: op CSV, or
`generated/profiler/.logs/cpp_device_perf_report.csv` when tracy-capture crashed), permanent stage zones ON
in every run. One fresh-cache profiled run per entry. Harnesses:
- breakdown: `tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/test_perf_t1.py` (GQKV_PERF_T1=1);
- whole-op A/B and plan pins: `test_perf2.py` in this dir (GQKV_PERF2=1). Variant `base` runs the pre-Perf-2
  op (descriptor + kernels frozen from git HEAD `0aa6e9dbf08` in `head/`), so before/after share a session.

## Breakdown (cumulative peel), current op = Perf 1 HEAD

| variant | 004 | 001 | 002 | 003 |
|---|---|---|---|---|
| full | 127 644 | 5 942 | 6 881 | 9 541 |
| no SiLU | 120 554 | 5 524 | 5 812 | 8 019 |
| no SiLU, no MAC | 112 676 | | | |
| DM only (all compute stubbed) | 110 428 | 4 812 | 5 157 | 6 858 |
| writes only | 74 964 | 2 253 | 2 475 | 4 049 |
| sync only | 8 334 | 2 113 | 2 428 | 2 593 |
| compute only (no DM) | 108 547 | 3 384 | 4 339 | 6 184 |
| compute only, no SiLU | 40 441 | 2 656 | 3 319 | 4 136 |
| tilize only | 18 273 | | | |
| quartic body pinned (004) | 147 924 | | | |
| no reads / no writes (004) | 114 691 / 123 866 | | | |

004 zones (run 1024 full / 10240 quartic / 4096 DM-only of `/tmp/r2_004_ladder.csv`):
- full: TRISC_2 `compute_conv_silu` 92.99 p50 / 93.03 max (uniform), `compute_tilize` 24.4 p50; TRISC_2 span
  p50 114.5, max 126.7 (max/p50 1.11) — the pack thread is the critical path.
- quartic: TRISC_2 `compute_conv_silu` 76.2 uniform; BRISC span max/p50 1.46; tail at physical x 13-15,
  y 6-10: `writer_out_issue` up to 55 us, `writer_split_barrier` up to 43, `writer_split_issue` up to 36,
  TRISC_0 `compute_rows_wait` 55-78 (vs ~10 at the left).
- DM only: BRISC end 28 -> 110 us, smooth gradient top-left -> bottom-right; NoC0 `reader_rows_barrier`
  uniform 7-13 us.

## Combined op (work stealing + Perf 2 SiLU body), plan pins — NOT graduated (claim hang, see changelog)

| cell | default (split 16) | split 8 | split 24 | library body | library + split 16 |
|---|---|---|---|---|---|
| 004 | 112 561 | 110 426 | 122 885 | 124 803 | 124 665 |
| 020 | 50 683 | 61 958 | 56 249 | 61 365 | |
| 020 fp32 DEST (library) | 72 755 (split 16) | | 70 456 | | |
| u1multi [1,1024,1952] (library / bf16 body) | 95 380 / 90 327 | | | | |

Combined vs base A/B (pass 1 base->combined, pass 2 combined->base), us:
004 123.6->112.4 / 123.7 vs 117.2; 020 60.7->52.0 / 60.6 vs 48.9; 008 16.1->13.2 / 15.8 vs 12.9;
019 19.44->20.07 / 19.65 vs 20.12 (+2.4..3.3%: owner's first claim on the critical path); single-row cells
3-10% faster (= the SiLU body); u1multi with split 16: 73.9 vs 90.8 (split 24).

Claim-protocol hang (stealing suite `test_work_stealing.py`, archived here, 24 tests incl. stress schedule):
optimizer's protocol 3/4 runs HANG; with the host pre-claim of block 0 1/4 (+ one earlier); with full
NOC_RET_ADDR programming (logical node id) + 4-slot pre-armed sentinel ring 2/4. Signature every time: one
reader spins in `claim_fetch_add` on its sentinel after `noc_async_atomic_barrier()` passed, all other cores
done. Every first failure observed was a stress-schedule cell (tileT256 / rmT256 / 008).

## Graduated op (Perf 2 SiLU body only) vs base — two passes, alternating order

| cell | base A | base B | grad A | grad B | d A | d B |
|---|---|---|---|---|---|---|
| 004 | 124 275 | 129 244 | 123 903 | 123 409 | -0.3% | -4.5% (base bimodal) |
| 020 | 60 344 | 60 592 | 60 544 | 60 602 | +0.3% | 0.0% |
| 001 | 6 039 | 6 066 | 5 844 | 5 873 | -3.2% | -3.2% |
| 002 | 6 936 | 6 824 | 6 443 | 6 497 | -7.1% | -4.8% |
| 003 | 9 515 | 9 579 | 8 697 | 8 795 | -8.6% | -8.2% |
| 005 | 6 590 | 6 516 | 6 115 | 6 073 | -7.2% | -6.8% |
| 006 | 5 530 | 5 585 | 5 153 | 5 146 | -6.8% | -7.9% |
| 007 | 9 678 | 9 596 | 8 693 | 8 726 | -10.2% | -9.1% |
| 010 | 7 254 | 7 213 | 6 947 | 6 819 | -4.2% | -5.5% |
| 011 | 11 689 | 11 579 | 11 670 | 11 637 | -0.2% | +0.5% |
| 012 | 14 793 | 15 099 | 13 464 | 13 390 | -9.0% | -11.3% |
| 013 | 6 134 | 6 082 | 5 755 | 5 719 | -6.2% | -6.0% |
| 014 | 8 342 | 8 344 | 7 841 | 7 847 | -6.0% | -6.0% |
| 015 | 5 709 | 5 779 | 5 456 | 5 339 | -4.4% | -7.6% |
| 016 | 14 403 | 14 302 | 13 094 | 13 453 | -9.1% | -5.9% |
| 017 | 4 950 | 4 915 | 4 771 | 4 767 | -3.6% | -3.0% |
| 019 | 19 536 | 19 401 | 19 566 | 19 564 | +0.2% | +0.8% |
| u1multi | 89 538 | 91 503 | 93 004 | 92 589 | +3.9% | +1.2% (library either way) |

After narrowing the library carve-out to >= 5 tile-rows per core (008 now runs the bf16 body), two passes:
008 16 157 / 15 941 -> 15 257 / 15 256 (-5.6% / -4.3%); 004 131 345 / 123 904 -> 124 114 / 124 267 (flat);
020 60 858 / 60 737 -> 60 285 / 60 473 (flat); 019 19 690 / 19 489 -> 19 468 / 19 435 (flat).

Multi-row cells, library (default before narrowing) vs bf16 body pinned, two passes:
004 125 078 / 125 045 -> 141 474 / 139 359 (+13.1% / +11.4%); 020 60 628 / 60 193 -> 62 865 / 62 451
(+3.7% / +3.8%); 008 16 610 / 16 190 -> 15 217 / 15 533 (-8.4% / -4.1%); u1multi 93 024 / 92 716 ->
92 323 / 93 113 (flat).
