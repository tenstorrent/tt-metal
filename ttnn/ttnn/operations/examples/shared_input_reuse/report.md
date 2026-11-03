# shared_input_reuse — device reports

## Blackhole P150 — bh-qb-11-special-dnijemcevic-for-reservation-42432

Busy AICLK, enabled GDDR count, and firmware were not captured by the original run.

- arch: Arch.BLACKHOLE
- cores: 22 (2 × 11 grid); injector = top-left (0,0)
- shared input: a large matrix [R=9728, C=128] = 1216 tiles (2.38 MB bf16),
  streamed in 19 chunks of 16×4 = 64 tiles (cb_in holds one chunk — L1 can't hold the whole input)
- job: fold the whole stream into 1 running tile-sum/core in DEST (output = 22 tiles << input); rounds=10x5
- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)
- baseline reads the whole 2.38 MB stream from DRAM on every core (22× = 52.2 MB of DRAM reads); mcast
  reads each chunk once on the injector and NoC-broadcasts it, so DRAM sees the stream once.

| variant | ns/op | ±% | vs per_core_dram |
|---------|-------|----|------------------|
| per_core_dram | 135415.3 | 0.3 | 1.00x |
| mcast | 78987.4 | 0.1 | 1.71x |

## Blackhole P100a — bh-43-special-sjovic-for-reservation-97381

- arch: blackhole
- busy AICLK: 1350 MHz
- enabled GDDR banks: 7
- firmware: 19.12.0
- measured: 2026-09-23
- cores: 22 (2 × 11 grid); injector = top-left (0,0)
- shared input: a large matrix [R=9728, C=128] = 1216 tiles (2.38 MB bf16), streamed in 19 chunks of
  16×4 = 64 tiles
- job: fold the whole stream into 1 running tile-sum/core in DEST; rounds=10x5

| variant | ns/op | ±% | vs per_core_dram |
|---------|-------|----|------------------|
| per_core_dram | 161935.5 | 0.2 | 1.00x |
| mcast | 88860.5 | 0.0 | 1.82x |

## Wormhole B0 n150 — regime measurements (2026-10-12)

`test_shared_input_reuse_regimes`; 16 Tensix cores (2 x 8), injector (0,0); bf16; median of 10 interleaved windows x 5 launches. Same-stream chunk-size sweep (1216 tiles; ranges are min-max over the 10 windows):

| chunk tiles | chunks | per_core_dram ns | mcast ns | speedup |
|---|---|---|---|---|
| 1 | 1216 | 632497 (631672-633217) | 905974 (905624-906628) | 0.70x |
| 2 | 608 | 384640 | 541511 | 0.71x |
| 4 | 304 | 253944 (252902-254823) | 353620 (353418-353820) | 0.72x |
| 8 | 152 | 214490 (213272-215400) | 262540 (262352-262734) | 0.82x |
| 16 | 76 | 205610 (204783-207063) | 221500 (221252-221660) | 0.93x |
| 32 | 38 | 208500 (207125-209495) | 200334 (199982-200628) | 1.04x |
| 64 | 19 | 255839 (253145-258434) | 191315 (190979-191756) | 1.34x |
| 128 (1280 tiles) | 10 | 287511 (281953-292885) | 198901 (198451-199062) | 1.45x |

Stream-length sweep at 64-tile chunks: 1 chunk 17856 -> 15268 ns (1.17x); 2: 28522 -> 24874 (1.15x); 4: 54526 -> 44359 (1.23x); 8: 107834 -> 83630 (1.29x); 40: 528122 -> 395822 (1.33x).

Test runs (helps / does_not_help speedup): 1.32x / 0.82x; 1.34x / 0.82x; 1.34x / 0.82x.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests, Release build |
| worker rectangle | 2 × 12 = 24 Tensix cores, injector = top-left (0,0) |
| metric / method | DEVICE KERNEL DURATION [ns], median of 10 interleaved windows × 5 launches (`test_shared_input_reuse_device_perf`, `test_shared_input_reuse_regimes`, and a probe using the regimes test's `_measure_interleaved_ns`) |

### Sweep (`test_shared_input_reuse_device_perf`)

Shared input [R=9728, C=128] = 1216 tiles (2.38 MiB bf16), 19 chunks of 16×4 = 64 tiles. The baseline reads
the whole stream from DRAM on every Tensix core (24× = 57.0 MiB of DRAM reads); `mcast` reads each chunk once.

| variant | ns/op | ±% | vs per_core_dram |
|---------|-------|----|------------------|
| per_core_dram | 150179.5 | 0.2 | 1.00x |
| mcast | 77760.8 | 0.1 | 1.93x |

### Boundary sweep / regimes (`test_shared_input_reuse_regimes` on BH p150b, plus same-code probe)

Same-stream chunk-size sweep (1216 tiles unless noted; ranges = min-max over the 10 windows where recorded):

| chunk tiles | chunks | per_core_dram ns | mcast ns | speedup | source |
|---|---|---|---|---|---|
| 4 | 304 | 169285 | 247015 | 0.69x | probe |
| 8 | 152 | 148567 (148167-148897) | 154494 (154410-154698) | 0.96x | regime `does_not_help_chunk8tiles_1216tiles` (assert <= 1.0) |
| 16 | 76 | 149415 | 113451 | 1.32x | probe; regime `measure_chunk16tiles_1216tiles` (logged only — verdict differs by arch) |
| 32 | 38 | 150358 | 90532 | 1.66x | probe |
| 64 | 19 | 150228 (149554-150601) | 77777 (77707-77864) | 1.93x | regime `helps_chunk64tiles_1216tiles` (assert >= 1.17) |
| 128 (1280 tiles) | 10 | 164538 | 76613 | 2.15x | probe |

### Reading it

- Crossover between 8- and 16-tile chunks on BH p150b, vs ~32 tiles on WH B0 n150: 16-tile chunks lose on
  WH B0 (0.93x) and win on BH p150b (1.32x).
- `per_core_dram` is flat at ~148–150 µs for 8- to 64-tile chunks: 24 readers × 2 490 368 B = 59.8 MB at
  ~400 GB/s, i.e. at the BH p150b DRAM ceiling (~415–430 GB/s) regardless of chunk size.
- `mcast` grows by ~0.54–0.67 µs per extra chunk (the per-chunk ack-and-broadcast handshake), about the same
  as WH B0 n150 (~0.48–0.60 µs). What differs is the injector's stream itself: ~65 µs for 2.49 MB on BH p150b
  vs ~180 µs on WH B0, leaving room for ~150 handshakes under the baseline instead of ~45.
- The 64-tile gain (1.93x) is the largest of the four boards measured (WH B0 n150 1.34x with 16 Tensix cores,
  BH P150 1.71x and P100a 1.82x with 22): more readers make the redundant baseline costlier.

## Blackhole p150b — bh-50-special-adamjanovic-for-reservation-143315 (2026-10-30)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 13×10 = 130 Tensix cores; busy AICLK, DRAM bank count and firmware not captured |
| commit | `30351ce33f5` + the resident-window variants below (uncommitted), Release build |
| metric / method | DEVICE KERNEL DURATION [ns]; regimes: median of 10 interleaved windows × 5 launches; window sweep: median over 3 DRAM placements (a pad tensor shifts every buffer by a different page count) of the median of 3 interleaved windows × 5 launches, [min–max] = speedup range over the placements |

### Streamed shared input (re-measured, same code)

The 2 × grid_x worker rectangle is 2 × 13 = **26 Tensix cores** on this board (24 on the earlier p150b). Same
1216-tile stream; `SIR_CHUNK_ROWS`/`SIR_D_COLS` probe of `test_shared_input_reuse_device_perf` (10 × 5 launches,
report redirected) plus `test_shared_input_reuse_regimes`:

| chunk tiles | chunks | per_core_dram ns | mcast ns | speedup | source |
|---|---|---|---|---|---|
| 1 (`d_cols` = 1) | 1216 | 542 976 (541 576–544 526) | 844 086 (843 896–844 216) | 0.64x | regime `does_not_help_chunk1tile_1216tiles` (assert <= 1.0) |
| 2 (`d_cols` = 2) | 608 | 310 047 | 442 906 | 0.70x | probe |
| 4 | 304 | 270 407 | 256 299 | 1.06x | probe |
| 8 | 152 | 242 398 (239 081–245 239) | 161 552 (161 529–161 690) | 1.50x | regime `measure_chunk8tiles_1216tiles` (was `does_not_help`, assert <= 1.0: fails here) |
| 16 | 76 | 232 619 (228 309–234 936) | 114 680 (114 576–114 723) | 2.03x | regime `measure_chunk16tiles_1216tiles` |
| 32 | 38 | 188 303 | 91 767 | 2.05x | probe |
| 64 | 19 | 177 524 (175 371–179 612) | 78 360 (78 290–78 441) | 2.27x | regime `helps_chunk64tiles_1216tiles` (assert >= 1.17) |

Reading it: `mcast` is within 1–5% of the earlier p150b (78.4 vs 77.8 µs at 64-tile chunks, 161.6 vs 154.5 µs at
8-tile chunks). The baseline is what changed: 26 readers × 2 490 368 B = 64.7 MB takes 177.5 µs at 64-tile chunks
(365 GB/s) but 242 µs at 8-tile chunks (267 GB/s), where the earlier board stayed flat at ~400 GB/s. With only one
chunk in flight per Tensix core per barrier, the per-core reads here are latency-bound, so the crossover moved from
between 8 and 16 tiles to between 2 and 4 tiles per chunk. The 8-tile regime now flips between boards and became
log-only; a 1-tile-chunk regime (WH B0 n150 0.70x, here 0.64x) carries the does-not-help assertion.

### Resident window, one-shot broadcast (new)

`out = x * w`: x [R, W] and w [1, W] are Layout::TILE bf16 in DRAM (TensorMemoryLayout::INTERLEAVED). x's
tile-rows are split over a rectangle of Tensix cores (contiguous tile ids per core; reader NoC 0, writer NoC 1,
8-tile blocks, 2 blocks deep); every Tensix core keeps all W/32 tiles of w resident in L1 (a 2048-byte tile per 32
elements of w). `r` = window bytes ÷ x bytes per Tensix core = 1 ÷ tile-rows of x per Tensix core.
`per_core`: every Tensix core reads w from DRAM after its first x block. `oneshot_mcast`: the top-left Tensix core
reads w in 32-tile chunks and multicasts each chunk to the rectangle, then multicasts a semaphore set (chunks
landed); no handshake. `lockstep_mcast`: the same with an ack-then-broadcast handshake per chunk, and the whole
window before the first x block.

Sweep (`test_shared_input_reuse_window_sweep`, 32-tile chunks; GB/s = x read + out write ÷ `per_core` time):

| Tensix cores | window tiles (bytes) | x MiB | x tile-rows per Tensix core | r | per_core ns | oneshot_mcast ns | lockstep_mcast ns | oneshot speedup | lockstep speedup | per_core GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1 (2 048 B) | 16.0 | 512 | 0.002 | 202 104 | 152 714 | 153 940 | 1.32x [1.19-1.36] † | 1.31x [1.29-1.35] † | 166 |
| 16 | 1 (2 048 B) | 64.0 | 2048 | 0.000 | 566 077 | 465 404 | 455 859 | 1.22x [1.17-1.25] † | 1.24x [1.20-1.25] † | 237 |
| 16 | 8 (16 384 B) | 16.0 | 64 | 0.016 | 153 792 | 152 746 | 150 636 | 1.01x [0.98-1.04] | 1.02x [0.97-1.08] | 218 |
| 16 | 8 (16 384 B) | 64.0 | 256 | 0.004 | 481 114 | 465 651 | 461 297 | 1.03x [0.95-1.06] | 1.04x [0.95-1.05] | 279 |
| 16 | 32 (65 536 B) | 16.0 | 16 | 0.06 | 148 114 | 138 295 | 141 434 | 1.07x [1.04-1.07] | 1.05x [1.01-1.05] | 227 |
| 16 | 32 (65 536 B) | 64.0 | 64 | 0.016 | 443 179 | 429 540 | 430 444 | 1.03x [1.01-1.05] | 1.03x [0.99-1.04] | 303 |
| 16 | 128 (262 144 B) | 16.0 | 4 | 0.25 | 173 140 | 146 583 | 147 514 | 1.18x [1.12-1.19] | 1.17x [1.10-1.23] | 194 |
| 16 | 128 (262 144 B) | 64.0 | 16 | 0.06 | 452 786 | 436 900 | 428 950 | 1.04x [1.02-1.09] | 1.06x [1.05-1.10] | 296 |
| 16 | 256 (524 288 B) | 16.0 | 2 | 0.5 | 187 858 | 152 786 | 156 359 | 1.23x [1.21-1.26] | 1.20x [1.20-1.24] | 179 |
| 16 | 256 (524 288 B) | 64.0 | 8 | 0.125 | 469 405 | 440 744 | 429 878 | 1.07x [1.06-1.10] | 1.09x [1.09-1.13] | 286 |
| 32 | 1 (2 048 B) | 16.0 | 256 | 0.004 | 114 783 | 113 516 | 111 498 | 1.01x [0.99-1.05] | 1.03x [1.00-1.05] | 292 |
| 32 | 1 (2 048 B) | 64.0 | 1024 | 0.001 | 572 966 | 485 428 | 465 772 | 1.18x [1.18-1.24] † | 1.23x [1.20-1.30] † | 234 |
| 32 | 8 (16 384 B) | 16.0 | 32 | 0.03 | 112 959 | 111 565 | 110 927 | 1.01x [0.98-1.03] | 1.02x [0.98-1.06] | 297 |
| 32 | 8 (16 384 B) | 64.0 | 128 | 0.008 | 491 916 | 483 892 | 481 822 | 1.02x [0.99-1.02] | 1.02x [1.02-1.03] | 273 |
| 32 | 32 (65 536 B) | 16.0 | 8 | 0.125 | 113 595 | 110 462 | 108 348 | 1.03x [1.01-1.04] | 1.05x [1.04-1.08] | 295 |
| 32 | 32 (65 536 B) | 64.0 | 32 | 0.03 | 438 971 | 444 809 | 428 528 | 0.99x [0.98-1.00] | 1.02x [1.01-1.03] | 306 |
| 32 | 128 (262 144 B) | 16.0 | 2 | 0.5 | 130 046 | 110 874 | 111 019 | 1.17x [1.17-1.18] | 1.17x [1.15-1.18] | 258 |
| 32 | 128 (262 144 B) | 64.0 | 8 | 0.125 | 437 854 | 454 947 | 441 503 | 0.96x [0.96-0.96] | 0.99x [0.98-1.00] | 307 |
| 32 | 256 (524 288 B) | 16.0 | 1 | 1.0 | 149 199 | 126 212 | 126 959 | 1.18x [1.17-1.19] | 1.18x [1.17-1.18] | 225 |
| 32 | 256 (524 288 B) | 64.0 | 4 | 0.25 | 443 425 | 450 163 | 447 676 | 0.99x [0.97-0.99] | 0.99x [0.98-1.00] | 303 |
| 64 | 1 (2 048 B) | 16.0 | 128 | 0.008 | 93 771 | 97 776 | 95 784 | 0.96x [0.96-0.96] | 0.98x [0.97-0.98] | 358 |
| 64 | 1 (2 048 B) | 64.0 | 512 | 0.002 | 480 002 | 480 600 | 477 999 | 1.00x [0.99-1.02] | 1.00x [0.99-1.01] | 280 |
| 64 | 8 (16 384 B) | 16.0 | 16 | 0.06 | 97 451 | 97 881 | 96 336 | 1.00x [0.99-1.00] | 1.01x [1.01-1.02] | 344 |
| 64 | 8 (16 384 B) | 64.0 | 64 | 0.016 | 468 204 | 470 195 | 479 218 | 1.00x [0.97-1.01] | 0.98x [0.97-0.98] | 287 |
| 64 | 32 (65 536 B) | 16.0 | 4 | 0.25 | 102 472 | 97 902 | 96 781 | 1.05x [1.04-1.06] | 1.06x [1.05-1.07] | 327 |
| 64 | 32 (65 536 B) | 64.0 | 16 | 0.06 | 471 302 | 463 536 | 466 580 | 1.02x [1.01-1.03] | 1.01x [1.00-1.02] | 285 |
| 64 | 128 (262 144 B) | 16.0 | 1 | 1.0 | 139 440 | 103 124 | 105 233 | 1.35x [1.30-1.38] | 1.33x [1.28-1.34] | 241 |
| 64 | 128 (262 144 B) | 64.0 | 4 | 0.25 | 491 781 | 467 450 | 466 894 | 1.05x [1.03-1.07] | 1.05x [1.04-1.07] | 273 |
| 64 | 256 (524 288 B) | 32.0 ‡ | 1 | 1.0 | 269 580 | 229 891 | 229 733 | 1.17x [1.17-1.18] | 1.17x [1.16-1.18] | 249 |
| 64 | 256 (524 288 B) | 64.0 | 2 | 0.5 | 508 617 | 480 011 | 476 647 | 1.06x [1.05-1.08] | 1.07x [1.05-1.08] | 264 |
| 130 | 1 (2 048 B) | 16.0 | 63 | 0.016 | 86 321 | 87 892 | 87 137 | 0.98x [0.98-0.98] | 0.99x [0.99-0.99] | 389 |
| 130 | 1 (2 048 B) | 64.0 | 252 | 0.004 | 367 851 | 339 902 | 344 834 | 1.08x [1.08-1.09] | 1.07x [1.06-1.08] | 365 |
| 130 | 8 (16 384 B) | 16.0 | 7.9 | 0.13 | 90 056 | 91 025 | 87 112 | 0.99x [0.99-0.99] | 1.03x [1.03-1.04] | 373 |
| 130 | 8 (16 384 B) | 64.0 | 31.5 | 0.03 | 348 518 | 348 063 | 346 724 | 1.00x [1.00-1.01] | 1.01x [1.01-1.01] | 385 |
| 130 | 32 (65 536 B) | 16.0 | 2 | 0.5 | 111 065 | 91 589 | 89 211 | 1.21x [1.18-1.22] | 1.24x [1.21-1.25] | 302 |
| 130 | 32 (65 536 B) | 64.0 | 7.9 | 0.13 | 373 774 | 358 502 | 354 215 | 1.04x [1.04-1.05] | 1.06x [1.05-1.06] | 359 |
| 130 | 128 (262 144 B) | 32.5 ‡ | 1 | 1.0 | 271 336 | 179 437 | 185 955 | 1.51x [1.47-1.53] | 1.46x [1.41-1.49] | 251 |
| 130 | 128 (262 144 B) | 64.0 | 2 | 0.5 | 440 084 | 363 526 | 360 701 | 1.21x [1.19-1.22] | 1.22x [1.20-1.24] | 305 |
| 130 | 256 (524 288 B) | 65.0 ‡ | 1 | 1.0 | 531 714 | 380 228 | 388 098 | 1.40x [1.39-1.40] | 1.37x [1.36-1.38] | 256 |

‡ the 16 MiB point is below one tile-row per Tensix core, so the sweep raised it to one tile-row each (the
16 MiB and 64 MiB rows for 64 cores × 256 tiles and 130 cores × 128/256 tiles collapse to the stream shown).
† not a window effect, see "Reading it". Regimes (`test_shared_input_reuse_regimes`, 10 × 5 launches, one
placement):

| regime | case | per_core ns | lever ns | speedup |
|---|---|---|---|---|
| `window_helps_fullgrid_w128_r050` (assert >= 1.10) | 130 Tensix cores, 128-tile window, x [8320, 4096] = 65 MiB, r = 0.5 | 437 387 (434 787–441 597) | oneshot 364 948 (363 375–369 502) | 1.20x |
| `window_does_not_help_fullgrid_w8_r0125` (assert <= 1.05) | 130 Tensix cores, 8-tile window, x [33280, 256] = 16.2 MiB, r = 0.125 | 92 383 (91 580–94 501) | oneshot 92 609 (92 132–92 950) | 1.00x |
| `window_does_not_help_32cores_w128_r0125` (assert <= 1.05) | 32 Tensix cores (4 × 8), 128-tile window, x [8192, 4096] = 64 MiB, r = 0.125 | 442 016 (437 720–446 510) | oneshot 442 592 (434 894–458 428) | 1.00x |
| `window_does_not_help_lockstep_chunk1` (assert <= 1.05) | 130 Tensix cores, 32-tile window in 1-tile chunks, x [8320, 1024] = 16.2 MiB, r = 0.5 | 124 342 (123 877–125 025) | lockstep 141 805 (141 345–142 284) | 0.88x |

Chunk size at 130 Tensix cores, x 16 MiB (32-tile window) or one tile-row per Tensix core (256-tile window); median of
5 × 5 launches at one placement, except the 8-tile-chunk row (median of 2 placements × 3 × 5 launches, from the CLI).
Each chunk size is a separate session, so `per_core` moves between rows by up to 9%:

| window tiles | chunk tiles (handshakes) | per_core ns | oneshot_mcast ns | lockstep_mcast ns | oneshot | lockstep |
|---|---|---|---|---|---|---|
| 32 | 1 (32) | 123 161 | 100 299 | 140 367 | 1.23x | 0.88x |
| 32 | 4 (8) | 113 654 | 90 795 | 99 468 | 1.25x | 1.14x |
| 32 | 32 (1) | 123 862 | 116 683 | 112 083 | 1.06x | 1.11x |
| 256 | 1 (256) | 615 802 | 496 040 | 797 324 | 1.24x | 0.77x |
| 256 | 4 (64) | 566 778 | 406 434 | 477 302 | 1.39x | 1.19x |
| 256 | 8 (32) | 558 165 | 394 222 | 425 109 | 1.42x | 1.31x |
| 256 | 32 (8) | 591 387 | 445 796 | 458 719 | 1.33x | 1.29x |

### Reading it

- **The deciding quantity is r, not the Tensix-core count.** `per_core` adds one window of DRAM reads to every
  Tensix core on top of that core's own x read and out write, so the expected gain is about
  1 + r/2 when the stream is DRAM-bound. Measured: r ≥ 0.5 gives 1.06–1.51x (1.21–1.51x on the full grid,
  1.06–1.35x on 16–64 Tensix cores); r = 0.25 gives 0.99–1.18x; r ≤ 0.125 gives 0.96–1.07x, never a gain worth
  having. On the full grid the time saved matches the DRAM ceiling: 130 × 256 KB = 34 MB saved 77 µs at
  64 MiB (≈ 440 GB/s), 130 × 64 KB = 8.5 MB saved 19 µs at 16 MiB.
- **Few Tensix cores with a long stream do not gain.** At 32 Tensix cores and 8 tile-rows each (r = 0.125)
  `oneshot_mcast` is 0.96–1.00x; at 64 Tensix cores and 4 tile-rows (r = 0.25) 1.05x. The same windows on a
  short stream (r ≥ 0.5) gain 1.17–1.35x, so it is the window's share of each core's bytes, not the core count,
  that decides.
- **Windows ≤ 16 KB (8 tiles) never gain** on ≥ 64 Tensix cores: 0.96–1.03x.
- **Waiting for the whole window costs nothing; the handshake does.** Every Tensix core's first tile-row needs
  all W/32 window tiles, so the stream stalls until the whole window is resident in every variant. With one
  32-tile chunk, `lockstep_mcast` equals `oneshot_mcast`. Each extra ack-then-broadcast round costs
  ≈ 1.2–1.3 µs on 130 Tensix cores: 32 handshakes cost +40 µs (0.88x of `per_core`), 256 cost +300 µs (0.77x).
- **† Narrow streams on 16–32 Tensix cores are timing-sensitive.** With a 1-tile (2 KB) window, x is one tile
  wide, and `per_core` was 1.18–1.32x slower than `oneshot_mcast` on 16–32 Tensix cores. A 2 KB read cannot cost
  50 µs: a debug build of `per_core` that skipped the window read (wrong output, timing only) measured 154 µs
  against `oneshot_mcast`'s 156 µs (16 Tensix cores, 16 MiB), and with ±1 tile-row of stream the same point
  swung 0.83–1.33x. The per-core read shifts when each core starts, and the latency-bound one-tile-wide stream
  amplifies it. This is not a byte-count effect, so the rule above ignores it.
