# SFPU work-scoping — isolated MATH-thread cost, whole tile vs a scoped face subset

box=bh-50-special-mstaletovic-for-reservation-49528  arch=BH  clock=1350MHz  cores=1  placement=single-core sharded-L1  N=5 (median)  reps=2000 (in-kernel math loop)
metric: MATH-thread (TRISC_1) ns per SFPU call, from a DeviceZoneScopedN around a math-only loop.
copy(seed) and pack are OUTSIDE the zone, so the number is pure SFPU math cycles — no unpack, no
pack, no CB handshake, no per-tile copy/pack floor. Input bf16, one Tensix core, sharded L1.

scopes (32-lane vector ops in []): rc (RC, whole tile [32]; BASELINE) | r (R, top half [16]) | c (C, left half [16]) | r_iter2 (R + ITERATIONS=2, ROW 0 [4]) | c_skip (C even-parity stride, COL 0 [8]) | face (None, face 0 [8]) | face_iter1 (None + IT=1, [0,0] [1]). none = empty loop (overhead).

## func = rsqrt — MATH ns per SFPU call; (speedup vs rc); ns per vector op

| scope | how | vec ops | math ns/call | speedup vs rc | ns / vector |
|---|---|---|---|---|---|
| none | empty loop (no SFPU) | — | 0.0±20% | — | — |
| rc | VectorMode::RC (4 faces) | 32 | 748.2±0% | 1.00x | 23.4 |
| r | VectorMode::R (2 faces, top) | 16 | 378.5±0% | 1.98x | 23.7 |
| c | VectorMode::C (2 faces, left) | 16 | 378.5±0% | 1.98x | 23.7 |
| r_iter2 | VectorMode::R + ITERATIONS=2 | 4 | 103.1±0% | 7.26x | 25.8 |
| c_skip | VectorMode::C, even-parity stride (dst_reg+=2) | 8 | 194.9±0% | 3.84x | 24.4 |
| face | VectorMode::None (face 0) | 8 | 188.9±0% | 3.96x | 23.6 |
| face_iter1 | VectorMode::None + ITERATIONS=1 | 1 | 28.3±0% | 26.46x | 28.3 |

isolation check: max unpack=0.007 ns/call, max pack=0.007 ns/call inside the zone (≈0 → the SFPU is alone on the math thread).

## func = recip — MATH ns per SFPU call; (speedup vs rc); ns per vector op

| scope | how | vec ops | math ns/call | speedup vs rc | ns / vector |
|---|---|---|---|---|---|
| none | empty loop (no SFPU) | — | 0.0±26% | — | — |
| rc | VectorMode::RC (4 faces) | 32 | 890.4±0% | 1.00x | 27.8 |
| r | VectorMode::R (2 faces, top) | 16 | 449.6±0% | 1.98x | 28.1 |
| c | VectorMode::C (2 faces, left) | 16 | 449.6±0% | 1.98x | 28.1 |
| r_iter2 | VectorMode::R + ITERATIONS=2 | 4 | 120.8±0% | 7.37x | 30.2 |
| c_skip | VectorMode::C, even-parity stride (dst_reg+=2) | 8 | 88.2±0% | 10.09x | 11.0 |
| face | VectorMode::None (face 0) | 8 | 224.5±0% | 3.97x | 28.1 |
| face_iter1 | VectorMode::None + ITERATIONS=1 | 1 | 32.7±0% | 27.22x | 32.7 |

isolation check: max unpack=0.007 ns/call, max pack=0.007 ns/call inside the zone (≈0 → the SFPU is alone on the math thread).

Notes: an SFPU vector op = 4 rows x 8 stride-2 columns; a 32x32 tile is 32 vector ops (4 faces x 4 row-groups x 2 column parities). The MATH cost is ~flat per vector op (see ns/vector), so the ladder is just how many vector ops the scope runs: rc=32 -> r/c=16 -> c_skip/face=8 -> r_iter2=4 -> face_iter1=1. Two axis-optimal tricks vs the coarse half-tile modes: for a ROW-0 result the row waste is the OUTER walk axis, so ITERATIONS truncates it — r_iter2 = VectorMode::R + ITERATIONS=2 keeps the top row-group of both top faces = 4 vectors (a pure knob turn). For a COL-0 result the waste is column PARITY, the INNER walk axis (vectors alternate even/odd), so ITERATIONS can't isolate it — c_skip strides the DEST address by 2 (raw sfpi) to keep only the even-parity vectors (which hold column 0) of the two left faces = 8 vectors. That is why r_iter2 (4) beats c_skip (8), and why the row trick is a knob but the column trick needs raw sfpi. This is the SFPU cost in ISOLATION; in a full op the copy/pack/DRAM around it dilute the win, and a data-movement-bound op won't show it. recip costs more per vector than rsqrt (heavier op); its c_skip uses the Newton reciprocal body (the stock recip fast path uses SFPLOADMACRO addressing that can't be strided the same way), so its per-vector cost differs slightly from stock recip.

## 2026-10-12 - regime measurements (Wormhole B0 n150)
Source: `test_sfpu_tile_scope_regimes`, 1 Tensix core, bf16 tile in sharded L1, 11 interleaved trials per variant; median (min-max). The report above this section was measured on Blackhole.

| func | metric | baseline | lever | speedup |
|---|---|---:|---:|---:|
| rsqrt | math zone ns/call, reps=2000 | rc 1139.1 (1139.1-1139.1) | r_iter2 156.1 (156.1-156.2) | 7.30x |
| recip | math zone ns/call, reps=2000 | rc 563.1 | r_iter2 84.2 | 6.69x |
| rsqrt | whole op kernel ns, reps=1 | rc 1642 (1634-1656) | r_iter2 739 (738-749) | 2.22x |
| recip | whole op kernel ns, reps=1 | rc 1106 (1089-1124) | r_iter2 688 (685-708) | 1.61x |
| rsqrt | math zone ns/call, reps=2000 | c 576.1 (576.1-576.1) | r 576.1 (576.1-576.1) | 1.00x |
| recip | math zone ns/call, reps=2000 | c 288.1 | r 288.1 | 1.00x |

Three runs: rsqrt zone 7.30x x3, recip zone 6.69x x3, rsqrt whole-op 2.21/2.22/2.22x, recip whole-op 1.59/1.61/1.60x, c-vs-r 1.00x x3 for both.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz (this test uses 1 Tensix core) |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the blocks above: `test_sfpu_tile_scope_device_perf` (MATH-thread (TRISC_1) ns per SFPU call, N=5 median, reps=2000) and `test_sfpu_tile_scope_regimes` (11 interleaved trials, median (min-max); math zone reps=2000, whole op = DEVICE KERNEL DURATION with reps=1) |

### Sweep (`test_sfpu_tile_scope_device_perf`)

#### func = rsqrt — MATH ns per SFPU call; (speedup vs rc); ns per vector op

| scope | how | vec ops | math ns/call | speedup vs rc | ns / vector |
|---|---|---|---|---|---|
| none | empty loop (no SFPU) | — | 0.0±18% | — | — |
| rc | VectorMode::RC (4 faces) | 32 | 795.6±0% | 1.00x | 24.9 |
| r | VectorMode::R (2 faces, top) | 16 | 402.1±0% | 1.98x | 25.1 |
| c | VectorMode::C (2 faces, left) | 16 | 402.2±0% | 1.98x | 25.1 |
| r_iter2 | VectorMode::R + ITERATIONS=2 | 4 | 109.0±0% | 7.30x | 27.2 |
| c_skip | VectorMode::C, even-parity stride (dst_reg+=2) | 8 | 206.8±0% | 3.85x | 25.8 |
| face | VectorMode::None (face 0) | 8 | 200.8±0% | 3.96x | 25.1 |
| face_iter1 | VectorMode::None + ITERATIONS=1 | 1 | 29.7±0% | 26.75x | 29.7 |

isolation check: max unpack=0.007 ns/call, max pack=0.007 ns/call inside the zone (≈0 → the SFPU is alone on the math thread).

#### func = recip — MATH ns per SFPU call; (speedup vs rc); ns per vector op

| scope | how | vec ops | math ns/call | speedup vs rc | ns / vector |
|---|---|---|---|---|---|
| none | empty loop (no SFPU) | — | 0.0±0% | — | — |
| rc | VectorMode::RC (4 faces) | 32 | 206.8±0% | 1.00x | 6.5 |
| r | VectorMode::R (2 faces, top) | 16 | 109.0±0% | 1.90x | 6.8 |
| c | VectorMode::C (2 faces, left) | 16 | 109.0±0% | 1.90x | 6.8 |
| r_iter2 | VectorMode::R + ITERATIONS=2 | 4 | 37.9±0% | 5.46x | 9.5 |
| c_skip | VectorMode::C, even-parity stride (dst_reg+=2) | 8 | 88.2±0% | 2.34x | 11.0 |
| face | VectorMode::None (face 0) | 8 | 54.2±0% | 3.82x | 6.8 |
| face_iter1 | VectorMode::None + ITERATIONS=1 | 1 | 12.7±0% | 16.28x | 12.7 |

isolation check: max unpack=0.007 ns/call, max pack=0.007 ns/call inside the zone (≈0 → the SFPU is alone on the math thread).

### Regimes (`test_sfpu_tile_scope_regimes` on BH p150b)

| func | metric | baseline | lever | speedup | verdict |
|---|---|---:|---:|---:|---|
| rsqrt | math zone ns/call, reps=2000 | rc 795.6 (795.6-795.6) | r_iter2 109.0 (109.0-109.0) | 7.30x | helps (assert ≥ 4.1) |
| recip | math zone ns/call, reps=2000 | rc 206.8 (206.7-206.8) | r_iter2 37.9 (37.9-37.9) | 5.46x | helps (assert ≥ 3.23) — WH 6.69x |
| rsqrt | whole op kernel ns, reps=1 | rc 1042 (1029-1053) | r_iter2 459 (444-490) | 2.27x | helps (assert ≥ 1.6) |
| recip | whole op kernel ns, reps=1 | rc 606 (583-619) | r_iter2 438 (425-449) | 1.38x | helps (assert ≥ 1.19) — WH 1.61x |
| rsqrt | math zone ns/call, reps=2000 | c 402.2 (402.1-402.2) | r 402.2 (402.2-402.2) | 1.00x | does_not_help (≤ 1.05) |
| recip | math zone ns/call, reps=2000 | c 109.0 (109.0-109.0) | r 109.0 (109.0-109.0) | 1.00x | does_not_help (≤ 1.05) |

### Reading it
- rsqrt reproduces the bh-50 run within 7 % (24.9 vs 23.4 ns per vector op) and matches WH B0 n150's ratios exactly
  (7.30x math zone); it costs about the same cycles per vector op on both arches (~34 vs ~36).
- recip is much cheaper than on the older bh-50 run (rc 206.8 vs 890.4 ns; 6.5 vs 27.8 ns per vector op) and than
  on WH B0 n150 (17.6 ns per vector op). Per-call fixed cost is then a larger share of the short `r_iter2` call
  (9.5 ns per vector op), so recip's scope gain is lower here: 5.46x math zone, 1.38x whole op (WH 6.69x / 1.61x).
- recip `c_skip` (Newton body, 11.0 ns per vector op) is now slower per vector than stock recip, so its 2.34x vs rc is
  less than the 3.82x of `face` at the same 8 vectors — the opposite of the bh-50 table above (10.09x).
- The c → r null holds to the nanosecond for both functions.
