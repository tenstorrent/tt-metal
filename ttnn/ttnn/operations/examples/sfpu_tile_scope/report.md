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
