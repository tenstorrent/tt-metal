# matmul_output_subblock — device report

- box: `bh-qb-11-special-dnijemcevic-for-reservation-42432`
- arch: Arch.BLACKHOLE
- shape: M=16 N=16 K=1 tiles (single K-block); kernel_iters=100; rounds=10x5
- output tiles: 256; DEST capacity = 8 tiles (fp16, no fp32 accumulator)
- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)
- takeaway: the matmul_block SRC-register operand reuse is the lever. A wide subblock (sb_w)
  loads one A-tile and reuses it across sb_w B-tiles; a tall subblock (sb_h) reuses one B-tile
  across sb_h A-rows; 1x1 reloads both operands per output tile. Bigger subblock -> fewer SRC
  loads (+ fewer per-subblock cycles) -> faster, up to the DEST budget (8 tiles).

| variant | sb (h×w) | reuses | ns/op | ±% | vs 1x1 |
|---------|----------|--------|-------|----|--------|
| sb_1x1 | 1×1 | none | 1039712.3 | 0.0 | 1.00x |
| sb_1x8 | 1×8 | A (across B) | 710175.8 | 0.0 | 1.46x |
| sb_8x1 | 8×1 | B (across A) | 710369.7 | 0.0 | 1.46x |
| sb_2x4 | 2×4 | A (across B) | 712838.4 | 0.0 | 1.46x |
| sb_4x2 | 4×2 | B (across A) | 712937.4 | 0.0 | 1.46x |
| sb_2x2 | 2×2 | A (across B) | 741177.5 | 0.0 | 1.40x |

## 2026-10-12 - regimes (Wormhole B0 n150), `test_matmul_output_subblock_regimes`
11 interleaved trials per variant, kernel-iters 20, ns per matmul; three runs identical to within 0.1 percent.

| Mt x Nt, Kt (tiles) | baseline | median ns | lever | median ns | speedup | baseline min..max | lever min..max |
|---|---|---:|---|---:|---:|---|---|
| 16x16, 1 | sb_1x1 | 24312 | sb_2x4 | 13905 | 1.75x | 24273..24325 | 13886..13915 |
| 16x16, 1 | sb_8x1 | 19167 | sb_1x8 | 13662 | 1.40x | 19151..19179 | 13647..13676 |
| 2x2, 1 | sb_1x1 | 670 | sb_2x2 | 656 | 1.02x | 666..675 | 654..659 |

Wormhole B0 Kt sweep at 16x16 output tiles (ns/op per 100 in-kernel iterations, vs sb_1x1), 10 trials, `test_matmul_output_subblock_device_perf`:
- Kt=1: 1x1 2418890; 1x8 1.77x; 8x1 1.26x; 2x4 1.74x; 4x2 1.58x; 2x2 1.56x.
- Kt=2: 1x1 3591548; 1x8 1.57x; 8x1 1.22x; 2x4 1.64x; 4x2 1.66x; 2x2 1.46x.
- Kt=4: 1x1 7018784; 1x8 1.61x; 8x1 1.36x; 2x4 1.69x; 4x2 1.69x; 2x2 1.51x.
- Kt=8: 1x1 13626296; 1x8 1.61x; 8x1 1.42x; 2x4 1.68x; 4x2 1.68x; 2x2 1.53x.
- Kt=12: 1x1 20256916; 2x4 1.69x; 4x2 1.69x. Kt=16: does not fit L1.
Small outputs (1x1 -> largest legal subblock, Kt=1): 2x2 1.01x; 1x8 (Mt=1,Nt=8) 1.07x; 2x4 (Mt=2,Nt=4) 1.06x; 4x4 1.27x (2x4); 2x2 at Kt=8 1.34x.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz (this test uses 1 Tensix core) |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the blocks above: DEVICE KERNEL DURATION [ns]; `test_matmul_output_subblock_device_perf` (kernel-iters 100, rounds 10×5, ns per 100 iterations) and `test_matmul_output_subblock_regimes` (kernel-iters 20, 11 interleaved trials, ns per matmul, median) |

### Sweep (`test_matmul_output_subblock_device_perf`) — M=16 N=16 K=1 tiles, 256 output tiles, DEST = 8 tiles

| variant | sb (h×w) | reuses | ns/op | ±% | vs 1x1 |
|---------|----------|--------|-------|----|--------|
| sb_1x1 | 1×1 | none | 1041241.2 | 0.0 | 1.00x |
| sb_1x8 | 1×8 | A (across B) | 711794.1 | 0.0 | 1.46x |
| sb_8x1 | 8×1 | B (across A) | 712006.2 | 0.0 | 1.46x |
| sb_2x4 | 2×4 | A (across B) | 714668.8 | 0.0 | 1.46x |
| sb_4x2 | 4×2 | B (across A) | 714518.3 | 0.0 | 1.46x |
| sb_2x2 | 2×2 | A (across B) | 742847.0 | 0.0 | 1.40x |

Reproduces the bh-qb-11 table at the top of this file to within 0.3 %.

### Regimes (`test_matmul_output_subblock_regimes` on BH p150b)

| Mt x Nt, Kt (tiles) | baseline | median ns | lever | median ns | speedup | verdict | baseline min..max | lever min..max |
|---|---|---:|---|---:|---:|---|---|---|
| 16x16, 1 | sb_1x1 | 10429 | sb_2x4 | 7161 | 1.46x | helps (assert ≥ 1.23) | 10427..10435 | 7159..7162 |
| 16x16, 1 | sb_8x1 | 7129 | sb_1x8 | 7128 | 1.00x | measure (logged only — verdict differs by arch) — WH 1.40x | 7128..7131 | 7128..7130 |
| 16x16, 4 | sb_1x1 | 29238 | sb_2x4 | 26654 | 1.10x | measure (logged only — verdict differs by arch) — WH ~1.69x | — | — |
| 2x2, 1 | sb_1x1 | 372 | sb_2x2 | 382 | 0.97x | does_not_help (≤ 1.05) | 370..376 | 380..386 |

Boundary sweep (BH p150b, `sb_1x1` → best 8-tile subblock, or largest legal subblock for small outputs):
- 16×16 output tiles: Kt=1 1.46x; Kt=2 1.17x; Kt=4 1.10x; Kt≥8 does not fit L1 at 16×16 output tiles on this build.
- Kt=1: 8×8 output tiles 1.34x; 4×4 1.08x. 2×2 output tiles at Kt=8: 0.98x (WH 1.34x).

### Reading it
- The Kt=1 subblock win is 1.46x on BH p150b vs 1.75x on WH B0 n150, and on BH it collapses with K (1.17x at Kt=2,
  1.10x at Kt=4) where WH B0 n150 holds 1.66–1.69x up to Kt=12.
- Tall `8×1` == wide `1×8` on BH p150b (1.00x), as on bh-qb-11; the WH B0 n150 orientation penalty (`sb_8x1` 1.26x
  vs `sb_1x8` 1.77x) does not exist on Blackhole. Any 8-tile subblock is fine on BH.
- BH p150b's `sb_1x1` baseline is 2.3x faster than WH B0 n150's (10429 vs 24312 ns) — more than the 1.35x AICLK ratio —
  so what a bigger subblock saves on BH is mostly the per-subblock fixed cost, which K inner steps already amortize.
- Small outputs gain nothing on BH p150b even with deep K (2×2 at Kt=8: 0.98x); on WH B0 n150 deep K rescued them (1.34x).
