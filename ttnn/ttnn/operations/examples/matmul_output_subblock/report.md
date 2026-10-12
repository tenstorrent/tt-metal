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
