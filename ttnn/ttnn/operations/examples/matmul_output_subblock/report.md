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
