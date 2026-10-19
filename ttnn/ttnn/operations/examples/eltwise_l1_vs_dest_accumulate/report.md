# eltwise_l1_vs_dest_accumulate — device report

- box: `bh-qb-11-special-dnijemcevic-for-reservation-42432`
- arch: blackhole
- shape: 64 blocks x 1 tiles (fp32); kernel_iters=100; rounds=10x5
- accumulation steps per launch: 6400
- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)
- takeaway: the win tracks how much L1 traffic the accumulator pays. rmw round-trips acc through
  unpack+add+pack every tile; pack_l1_acc only packs acc (packer folds DEST onto it, never
  unpacks) once per pair; dest_acc keeps the running sum in DEST and touches L1 once, at the end.

| method | ns/op | ±% | vs rmw |
|--------|-------|----|--------|
| rmw | 975683.2 | 0.0 | 1.00x |
| pack_l1_acc | 191806.7 | 0.0 | 5.09x |
| dest_acc | 92150.7 | 0.0 | 10.59x |

## 2026-10-12 - regimes (Wormhole B0 n150), `test_eltwise_l1_vs_dest_accumulate_regimes`
11 interleaved trials per variant, kernel-iters 100, ns per accumulation pass; three runs identical to within 0.1 percent.

| B steps | baseline | median ns | lever | median ns | speedup | baseline min..max | lever min..max |
|---:|---|---:|---|---:|---:|---|---|
| 64 | rmw | 23784 | pack_l1_acc | 5769 | 4.12x | 23653..23791 | 5735..5808 |
| 2 | pack_l1_acc | 367 | dest_acc | 370 | 0.99x | 366..368 | 365..373 |

Exploratory step sweep on WH B0 (`test_eltwise_l1_vs_dest_accumulate_device_perf`, ns per pass; rmw / pack_l1_acc / dest_acc):
B=2: 745 / 368 / 369; B=4: 1484 / 539 / 495; B=8: 2973 / 884 / 754; B=16: 5940 / 1599 / 1278; B=64: 23639 / 5768 / 4447.
With kernel_iters=1 (cold, includes launch/startup) B=2 is 1199 / 555 / 554 ns, B=4 is 1904 / 883 / 602 ns.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz (this test uses 1 Tensix core) |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the blocks above: DEVICE KERNEL DURATION [ns]; `test_eltwise_l1_vs_dest_accumulate_device_perf` (64 blocks × 1 tile, fp32, kernel-iters 100, rounds 10×5, ns per 100 passes) and `test_eltwise_l1_vs_dest_accumulate_regimes` (kernel-iters 100, 11 interleaved trials, ns per accumulation pass, median) |

### Sweep (`test_eltwise_l1_vs_dest_accumulate_device_perf`) — 64 steps × 1 tile, 6400 accumulation steps per launch

| method | ns/op | ±% | vs rmw |
|--------|-------|----|--------|
| rmw | 977408.2 | 0.0 | 1.00x |
| pack_l1_acc | 185677.4 | 0.0 | 5.26x |
| dest_acc | 92178.3 | 0.0 | 10.60x |

### Regimes (`test_eltwise_l1_vs_dest_accumulate_regimes` on BH p150b)

| B steps | baseline | median ns | lever | median ns | speedup | verdict | baseline min..max | lever min..max |
|---:|---|---:|---|---:|---:|---|---|---|
| 64 | rmw | 9772 | pack_l1_acc | 1857 | 5.26x | helps (assert ≥ 2.55) | 9762..9789 | 1856..1857 |
| 64 | pack_l1_acc | 1857 | dest_acc | 922 | 2.01x | helps (assert ≥ 1.15) — WH 1.30x | — | — |
| 2 | pack_l1_acc | 155 | dest_acc | 155 | 1.00x | does_not_help (≤ 1.05) | 155..156 | 155..155 |

Step sweep on BH p150b (speedup per step count B = 2 / 4 / 8 / 16 / 64):
- rmw → pack_l1_acc: 2.25x / 3.15x / 3.89x / 4.58x / 5.27x (WH 2.03x … 4.10x).
- pack_l1_acc → dest_acc: 1.00x / 1.15x / 1.40x / 1.65x / 2.01x (WH 0.99x / 1.09x / 1.17x / 1.25x / 1.30x).

### Reading it
- Same ordering as WH B0 n150 and as the bh-qb-11 run (rmw < pack_l1_acc < dest_acc), but every gap is wider:
  dest_acc vs rmw is 10.60x at 64 steps.
- DEST-resident accumulation pays sooner and more on BH p150b: 1.15x already at 4 steps, 2.01x at 64 (WH 1.09x / 1.30x).
  The marginal cost per step is ~27 ns (pack_l1_acc) vs ~12 ns (dest_acc) on BH p150b, ~87 vs ~66 ns on WH B0 n150.
- B = 2 is flat on both boards: one pair costs one add plus one pack whichever way it is folded.
