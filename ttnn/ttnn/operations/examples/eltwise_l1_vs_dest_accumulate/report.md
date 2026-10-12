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
