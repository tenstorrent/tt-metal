# eltwise_l1_vs_dest_accumulate — device report

- box: `bh-50-special-malimpic-for-reservation-134223`
- arch: Arch.BLACKHOLE
- shape: 64 blocks x 1 tiles (fp32); kernel_iters=100; rounds=10x5
- accumulation steps per launch: 6400
- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)
- takeaway: the win tracks how much L1 traffic the accumulator pays. rmw round-trips acc through
  unpack+add+pack every tile; pack_l1_acc only packs acc (packer folds DEST onto it, never
  unpacks) once per pair; dest_acc keeps the running sum in DEST and touches L1 once, at the end.

| method | ns/op | ±% | vs rmw |
|--------|-------|----|--------|
| rmw | 977641.9 | 0.0 | 1.00x |
| pack_l1_acc | 185658.3 | 0.0 | 5.27x |
| dest_acc | 92162.0 | 0.0 | 10.61x |
