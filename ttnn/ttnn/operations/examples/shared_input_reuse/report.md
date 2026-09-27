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
