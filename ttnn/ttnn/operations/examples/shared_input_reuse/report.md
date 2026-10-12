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
