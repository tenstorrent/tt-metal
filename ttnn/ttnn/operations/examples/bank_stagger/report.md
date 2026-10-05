# bank_stagger — measured report

| stamp | value |
|---|---|
| box | `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` |
| arch | Blackhole p300c (11×10 = 110 compute grid, 8 DRAM banks) |
| date | 2026-10-05 |
| metric | `DEVICE KERNEL DURATION [ns]`, read in-process (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 3 warmup launches per variant, then 21 trials × 10 launches, variants interleaved per trial, median of trials; every variant checked bit-exact first |

> Illustrative, not a CI bound. Add another arch as a new block rather than replacing this one.

Op: tilize bf16, `ROW_MAJOR` interleaved DRAM → `TILE` interleaved DRAM, one 32-row × `chunk`-tile
unit per core. `nt_h` = tile-rows, `n_w` = units per tile-row (cores sharing the same rows):
`nt_h = 1` is width blocking, `n_w = 1` height blocking. `spread` = (max − min) / median over trials.

## Blackhole — 1 block per core (N=21, kernel-iters=1)

```
         shape chunk read B nt_h  n_w cores  variant        ns  spread  vs none
     32x7040       2    128    1  110   110  none         7456    1.5%    base
     32x7040       2    128    1  110   110  read         7174    2.2%  1.039x

     32x56320     16   1024    1  110   110  none        19066    2.2%    base
     32x56320     16   1024    1  110   110  read        17669    0.8%  1.079x

   3520x512       16   1024  110    1   110  none        18580    1.5%    base
   3520x512       16   1024  110    1   110  read        17593    1.4%  1.056x
```

## Reading

- `read` saves 1.0–1.4 µs per launch at 1024 B reads (1.06–1.08×) and 0.3 µs at 128 B (1.04×),
  in both width and height blocking.
- Not shown because they measure on par with the baseline or with `read` (within the spread):
  `write`, `both`, and any variant with the work repeated 8 times inside one launch.

## Wormhole B0 (earlier, 64 cores, 12 banks)

Box `bgd-lab-16-special-dstoiljkovic-for-reservation-114068`, commit `3f32706f595`, 1 block per core.
`both` was 1.07–1.14× on width blocking with ≥ 256 B reads. Unlike BH, the write switch carried the
gain at `chunk = 4` (1.09–1.12×), where every core's first write landed on 3 of 12 banks.
