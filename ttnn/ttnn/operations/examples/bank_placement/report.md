# bank_placement — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-16-special-dstoiljkovic-for-reservation-114068` |
| arch | Wormhole B0 (8×8 = 64 compute grid, 12 DRAM banks) |
| commit | `3f32706f595` (+ this example) |
| date | 2026-10-05 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 3 warmup launches per variant, then N trials × 10 launches, variants interleaved per trial, median of trials; every variant checked bit-exact before timing |

> Numbers are illustrative of the *effect*, not a CI bound — single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.bank_placement` to measure your own sizes.
> A different arch should be **appended** as a new block, not overwritten.

Op: DRAM interleaved → DRAM interleaved bf16 copy, 3072 pages, **12 cores = one per DRAM bank**,
reads on NoC0, writes on NoC1, 8 pages per barrier, double-buffered CB. GB/s counts read + write bytes.
`spread` = (max − min) / median over the trials.

## Wormhole B0 — run 1 (N=5)

```
bank_placement   box=bgd-lab-16-special-dstoiljkovic-for-reservation-114068  arch=WORMHOLE_B0  grid=8x8  N=5 trials x 10 launches (median)  kernel-iters=1  block=8
  op = DRAM interleaved -> DRAM interleaved copy, one core per DRAM bank (bank id = list index)
    row_major      cores: 0,0 1,0 2,0 3,0 4,0 5,0 6,0 7,0 0,1 1,1 2,1 3,1
    bank_near      cores: 3,7 0,0 0,3 0,4 4,0 7,7 4,1 4,6 4,5 6,2 4,3 4,4
    bank_shuffled  cores: 4,1 4,6 4,5 6,2 4,3 4,4 3,7 0,0 0,3 0,4 4,0 7,7

  pattern  pages page B     MB  placement             ns   GB/s  spread  vs row_major
  affine    3072    512    1.6  row_major          31125    102.8    1.9%    base
  affine    3072    512    1.6  bank_near          28613    111.8    0.9%  1.088x
  affine    3072    512    1.6  bank_shuffled      31633    101.2    1.6%  0.984x

  spread    3072    512    1.6  row_major          28422    112.6    1.6%    base
  spread    3072    512    1.6  bank_near          28905    110.7    1.4%  0.983x
  spread    3072    512    1.6  bank_shuffled      28935    110.6    1.3%  0.982x

  affine    3072   2048    6.3  row_major          72398    174.0    0.3%    base
  affine    3072   2048    6.3  bank_near          72041    174.9    0.1%  1.005x
  affine    3072   2048    6.3  bank_shuffled     117423    107.3    0.3%  0.617x

  spread    3072   2048    6.3  row_major          64408    195.6    0.2%    base
  spread    3072   2048    6.3  bank_near          97825    128.8    0.6%  0.658x
  spread    3072   2048    6.3  bank_shuffled      97507    129.2    1.3%  0.661x

  affine    3072   8192   25.2  row_major         256677    196.4    1.0%    base
  affine    3072   8192   25.2  bank_near         243439    207.0    0.2%  1.054x
  affine    3072   8192   25.2  bank_shuffled     557190     90.5    0.7%  0.461x

  spread    3072   8192   25.2  row_major         280854    179.5    1.4%    base
  spread    3072   8192   25.2  bank_near         466319    108.1    0.8%  0.602x
  spread    3072   8192   25.2  bank_shuffled     466533    108.0    1.0%  0.602x

```

(Run 1's GB/s column was mis-scaled when printed and is recomputed here from its ns.)

## Wormhole B0 — run 2 (N=7)

```
bank_placement   box=bgd-lab-16-special-dstoiljkovic-for-reservation-114068  arch=WORMHOLE_B0  grid=8x8  N=7 trials x 10 launches (median)  kernel-iters=1  block=8
  op = DRAM interleaved -> DRAM interleaved copy, one core per DRAM bank (bank id = list index)
    row_major      cores: 0,0 1,0 2,0 3,0 4,0 5,0 6,0 7,0 0,1 1,1 2,1 3,1
    bank_near      cores: 3,7 0,0 0,3 0,4 4,0 7,7 4,1 4,6 4,5 6,2 4,3 4,4
    bank_shuffled  cores: 4,1 4,6 4,5 6,2 4,3 4,4 3,7 0,0 0,3 0,4 4,0 7,7

  pattern  pages page B     MB  placement             ns   GB/s  spread  vs row_major
  affine    3072    512    1.6  row_major          31251  100.7    1.6%    base
  affine    3072    512    1.6  bank_near          28587  110.0    1.7%  1.093x
  affine    3072    512    1.6  bank_shuffled      31633   99.4    1.8%  0.988x

  spread    3072    512    1.6  row_major          28432  110.6    0.9%    base
  spread    3072    512    1.6  bank_near          28977  108.6    1.3%  0.981x
  spread    3072    512    1.6  bank_shuffled      28966  108.6    1.7%  0.982x

  affine    3072   2048    6.3  row_major          72422  173.7    0.5%    base
  affine    3072   2048    6.3  bank_near          71942  174.9    0.2%  1.007x
  affine    3072   2048    6.3  bank_shuffled     117605  107.0    0.9%  0.616x

  spread    3072   2048    6.3  row_major          64382  195.4    0.9%    base
  spread    3072   2048    6.3  bank_near          97648  128.9    1.5%  0.659x
  spread    3072   2048    6.3  bank_shuffled      97376  129.2    0.7%  0.661x

  affine    3072   8192   25.2  row_major         256501  196.2    1.0%    base
  affine    3072   8192   25.2  bank_near         243386  206.8    0.1%  1.054x
  affine    3072   8192   25.2  bank_shuffled     557764   90.2    1.0%  0.460x

  spread    3072   8192   25.2  row_major         281984  178.5    1.8%    base
  spread    3072   8192   25.2  bank_near         466463  107.9    0.8%  0.605x
  spread    3072   8192   25.2  bank_shuffled     467095  107.8    1.0%  0.604x

```

## Reading

- **Which bank a core serves matters a lot once each core has a home bank.** `bank_near` vs
  `bank_shuffled` use the *same 12 cores*; only the core ↔ bank pairing differs. With 2 KB pages
  the matched pairing is **1.63×** faster (72 vs 118 µs), with 8 KB pages **2.29×** (243 vs 558 µs).
  The mismatched pairing sends every core's traffic across the grid to a far bank, and the routes
  overlap. At 512 B pages the copy is limited by how fast each core can issue (~100 GB/s) and the
  pairing matters much less (1.11×).
- **But a plain row-major line is already almost as good.** `bank_near` beats `row_major` by only
  1.01–1.09×. The line along rows 0–1 is short-haul to the banks and its routes do not overlap, so
  there is little left to win.
- **The bank-near cores are a bad choice for anything else.** In the `spread` control every core walks
  all 12 banks; the bank-near set is then **0.60–0.66×** of row-major at 2 KB and 8 KB pages. Six of
  its 12 cores sit in one grid column (x = 4) and three in another (x = 0), and a column of readers
  shares one route into the DRAM.
- So: use the bank-near assignment only when each core's traffic really is one bank (a DRAM-sharded
  input, or a strided interleaved read with stride = bank count). Never pair a core with a far bank.
  When traffic touches all banks, the bank-near set is worse than a row line.
