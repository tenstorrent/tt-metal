# double_buffer — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` |
| arch | Wormhole B0 (8×8 compute grid, DRAM grid 12×1) |
| commit | `bf5ebe7aacb` |
| date | 2026-07-09 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 5 warmup + 20 timed launches per case, flush-bracketed, on-device duration averaged; whole matrix in one device session |
| GB/s | read+write DRAM traffic (`2 × tiles × tile_bytes`) ÷ device kernel time; `bytes/ns == GB/s`. `tile_bytes` is read back from the tensor (`buffer_aligned_page_size`), never hard-coded |

> Numbers are illustrative of the *effect*, not a CI bound — single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.double_buffer` to measure your own params.
> A different arch (e.g. Blackhole) should be appended as a new block, not overwritten.

The theme is one thing: **how many bytes you keep in flight on the NoC** decides whether the
pipeline is latency-bound (NoC mostly idle) or bandwidth-bound (NoC saturated). Three kernel-level
levers move bytes-in-flight, all swept here with the eltwise compute held light (one relu):
- **`block`** — async reads/writes issued per barrier (more transactions in flight). `block=1` = trap.
- **`variant`** — CB depth: `single_buffered` (1×block) vs `double_buffered` (2×block); double lets the
  reader run ahead so the read (NoC0) and write (NoC1) streams overlap.
- **transfer size** — bytes per tile, set by dtype: bfloat8_b 1088 B, bfloat16 2048 B, float32 4096 B
  (more bytes per transaction). Kernels are byte-identical; only CT args + CB sizes change.
`vs trap` is relative to `block=1, single_buffered` at that dtype.

## Config A — 1 core, `shape=(512,512)`, **bfloat16** (256 tiles, 2048 B/tile, 1.05 MB traffic) — the block × depth curve

```
    block  variant             ns/op     GB/s   vs trap
        1  single_buffered   162383.5      6.5   (trap)
        1  double_buffered   131362.6      8.0   1.24x
        2  single_buffered   110781.6      9.5   1.47x
        2  double_buffered    81445.6     12.9   1.99x
        4  single_buffered    93567.5     11.2   1.74x
        4  double_buffered    58512.8     17.9   2.78x   <- best
        8  single_buffered    84773.6     12.4   1.92x
        8  double_buffered    59048.9     17.8   2.75x
       16  single_buffered    81337.6     12.9   2.00x
       16  double_buffered    60138.7     17.4   2.70x
       32  single_buffered    80793.8     13.0   2.01x
       32  double_buffered    62310.9     16.8   2.61x
```

## Config B — transfer size, 1 core, `shape=(512,512)` — bigger transactions hit higher bandwidth

**bfloat8_b** (256 tiles, 1088 B/tile, 0.56 MB traffic):
```
    block  variant             ns/op     GB/s   vs trap
        1  single_buffered   141666.8      3.9   (trap)
        1  double_buffered   121852.3      4.6   1.16x
        4  single_buffered    85090.4      6.5   1.66x
        4  double_buffered    56866.9      9.8   2.49x   <- best
        8  double_buffered    57191.7      9.7   2.48x
       32  single_buffered    70506.8      7.9   2.01x
       32  double_buffered    59221.0      9.4   2.39x
```

**float32** (256 tiles, 4096 B/tile, 2.10 MB traffic):
```
    block  variant             ns/op     GB/s   vs trap
        1  single_buffered   190214.4     11.0   (trap)
        1  double_buffered   162457.5     12.9   1.17x
        4  single_buffered   118128.5     17.8   1.61x
        4  double_buffered    82737.4     25.3   2.30x
        8  single_buffered   110800.5     18.9   1.72x
        8  double_buffered    66231.7     31.7   2.87x   <- best
       16  double_buffered    68405.8     30.7   2.78x
       32  double_buffered    72717.3     28.8   2.62x
```

Best achieved bandwidth scales with the transaction size: **9.8 (bfp8) → 17.9 (bf16) → 31.7 (fp32)
GB/s** — roughly linear in `tile_bytes`, because a larger transfer amortizes fixed per-transaction
NoC/DRAM latency. But best *wall time* goes the other way (56.9 → 58.5 → 66.2 µs): fp32 moves 2× the
bytes of bf16 for the same tiles, so it is slower despite the higher bandwidth. The fastest bytes are
the ones you don't move — a smaller dtype wins on latency, a bigger one on per-transaction efficiency.

## Config C — 64 cores, `shape=(2048,2048)`, bfloat16 (4096 tiles, 16.78 MB traffic) — bandwidth-bound

```
    block  variant             ns/op     GB/s   vs trap
        1  single_buffered    87915.8    190.8   (trap)
        1  double_buffered    88400.2    189.8   0.99x
        4  single_buffered    91532.0    183.3   0.96x
        4  double_buffered    90117.1    186.2   0.98x
        8  double_buffered    90778.7    184.8   0.97x
       32  single_buffered    95665.2    175.4   0.92x
       32  double_buffered    91686.1    183.0   0.96x
```

## Config D — extended block sweep, 1 core, `shape=(1024,1024)` (1024 tiles), double_buffered — the plateau

Batching past ~4 outstanding reads does **not** keep helping; each dtype hits a flat ceiling set by
its transfer size (block=256 double-buffered OOMs L1, so that is also the practical cap on `block`):

```
    block   bfloat8_b (1088 B)   bfloat16 (2048 B)   float32 (4096 B)
        4        10.0 GB/s            18.3 GB/s          25.8 GB/s
        8        10.0 GB/s            18.3 GB/s          32.9 GB/s   <- fp32 ceiling
       16        10.0 GB/s            18.2 GB/s          32.7 GB/s
       32         9.9 GB/s            18.0 GB/s          32.1 GB/s
       64         9.8 GB/s            17.7 GB/s            —  (L1)
```

Dividing GB/s by tile bytes gives a **near-constant ~8–9 M tile-transactions/s** at the ceiling
(bfp8 9.2 M/s, bf16 8.9 M/s, fp32 8.0 M/s ≈ 110–125 ns per completed transaction). So a single core
is **transaction-rate-limited**, and **achieved GB/s ≈ transaction-rate × bytes-per-transaction**.
bf16 caps near **18 GB/s** and bfp8 near **10** no matter how big the block; only fp32's 4096 B tile
reaches ~**33 GB/s**. To push bf16 to ~32 GB/s on one core you must make each NoC transaction bigger
(coalesce multiple bank-contiguous tiles per read), not issue more of them.

## Findings
- **`block=1` (one read, then a barrier) is the trap** — it pins the pipeline to DRAM *latency*.
  On one core, bf16: 6.5 GB/s, 162 µs — the worst cell in the matrix.
- **Batching reads helps only up to ~4 outstanding, then plateaus.** Verified flat from `block=4` to
  `block=64` (bf16 double: 18.3 → 17.7 GB/s). Past ~4 you are transaction-rate-bound, not
  latency-bound, so more outstanding reads buy nothing; a bigger block only spends more L1 and slightly
  coarsens the hand-off. Sweet spot ~4–8.
- **Double buffering compounds and raises the ceiling.** Best cell is `block=4, double_buffered`:
  **17.9 GB/s, 2.78× vs the trap** (bf16) — the single-core NoC limit for a 2 KB transaction.
- **The single-core ceiling is set by TRANSFER SIZE, not block count.** ~8–9 M tile-transactions/s
  regardless of size → GB/s scales with bytes/transaction: bfp8 ~10, bf16 ~18, fp32 ~33 GB/s. Same
  kernel; only the tile bytes differ. A smaller dtype moves less data (wins wall time); a bigger dtype
  moves more per transaction (wins GB/s). Higher bf16 single-core BW needs coalesced (multi-tile)
  transactions, not more/deeper buffers.
- **When DRAM-bandwidth-bound, none of it matters.** At 64 cores even the `block=1` single-buffered
  trap already hits **190.8 GB/s** (≈ this part's DRAM peak); every richer setting is within noise or
  a hair slower. Enough cores saturate DRAM on their own — the pipeline structure is no longer the
  bottleneck, and the only lever left is moving less data.
- **Scope / caveat:** compute is held light. If compute is the bottleneck, the compute engine's
  independent unpack/math/pack threads already hide the DRAM traffic at any block/depth, so these
  levers are moot — they pay off only while data movement is what stalls the pipeline.
