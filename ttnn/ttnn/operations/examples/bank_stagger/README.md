# bank_stagger — bank-de-clustered issue order

**Problem type:** data movement below the ceiling: contention  ·  **Concept(s):** DRAM bank contention — per-core rotation of the issue order
**First profiled on:** `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` · Blackhole p300c · 11×10=110 grid · 8 DRAM banks · 2026-10-05
**Measured on:** BH p300c (`bh-qbge-09-special-dstoiljkovic-for-reservation-117054` / `…-131852`, 2026-10-05 / 2026-10-13) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19) · WH B0 n150 not measured

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
DRAM is split into banks (8 on Blackhole, 12 on Wormhole). If every core walks its data in the same
order and that data lines up on the banks, the whole grid reads one bank at a time while the others
wait. Across the op the load is spread evenly over the banks; at any one moment it sits on one.

The example uses a width-sharded DRAM source with one shard per bank, so each column block *is* one
bank. Every core walks its column blocks in the same order:

```
block step       0        1        2    ...
core 0 reads   shard 0  shard 1  shard 2
core 1 reads   shard 0  shard 1  shard 2
bank             0        1        2         <- the whole grid on one bank per block
```

## The fix — one compile-time switch
Each core starts its walk at a different block (`core % n_w`) and wraps around, so at every step the
cores are spread over all banks. Same transactions, sizes and L1 addresses; only the order changes.
`stagger_blocks` is a compile-time flag in the reader and writer — off compiles the plain in-order
loop, so you can flip it per call and check whether it pays on your shape.

```python
from ttnn.operations.examples.bank_stagger import bank_stagger
out = bank_stagger(x, stagger_blocks=True, chunk_wt=4)
```

The op is a real tilize (`ROW_MAJOR` width-sharded DRAM → `TILE` interleaved DRAM), work units of
32 rows × `chunk` tiles, one or two tile-rows (8 or 16 blocks) per core.

## Measured result
*Blackhole p300c, 110 Tensix cores, one launch; median of 7 trials × 10 launches (BH p150b reproduces it, see below). Full table in
[`report.md`](report.md).*

```
  shape       read B  blocks/core   none ns  stagger ns  speedup   saved
  3520x1024      256       8         101361      77087    1.32x     24 µs
  7040x1024      256      16         194525     163880    1.19x     31 µs
  3520x4096     1024       8         184246     158186    1.16x     26 µs
  7040x4096     1024      16         357802     316582    1.13x     41 µs
```

The **relative gain decreases as blocks per core increase** in these measurements: doubling from
8 to 16 blocks reduces speedup from 1.32× to 1.19× at 256 B reads, and from 1.16× to 1.13× at
1024 B reads. Total time saved increases (24 → 31 µs and 26 → 41 µs), but the workload doubles,
so savings per unit of work decrease. The benefit is stronger for the shorter workloads tested;
do not extrapolate the 8-block speedups to workloads with many blocks per core. These measurements
do not establish why the relative gain weakens or whether it eventually disappears.

## Interleaved source — row stagger (`stagger_rows`)
Interleaved DRAM puts row `r` in bank `r % num_banks`. Two splits start every core on the same bank:

- **Width blocking:** cores read the same 32 rows at different width offsets.
- **Height blocking on Blackhole:** each core's rows start at `32·k`, and `32·k % 8 == 0`.

`stagger_rows` starts core k at row `k % 32` inside each block and wraps (`variant rows`).

```
  shape       split   read B  blocks/core  none ns  rows ns  speedup  saved
  32x7040     width     128       1          7447     7157   1.04x    0.3 µs
  32x56320    width    1024       1         19017    17749   1.07x    1.3 µs
  3520x512    height   1024       1         18534    17589   1.05x    0.9 µs
  128x56320   width    1024       4         66462    66040   1.01x    0.4 µs
```

The gain shrinks as the work per core grows: the time saved stays roughly flat (~0.3–1.3 µs)
while the total time grows. These measurements do not establish why. Full table in
[`report.md`](report.md).

## Where else it applies
The same fix — start each core at a different index — applies wherever the cores' data lines up on
the banks and they walk it in the same order:

- **Writes** into a width-sharded DRAM output, or of consecutive tile pages whose count is a multiple
  of the bank count: every core starts writing on the same bank. Rotate the write order.

## When it helps / when it does not
**Rule:** rotate each Tensix core's block order (`stagger_blocks`) when the source is width-sharded with one shard
per DRAM bank and every Tensix core walks the blocks in the same order: **~1.3× at 256 B reads and ~1.2× at
1024 B with 8 blocks per Tensix core**, shrinking as blocks per Tensix core grow (~1.1–1.2× at 16). On an
interleaved source the row rotation (`stagger_rows`) is worth only 1.04–1.07× with one block per Tensix core
and ~1.0× with several. Measured on two Blackhole boards with the same verdicts; **not measured on WH B0 n150**
— bank conflicts are an 8-DRAM-bank effect here, and WH has 12 DRAM banks. bf16 tilize (`ROW_MAJOR` DRAM →
`TILE` interleaved DRAM), full grid; baseline = `none`, lever = `stagger` or `rows`. Each board's shapes give
one or two tile-rows per Tensix core on its grid (110 vs 120 tile-rows).

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p300c · 11×10 = 110 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|---|
| **Helps:** width-sharded source, 256 B reads, 8 blocks per Tensix core (`stagger`) | not measured | `3520×1024`: 101 361 → 77 087 ns **1.32×** | `3840×1024`: 111 677 → 84 255 ns **1.33×** |
| **Helps:** width-sharded source, 1024 B reads, 8 blocks per Tensix core (`stagger`) | not measured | `3520×4096`: 184 246 → 158 186 ns **1.16×** | `3840×4096`: 203 545 → 171 495 ns **1.19×** |
| **Gain shrinks:** width-sharded, 256 B reads, 16 blocks per Tensix core (`stagger`) | not measured | `7040×1024`: 194 525 → 163 880 ns **1.19×** | `7680×1024`: 208 627 → 175 265 ns **1.19×** |
| **Gain shrinks:** width-sharded, 1024 B reads, 16 blocks per Tensix core (`stagger`) | not measured | `7040×4096`: 357 802 → 316 582 ns **1.13×** | `7680×4096`: 385 965 → 345 965 ns **1.12×** |
| **Does not help:** width-sharded source, row rotation (`rows`) | not measured | not measured | all 4 shapes above: **0.99–1.01×** |
| **Small:** interleaved source, width blocking, 128 B reads, 1 block per Tensix core (`rows`) | not measured | `32×7040`: 7 447 → 7 157 ns **1.04×** | `32×7680`: 8 079 → 7 773 ns **1.04×** |
| **Small:** interleaved source, width blocking, 1024 B reads, 1 block (`rows`) | not measured | `32×56320`: 19 017 → 17 749 ns **1.07×** | `32×61440`: 21 134 → 19 998 ns **1.06×** |
| **Small:** interleaved source, height blocking, 1024 B reads, 1 block (`rows`) | not measured | `3520×512`: 18 534 → 17 589 ns **1.05×** | `3840×512`: 20 194 → 19 377 ns **1.04×** |
| **Does not help:** interleaved source, 4 blocks per Tensix core (`rows`) | not measured | `128×56320`: 66 462 → 66 040 ns **1.01×** | not measured |
| **Does not help:** interleaved source, block rotation (`stagger`) | not measured | not measured | the 3 shapes above: **0.99–1.00×** |
| **Boundary** | — | width-sharded: still 1.13× at 16 blocks per Tensix core; interleaved: 1.01× at 4 blocks | width-sharded: still 1.12× at 16 blocks per Tensix core |

**Why the boards differ:** the two Blackhole boards agree within ~0.03× on every shared row — both have 8 DRAM
banks, and 110 vs 120 Tensix cores only changes how many cores pile onto a bank. WH B0 n150 is untested: with 12
DRAM banks the width-sharded case becomes 12 shards and the height-blocking alignment argument (`32·k % 8 == 0`)
no longer holds as stated, so do not assume the Blackhole ratios there.

- **Helps** when the source is width-sharded with one shard per DRAM bank and each Tensix core has few blocks:
  without the rotation the whole grid reads one DRAM bank per block step, and starting Tensix core k at block
  `k % n_w` spreads the same requests over all 8 DRAM banks.
- **Gain shrinks** as blocks per Tensix core grow: doubling 8 → 16 blocks cuts the speedup from 1.32× to 1.19×
  (256 B) and 1.16× to 1.13× (1024 B) on BH p300c, 1.33× → 1.19× and 1.19× → 1.12× on BH p150b; total time saved
  grows (p300c: 24 → 31 µs, 26 → 41 µs) but less than the work does. Do not extrapolate the 8-block speedups.
- **Does not help** on an interleaved source once each Tensix core has several blocks: `rows` saves only
  ~0.3–1.3 µs whatever the work, so at four blocks per Tensix core it is 1.01×. These measurements do not
  establish why the saving stays flat.

*How measured:* `DEVICE KERNEL DURATION [ns]`, variants interleaved per trial, median of trials; every variant
checked bit-exact first. There is no `test_bank_stagger_regimes` yet.
- **BH p300c:** `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` (width-sharded, 2026-10-05) and
  `…-131852` (interleaved, 2026-10-13), 7 trials × 10 launches ([`report.md`](report.md)).
- **BH p150b:** `test_bank_stagger_device_perf`, 5 trials × 10 launches, `kernel-iters=1`,
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.
- **WH B0 n150:** not measured.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py::test_bank_stagger_device_perf`

## Run it
```bash
# the four shapes above, sized to the grid
python -m ttnn.operations.examples.bank_stagger

# your shape: HxWxCHUNK, split into one shard per DRAM bank
python -m ttnn.operations.examples.bank_stagger --cases 14080x1024x4

# interleaved source: add the il suffix
python -m ttnn.operations.examples.bank_stagger --variant none,rows --cases 32x56320x16il,3520x512x16il
```

Other flags: `--variant` (`none`, `stagger`, `rows`), `--trials` (default 5, median reported), `--launches`
(10 per trial), `--kernel-iters K` (repeat the work K times inside one launch), `--report PATH`.

```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py
```
