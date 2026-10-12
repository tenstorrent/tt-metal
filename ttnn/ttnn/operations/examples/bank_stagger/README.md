# bank_stagger — bank-de-clustered issue order

**Problem type:** data movement below the ceiling: contention  ·  **Concept(s):** DRAM bank contention — per-core rotation of the issue order
**First profiled on:** `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` · Blackhole p300c · 11×10=110 grid · 8 DRAM banks · 2026-10-05

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
*Blackhole, 110 cores, one launch; median of 7 trials × 10 launches. Full table in
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
Board: **Blackhole p300c** (11×10 = 110 Tensix cores, 8 DRAM banks), bf16 tilize (`ROW_MAJOR` DRAM →
`TILE` interleaved DRAM), all 110 Tensix cores. Baseline = `none` (every core walks its blocks and
rows in order); lever = `stagger` (`stagger_blocks`, width-sharded source) or `rows`
(`stagger_rows`, interleaved source). Numbers are the measurements above (median of 7 trials × 10
launches, variants interleaved per trial; full table in [`report.md`](report.md)). There is no
`test_bank_stagger_regimes` yet — these are Blackhole numbers and have not been re-measured on
Wormhole B0.

| Condition | Shape (H×W) | Read size | Blocks per Tensix core | none (ns) | lever (ns) | Speedup | Trials | Board |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **Helps:** width-sharded source, few blocks per core (`stagger`) | `3520×1024` | 256 B | 8 | 101361 | 77087 | **1.32×** | 7 | BH p300c |
| **Helps:** width-sharded source, larger reads (`stagger`) | `3520×4096` | 1024 B | 8 | 184246 | 158186 | **1.16×** | 7 | BH p300c |
| **Gain shrinks:** width-sharded source, twice the blocks per core (`stagger`) | `7040×1024` / `7040×4096` | 256 B / 1024 B | 16 | 194525 / 357802 | 163880 / 316582 | **1.19× / 1.13×** | 7 | BH p300c |
| **Small:** interleaved source, one block per core (`rows`) | `32×7040`, `32×56320`, `3520×512` | 128–1024 B | 1 | 7447–19017 | 7157–17749 | **1.04–1.07×** | 7 | BH p300c |
| **Does not help:** interleaved source, four blocks per core (`rows`) | `128×56320` | 1024 B | 4 | 66462 | 66040 | **1.01×** | 7 | BH p300c |

- **Helps** when the source is width-sharded with one shard per bank and each Tensix core has few
  blocks: without the rotation the whole grid reads one bank per block step, and starting core k at
  block `k % n_w` spreads the same requests over all 8 banks.
- **Gain shrinks** as blocks per core grow: doubling 8 → 16 blocks cuts the speedup from 1.32× to
  1.19× (256 B) and 1.16× to 1.13× (1024 B); total time saved grows (24 → 31 µs, 26 → 41 µs) but
  less than the work does. Do not extrapolate the 8-block speedups.
- **Does not help** on an interleaved source once each Tensix core has several blocks: `rows` saves
  only ~0.3–1.3 µs whatever the work, so at four blocks per core it is 1.01×. These measurements do
  not establish why the saving stays flat.

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
