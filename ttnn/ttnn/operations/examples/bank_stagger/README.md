# bank_stagger — bank-de-clustered issue order

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** DRAM bank contention — per-core rotation of the block and read issue order
**First profiled on:** `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` · Blackhole p300c · 11×10=110 grid · 8 DRAM banks · 2026-10-05

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
DRAM is split into banks (8 on Blackhole, 12 on Wormhole). If at each read step every core asks for
data in the same bank, that bank is busy while the others wait. Three ways of splitting the work
cause this.

**Case 1 — width-sharded source (any arch).** With one shard per bank, each column block of the
tensor *is* one bank. Every core walks its column blocks in the same order, so the whole grid reads
bank 0 for a whole block, then bank 1, and so on — one bank busy out of 8 for the entire op.

```
block step       0        1        2    ...
core 0 reads   shard 0  shard 1  shard 2
core 1 reads   shard 0  shard 1  shard 2
bank             0        1        2         <- the whole grid on one bank per block
```

**Case 2 — width blocking, interleaved source (any arch).** Interleaved DRAM puts row `r` in bank
`r % num_banks`. Cores get the same 32 rows at different width offsets, so they read the same row,
on the same bank, at each step.

**Case 3 — height blocking, interleaved source (Blackhole only).** Each core has its own 32 rows
starting at `32·k`. With 8 banks `32·k % 8 == 0`, so every core reads a different row on the same
bank at each step. With 12 banks (Wormhole) the starts already spread over 3 banks.

```
step             0       1       2    ...
core 0 reads   row 0   row 1   row 2        case 2: same rows, different columns
core 1 reads   row 32  row 33  row 34       case 3: different rows
bank             0       1       2          <- same bank for every core
```

## The fix — compile-time switches
Each core starts at a different index and wraps around. Same transactions, sizes and L1 addresses;
only the order changes. Off compiles the plain in-order loop, so you can flip each switch per call
and check whether it pays on your shape.

| Switch | Fixes | On: issue order |
|---|---|---|
| `stagger_blocks` | case 1 | reader and writer walk the core's blocks from `core % n_w` |
| `stagger_reads` | cases 2, 3 | reader issues row `(i + core % 32) % 32` inside a block |

```python
from ttnn.operations.examples.bank_stagger import bank_stagger
out = bank_stagger(x, stagger_blocks=True, stagger_reads=True)
```

The op is a real tilize (`ROW_MAJOR` DRAM → `TILE` interleaved DRAM), one 32-row × `chunk`-tile
block per work unit. A third switch, `stagger_writes`, rotates the writer's tile order; on Blackhole
it measures on par (within the trial spread), so it is not shown — keep it off unless your own
measurement says otherwise.

## Measured result
*Blackhole, 110 cores, one launch; median of 7 trials × 10 launches. Full tables in
[`report.md`](report.md).*

**Case 1 — `stagger_blocks`, width-sharded source (8 shards):**

```
  shape       read B  blocks/core   none ns  blocks ns  speedup   saved
  3520x1024      256       8         100789      76722   1.31x     24 µs
  7040x1024      256      16         193737     162812   1.19x     31 µs
  3520x4096     1024       8         183685     157840   1.16x     26 µs
  7040x4096     1024      16         358202     316906   1.13x     41 µs
```

**Cases 2 and 3 — `stagger_reads`, interleaved source, one block per core:**

```
  shape     read B  case      none ns   read ns  speedup   saved
  32x7040      128  width        7468      7192   1.04x     0.3 µs
  32x56320    1024  width       18958     17768   1.07x     1.2 µs
  3520x512    1024  height      18571     17566   1.06x     1.0 µs
```

- **Case 1 is the big win.** The pile-up lasts for whole blocks, so the saving grows with the work
  (24 → 36 µs at 256 B from 8 to 32 blocks per core); the ratio shrinks slowly as the run gets longer.
- **Cases 2 and 3 are a start-of-launch win.** A core's 32 in-flight rows already cover all banks;
  only the first wave is clustered, so the saving is a fixed 0.3–1.2 µs per launch.

**When to use it:** `stagger_blocks` whenever cores walk several column blocks of a width-sharded
DRAM source; `stagger_reads` for width or height blocking in short ops with one or a few blocks per
core. Each costs one compile-time flag and one runtime arg.

## Run it
```bash
# the default shapes, sized to the grid, all variants
python -m ttnn.operations.examples.bank_stagger

# your shape: HxWxCHUNK (interleaved) or HxWxCHUNKws (width-sharded, one shard per bank)
python -m ttnn.operations.examples.bank_stagger --cases 7040x1024x4ws --variant none,blocks
```

Other flags: `--trials` (default 5, median reported), `--launches` (10 per trial), `--kernel-iters K`
(repeat the work K times inside one launch), `--report PATH`.

```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py
```
