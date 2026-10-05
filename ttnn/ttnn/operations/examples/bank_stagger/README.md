# bank_stagger — bank-de-clustered issue order

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** DRAM bank contention — per-core rotation of the read and write issue order
**First profiled on:** `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` · Blackhole p300c · 11×10=110 grid · 8 DRAM banks · 2026-10-05

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
Interleaved DRAM puts page `p` in bank `p % num_banks` (8 on Blackhole, 12 on Wormhole); for a
`ROW_MAJOR` tensor one page is one row. If at each read step every core asks for a page on the same
bank, that bank is busy while the others wait. Two ways of splitting the work cause this.

**Case 1 — width blocking (any arch).** Cores get the same 32 rows at different width offsets, so
they read the same page at each step.

```
step             0       1       2    ...
core 0 reads   row 0   row 1   row 2        cols 0..chunk-1
core 1 reads   row 0   row 1   row 2        cols chunk..2*chunk-1
bank             0       1       2          <- same page, same bank for every core
```

**Case 2 — height blocking (Blackhole only).** Each core has its own 32 rows starting at `32·k`.
With 8 banks `32·k % 8 == 0`, so every core reads a different page on the same bank. With 12 banks
(Wormhole) the starts fall on banks 0, 4 and 8, so this case does not occur.

```
step             0       1       2    ...
core 0 reads   row 0   row 1   row 2
core 1 reads   row 32  row 33  row 34
bank             0       1       2          <- different pages, same bank for every core
```

## The fix — two compile-time switches
Each core starts at a different index and wraps around. Same transactions, sizes and L1 addresses;
only the order changes. Off compiles the plain in-order loop, so you can flip each switch per call
and check whether it pays on your shape.

| Switch | Kernel | On: issue order |
|---|---|---|
| `stagger_reads` | `bs_reader.cpp` | row `(i + core % 32) % 32` |
| `stagger_writes` | `bs_writer.cpp` | tile `(i + core % chunk) % chunk` |

```python
from ttnn.operations.examples.bank_stagger import bank_stagger
out = bank_stagger(x, stagger_reads=True, stagger_writes=False, chunk_wt=8)
```

The sweep compares `none`, `read`, `write` and `both`. The op is a real tilize (`ROW_MAJOR` →
`TILE`, interleaved DRAM in and out), one 32-row × `chunk`-tile block per core.

## Measured result
*Blackhole, 110 cores, one block per core; median of 21 trials × 10 launches. Full table in
[`report.md`](report.md).*

```
  shape     read B  case      none ns   read ns  speedup   saved/launch
  32x7040      128  width        7456      7174   1.039x        0.3 µs
  32x56320    1024  width       19066     17669   1.079x        1.4 µs
  3520x512    1024  height      18580     17593   1.056x        1.0 µs
```

The read switch is 1.06–1.08× faster at 1024 B reads and 1.04× at 128 B, in both cases. On
Blackhole the write switch measures on par (within the trial spread), so it is not shown; keep it off
unless your own measurement says otherwise.

**When to use it:** width blocking (any arch) or height blocking (Blackhole), in short ops with one
or a few blocks per core — the win comes from the start of the launch, when all cores read together.
It costs one compile-time flag and one runtime arg.

## Run it
```bash
# the three shapes above, sized to the grid, all four variants
python -m ttnn.operations.examples.bank_stagger

# your shape (HxWxCHUNK), baseline vs read switch
python -m ttnn.operations.examples.bank_stagger --cases 32x28160x8 --variant none,read
```

Other flags: `--trials` (default 5, median reported), `--launches` (10 per trial), `--kernel-iters K`
(repeat the work K times inside one launch), `--report PATH`.

```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py
```
