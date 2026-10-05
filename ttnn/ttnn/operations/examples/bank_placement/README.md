# bank_placement — which core serves which DRAM bank

**Difficulty:** ⭐ T1  ·  **Concept(s):** core placement relative to DRAM banks (core ↔ bank pairing)
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-114068` · WH B0 · 8×8=64 grid · 2026-10-05 · `3f32706f595`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
Sometimes each core's DRAM traffic all goes to one bank — a DRAM-sharded tensor, or an interleaved
tensor read with a stride equal to the bank count (page `p` is in bank `p % num_banks`). Then the
choice of *which core serves which bank* decides how far every byte travels on the NoC and which
routes the cores share. The device can report, per bank, the worker core it considers best placed
for that bank (`ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)`). This
example measures what that is worth, what getting it wrong costs, and when not to use it.

## What this isolates — and how
- **Concept:** the core ↔ work mapping; the work itself never changes.
- **Isolation setup:** NoC-contention row — DRAM interleaved in and out, one core per DRAM bank
  (12 on Wormhole), a plain copy (no compute), reads on NoC0, writes on NoC1, double-buffered CB,
  8 pages per barrier. Every variant issues the **same transactions with the same sizes and counts**;
  only which core runs which page list changes. Output is checked bit-exact.
- **Why it's kernel-level:** the core list and per-core runtime args are chosen by the program author.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `row_major` *(baseline)* | bank `b`'s work on core `b` of a row-major fill (rows 0–1) | — |
| `bank_near` | bank `b`'s work on the device's preferred core for bank `b` | shortest path to the bank, little route sharing |
| `bank_shuffled` | the **same cores** as `bank_near`, each serving the bank half-way round the list | isolates the pairing from the core set |

Two access patterns, set with `--pattern`:
- `affine` — core `b` copies bank `b`'s pages (stride = bank count): each core has a home bank.
- `spread` *(control)* — core `k` copies a contiguous run of pages: every core touches every bank.

## CLI — measure your own sizes
```bash
python -m ttnn.operations.examples.bank_placement [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `all` or comma list of `row_major,bank_near,bank_shuffled` | `all` | which placement(s) to run |
| `--pattern` | `all`, `affine`, `spread` | `all` | access pattern |
| `--cases` | comma list of `ROWSxWIDTH` | `3072x256,3072x1024,3072x4096` | bf16 `[ROWS, WIDTH]`; one row = one page of `WIDTH × 2` B; `ROWS` a multiple of the bank count |
| `--block` | int | `8` | pages per NoC barrier |
| `--trials` | int | `5` | trials per variant; median reported. Variants interleaved per trial |
| `--launches` | int | `10` | launches averaged inside one trial |
| `--kernel-iters` | int | `1` | in-kernel repeat — 1 = per-launch latency |
| `--report` | path | *(print only)* | also write the table to a file |

```bash
python -m ttnn.operations.examples.bank_placement --pattern affine --cases 3072x4096
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; full tables in [`report.md`](report.md).*

```
bank_placement   arch=WORMHOLE_B0  grid=8x8  12 cores (one per bank)  N=7 trials x 10 launches (median)
  pattern  page B  row_major ns  bank_near   bank_shuffled
  affine     512        31251     1.093x       0.988x
  affine    2048        72422     1.007x       0.616x
  affine    8192       256501     1.054x       0.460x
  spread     512        28432     0.981x       0.982x
  spread    2048        64382     0.659x       0.661x
  spread    8192       281984     0.605x       0.604x
```
(ratios are vs `row_major`; > 1 is faster)

**Reading of the result:**
- **The pairing matters a lot when each core has a home bank.** With the same 12 cores, serving the
  near bank instead of a far one is **1.6× faster at 2 KB pages and 2.3× at 8 KB**. A far pairing sends
  every core's traffic across the grid, and the routes overlap.
- **A row-major line is already close to the best.** `bank_near` beats it by only 1.01–1.09×.
- **The bank-near cores are worse for all-bank traffic.** In `spread`, the bank-near set runs at
  0.60–0.66× of row-major: six of its cores sit in one grid column, and a column of readers shares one
  route into DRAM.
- At 512 B pages each core is limited by how fast it issues reads, so placement barely matters.

**When to use it:** each core's traffic goes to one bank (DRAM-sharded inputs, stride = bank count) —
then take the device's bank-near assignment, and above all never pair a core with a far bank.
**When not:** each core walks all banks (contiguous interleaved runs, tilize-style 32-row blocks) —
there is no home bank, and the bank-near core set is slower than a row line.

## Run the predefined sweep
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_bank_placement.py
```

## Code
- `bank_placement.py` — `placement_cores` (the three placements) and `work_ranges` (the two patterns).
- `kernels/bp_reader.cpp`, `kernels/bp_writer.cpp` — the strided page copy; identical for every placement.
