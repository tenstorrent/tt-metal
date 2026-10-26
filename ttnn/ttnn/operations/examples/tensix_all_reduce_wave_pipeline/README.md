# tensix_all_reduce_wave_pipeline — a reduce-scatter exchange between compute steps, one-shot vs pipelined in waves

**Problem type:** cross-core communication  ·  **Concept(s):** overlapping a cross-core exchange with the compute on either side of it (waves + a non-blocking poll loop)
**First profiled on:** `bh-50` · BH p150b · 1350 MHz · 2026-10-26 · `094f60e1352`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A group of Tensix cores each holds part of every tile-row. Each core reduces its part of a row, the group
combines the partials (reduce-scatter: the row's owner sums them, then all-gathers the total), and each core
applies the total back to its tiles. Written the obvious way, the block's compute finishes, then the whole
exchange runs, then the apply step starts: the NoC is idle during compute and compute is idle during the
exchange. When both take similar time, about half the kernel is waiting.

## What this isolates — and how
- **Concept:** the schedule of the exchange relative to compute — one exchange per block vs. one per wave of
  tile-rows, with every exchange step driven from a non-blocking poll loop.
- **Isolation setup:** Tensix ↔ Tensix NoC with real compute on both sides (the overlap IS the concept). Inputs
  and outputs are L1 shards (no DRAM); the arithmetic is eltwise adds (row partial = sum of the row's tiles,
  total = sum of the group's partials, y = x + total). Both variants run the same three kernels with the same
  arithmetic in the same order — their outputs are bit-identical — and differ only in the wave count.
- **Why it's kernel-level:** how the exchange is cut and sequenced against compute is the kernel author's choice.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `one_shot` *(baseline)* | All row partials → exchange the whole block → apply all rows. | — |
| `waves` | Rows split into `--waves` waves. Compute does all partials (wave order), then its owned sums, then applies wave by wave as totals land; the reader sends wave k while wave k+1 is summed. One semaphore pair per wave. | The exchange of early waves hides under the compute of later ones, on both sides of the exchange. Each extra wave costs a semaphore round, so tiny waves lose. |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.tensix_all_reduce_wave_pipeline [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--waves` | int… | `1 2 4 8` | wave counts compared against `one_shot` (max 8); counts the geometry cannot split evenly are skipped |
| `--rows` | int | *(default cases)* | tile-rows per Tensix core |
| `--width-tiles` | int | `4` | tiles per tile-row per Tensix core |
| `--group-size` | int | `8` | Tensix cores per group, along a grid row |
| `--num-groups` | int | `8` | groups, one per grid row |
| `--trials` | int | `1` | measured launches per method; raise to 3 only if a number looks implausible |
| `--kernel-iters` | int | `1` | all-reduces inside one launch — **1 = per-launch latency; large = steady state** |
| `--report` | path | *(print only)* | also write the Markdown report |

```bash
# your geometry: 16 tile-rows x 8 tiles per Tensix core, 1 / 2 / 4 waves
python -m ttnn.operations.examples.tensix_all_reduce_wave_pipeline --rows 16 --width-tiles 8 --waves 2 4
```

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box.* 8 groups × 8 Tensix cores =
64 Tensix cores, group = one grid row, N = 3 (median), kernel-iters = 1. Full table: `report.md`.

```
tensix_all_reduce_wave_pipeline   box=bh-50  arch=BH  clock=1350MHz  N=3  kernel-iters=1
  cores=64  tile-rows x tiles=32x4   method=one_shot  ...  22007 ns  ✓
  cores=64  tile-rows x tiles=32x4   method=waves=4   ...  16290 ns  ✓  → 1.35×
  cores=64  tile-rows x tiles=8x16   method=waves=4   ...  12031 ns  ✓  → 1.16× (one_shot 13897)
  cores=64  tile-rows x tiles=8x1    method=waves=4   ...   5972 ns  ✓  → 0.92× (one_shot 5484)
```

**Reading of the result:** with 128 tiles per Tensix core the exchange is a large fraction of the kernel, and
4 waves hide most of it (1.35×). The gain needs work inside each wave: at 2 tiles per wave the per-wave
semaphore round costs more than it hides, and at one tile-row per wave a single owner takes the whole wave's
incast.

## When it helps / when it does not
Measured by `test_tensix_all_reduce_wave_pipeline_regimes` (asserts the direction).

| Regime | Condition | Geometry per Tensix core | Baseline ns | Lever ns | Speedup | Trials | Board |
|---|---|---|---|---|---|---|---|
| helps | ≥ 16 tiles per wave | 32 tile-rows × 4 tiles, 4 waves | 22 007 | 16 290 | 1.35× | 3 | BH p150b |
| helps | ≥ 16 tiles per wave, few rows | 8 × 16, 4 waves | 13 897 | 12 031 | 1.16× | 3 | BH p150b |
| does not help | 2 tiles per wave | 8 × 1, 4 waves | 5 484 | 5 972 | 0.92× | 3 | BH p150b |
| does not help | 1 tile per wave | 8 × 1, 8 waves | 5 484 | 8 410 | 0.65× | 3 | BH p150b |
| does not help | 1 tile-row per wave | 8 × 16, 8 waves | 13 897 | 13 536 | 1.03× | 3 | BH p150b |

Helps: each wave's exchange travels under the next wave's compute. Stops paying when a wave carries too
little compute to cover its semaphore round, or so few rows that one owner serializes the wave. WH B0: not
measured.

## Run the predefined sweep (regenerates `report.md`)
```bash
WP_TRIALS=3 WP_REPORT=ttnn/ttnn/operations/examples/tensix_all_reduce_wave_pipeline/report.md \
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_wave_pipeline.py
```

## Code
`program_descriptor_with_inline_kernels.py`:
- `_EXCHANGE_KERNEL` (reader) — the poll loop. Four non-blocking steps in priority order: hand a wave's
  totals to compute (`try_push_result`), hand an owner's gathered partials to compute (`try_push_gather`), send
  a wave's partials to its owners (`try_send`), all-gather an owner's summed rows (`try_allgather`).
- `_COMPUTE_KERNEL` — the wave order: all partials, owned sums per wave, apply per wave.
- `wave_plan()` — tile-rows per wave and per owner; `one_shot` is the same kernels with one wave.
