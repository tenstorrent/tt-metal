# compute_block_size — how big a block each compute helper should chew per call

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** the fixed per-helper-call overhead of a multi-phase
compute chain (phase-boundary data-format reconfig + LLK init/uninit + unpack/math/pack pipeline
fill and drain), attacked two ways — (1) **block granularity**: amortize it over more tiles per
call; (2) **skip redundant reconfigs**: when the data format never changes, the per-phase reconfig
is pure waste, so turn it off (keeping the inits).
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · 1000 MHz · 2026-07-13 · `d2cf2381bea`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You are computing `out = (A + B) @ C`, where `A`/`B` arrive **row-major** and the output must leave
**row-major**, so the compute kernel has five phases: tilize `A`, tilize `B`, add, matmul, untilize.
Every one of those phases is parallel over the M (row) dimension — so you get to choose *how much*
of M to push through the whole chain at once. The obvious, readable way is to loop over M a tile-row
at a time (`for row: tilize; tilize; add; matmul; untilize`), and it is correct. But each helper call
you make pays a fixed toll — it reconfigures the unpacker/packer data formats for its phase, runs the
LLK init/uninit, and fills then drains the three-stage compute pipeline. A tile-row-by-tile-row loop
pays that toll on **every** tile-row; doing the whole thing in one pass pays it **once**.

## What this isolates — and how
- **Concept:** the fixed per-call overhead of a compute helper (reconfig + init/uninit + pipeline
  fill/drain), and how processing a bigger block per call amortizes it. The **total work is identical
  across every variant** — same tiles tilized, same adds, same matmul MACs, same untilize — only the
  block granularity changes.
- **Isolation setup (pure compute):** everything lives in **sharded L1 on one Tensix core**. `A`, `B`,
  `C` are resident before the kernel runs and the output stays in L1 — there is **no DRAM movement**,
  and the kernel loops `--kernel-iters` times over the resident data for a steady-state number. So the
  measured delta is purely the on-core compute pipeline; nothing is waiting on data.
- **Why it's kernel-level:** how finely to chop a row-parallel chain into per-call blocks — and how
  large to size the intermediate CBs that hold each block — is entirely the kernel author's decision.
  It is not a model or dtype choice; the dtype (bf16), shape, math fidelity (HiFi2 + fp32 DEST accum),
  and matmul subblock shape are all held fixed.

Each variant runs the *same math* at a different block granularity, so a "win" means identical work,
fewer ns.

## The methods being compared
The knob is `block_rows` — how many tile-rows of M the whole five-phase chain processes per pass
(`num_blocks = M_tiles / block_rows`). All variants use one identical kernel; `block_rows` is a
compile-time arg.

| Variant | What it does | Why it should differ |
|---|---|---|
| `per_tile_row` *(baseline)* | `block_rows = 1` — one tile-row through all five phases, `M_tiles` times | pays every phase's fixed per-call overhead `M_tiles` times |
| `block2` | `block_rows = 2` | half as many passes → half the per-call overhead |
| `block4` | `block_rows = 4` | a quarter of the passes |
| `one_block` | `block_rows = M_tiles` — the whole M in a single pass through each phase | pays each phase's overhead exactly once |

(A variant is skipped when its block height does not evenly divide `M_tiles`.)

## CLI — measure your own shapes/params
This example is runnable on **any** shape you care about, not just the predefined sweep:

```bash
python -m ttnn.operations.examples.compute_block_size [options]
```

**Common flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,per_tile_row,block2,block4,one_block}` | `all` | which block granularities to run/compare |
| `--trials` | int | `5` | measured trials; report shows median ± std |
| `--kernel-iters` | int | `100` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | also write the report table to a file |

**Example-specific flags (the problem shape, in tiles of 32):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--m-tiles` | int | `8` | M height in tiles (the row axis being blocked) |
| `--k-tiles` | int | `4` | K inner dim in tiles |
| `--n-tiles` | int | `4` | N width in tiles |

```bash
# the full block-size spectrum on the default shape, steady-state
python -m ttnn.operations.examples.compute_block_size --kernel-iters 100

# just the two extremes, on a taller / wider problem
python -m ttnn.operations.examples.compute_block_size --variant per_tile_row one_block \
    --m-tiles 16 --k-tiles 4 --n-tiles 8
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full sweep in [`report.md`](report.md).*

```
compute_block_size  box=bgd-lab-...-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=100 (steady-state)
problem: M=256 K=128 N=128  (M_tiles=8, K_tiles=4, N_tiles=4)  dtype=bf16  HiFi2 fp32_dest_acc
  block_rows=1  num_blocks=8  per_tile_row  28650.9 ns ±0.0%  ✓            (baseline)
  block_rows=2  num_blocks=4  block2        22487.7 ns ±0.0%  ✓  → 1.27×
  block_rows=4  num_blocks=2  block4        19009.1 ns ±0.0%  ✓  → 1.51×
  block_rows=8  num_blocks=1  one_block     17397.6 ns ±0.0%  ✓  → 1.65×
```

**Reading of the result:** doing the whole chain in one pass is **1.65×** faster than looping a
tile-row at a time, with the identical math (PCC 0.99999 for every variant). The curve is monotonic
with **diminishing returns** (+0.27×, then +0.24×, then +0.14× as the block doubles) — the signature
of amortizing a *fixed* per-pass cost: the first doublings remove the most passes. From the numbers,
each extra pass costs ≈ 1.6 µs of pure overhead (≈ 320 ns per phase across the five phases) that buys
no additional work.

Where it applies and where it doesn't:
- **The win grows with the number of phases and shrinks with the per-block payload.** This chain has
  five phase transitions, each with its own reconfig + init — lots of fixed cost to amortize. On a
  wider problem the same sweep gives a smaller win (e.g. `N=256`/`N_tiles=8`: `per_tile_row` → `one_block`
  is **1.40×**, not 1.65×), because each matmul/untilize call now does more real work, so the fixed
  overhead is a smaller fraction of it.
- **It costs L1.** Bigger blocks need bigger intermediate CBs — the tilized/added/matmul-output
  scratch scales with `block_rows`. `one_block` holds the full M worth of intermediates; `per_tile_row`
  holds one tile-row. On a single core here that is a non-issue, but on an L1-tight kernel the block
  size is a speed/space trade, not a free win.
- **No effect if there is only one block anyway** (a problem already small enough in M that
  `M_tiles == block_rows`), or if the chain is a single phase (nothing to reconfigure between).

## Second lever — skip the redundant data-format reconfigs
Part of that fixed per-phase overhead is the **data-format reconfig** each helper issues at its
boundary (`reconfig_data_format` on the unpacker / `pack_reconfig_data_format` on the packer). The
helpers do it by default because *in general* consecutive phases may read/write different formats.
But in this op **every CB is bf16** — the format never changes through the whole chain — so that
reconfig is pure wasted MMIO. We can turn it off (keeping the **inits**: each phase is still a
different op and must be init'd) and measure the cost directly.

*Illustrative — full sweep in [`report_reconfig_ablation.md`](report_reconfig_ablation.md).*

```
reconfig ON vs OFF   box=bgd-lab-...-40918  arch=WH_B0  cores=1  N=5 (median)  kernel-iters=100
  block_rows=1  num_blocks=8  per_tile_row   28592 →  24017 ns   → 1.19×   (PCC 0.99999 / 0.99999)
  block_rows=2  num_blocks=4  block2         22522 →  19420 ns   → 1.16×   (PCC 0.99999 / 0.99999)
  block_rows=4  num_blocks=2  block4         18985 →  17456 ns   → 1.09×   (PCC 0.99999 / 0.99999)
  block_rows=8  num_blocks=1  one_block      17386 →  16642 ns   → 1.04×   (PCC 0.99999 / 0.99999)
```

**Reading of the result:** turning off the reconfigs is **correct** (PCC unchanged — the format
really was constant) and **faster**, by up to **1.19×**. Crucially the win is *largest where there
are the most phase transitions* (`per_tile_row`, 40 helper calls → 1.19×) and shrinks to **1.04×**
at `one_block` (5 calls). That is the giveaway that **this is the same mechanism as block size**,
attacked from the other side: block size removes *passes* (fewer transitions); killing reconfig
makes each *transition* cheaper. The two therefore **compound** — slowest (`per_tile_row`, reconfig
ON) `28592 ns` → fastest (`one_block`, reconfig OFF) `16642 ns` = **1.72×** combined, vs 1.65× from
block size alone. The per-reconfig cost works out to ≈ 110–150 ns.

**When it is NOT safe:** only skip the reconfig when the data format is genuinely constant across
the phase boundary. If any phase changes dtype — a `bfloat8_b` intermediate, an fp32 result packed
to a CB, mixed-precision inputs — you must keep the reconfig for that boundary or the next op reads
the bytes in the wrong format (silent corruption). The **inits always stay on** regardless; only the
*format* reconfig is the optional part.

## Run the predefined sweeps (regenerate the reports)
```bash
# block-size sweep -> report.md
CBS_REPORT=ttnn/ttnn/operations/examples/compute_block_size/report.md \
scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_compute_block_size.py::test_compute_block_size_device_perf

# reconfig ON-vs-OFF ablation (both modes correctness-gated) -> report_reconfig_ablation.md
CBS_ABLATION_REPORT=ttnn/ttnn/operations/examples/compute_block_size/report_reconfig_ablation.md \
scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_compute_block_size.py::test_compute_block_size_reconfig_ablation
```
Both honor the shape/variant/trial env vars (`CBS_M_TILES`, `CBS_K_TILES`, `CBS_N_TILES`,
`CBS_VARIANTS`, `CBS_TRIALS`, `CBS_KERNEL_ITERS`). The block-size sweep is also driven by the
`python -m ttnn.operations.examples.compute_block_size` CLI above; the reconfig ablation is
currently pytest-only.

## Code
The one compute kernel (all five helper phases inside the `block_rows`-sized loop) is inline in
`program_descriptor_with_inline_kernels.py`; the `reconfig` compile-time flag (default on) flips
every helper between its default reconfig and `NoReconfigure`/`None`. The sharded-L1 single-core
harness, the correctness gate, and both perf tests live in
`../../../../../tests/ttnn/unit_tests/operations/examples/test_compute_block_size.py`. Committed
reports: `report.md` (block size) and `report_reconfig_ablation.md` (reconfig on/off).
