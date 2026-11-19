# row_major_flat_tiles — skip tilize / untilize on ROW_MAJOR data when the math does not need tiles

**Problem type:** compute-bound: format conversion the math does not need  ·  **Concept(s):** (1) flat tiles: 1024 contiguous row-major elements in a CB page, used as a 32×32 tile with no tilize / untilize; (2) a per-row reduction over whole flat tiles (W a multiple of 1024): DEST accumulate + whole-tile scalar reduce + scalar broadcast
**First profiled on:** `bgd-lab-22-special-adamjanovic-for-reservation-169096` · WH B0 n150, 8×8 = 64 Tensix cores · 1000 MHz · 2026-11-19 · `9dc2b92132c` (+ this example, uncommitted) ([`report_wh_n150.md`](report_wh_n150.md))
**Blackhole:** not measured.

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
An op takes a `Layout::ROW_MAJOR` tensor (one page = one stick of W elements) and returns one. The usual
kernel stages 32 stick pieces, tilizes them, computes on tiles, untilizes, and writes stick pieces back. When
the math does not depend on where an element sits inside a tile, the two conversions are pure overhead: an
elementwise op maps element i of a page to element i of the output page in any layout, and a sum over a
whole tile is the same sum in any order. So 1024 contiguous row-major elements copied into a CB page can be
used as a tile directly — a **flat tile**. For a per-row reduction this works when every row is a whole
number of flat tiles (W % 1024 == 0): no flat tile then holds elements of two rows.

## What this isolates — and how
- **Concepts:** (1) drop tilize → compute → untilize for flat tiles; (2) the per-row reduction on flat tiles:
  per row, sum its Wf = W / 1024 flat tiles in DEST, collapse the tile to one scalar (REDUCE_SCALAR) × 1/W,
  subtract that scalar from the row's flat tiles (scalar broadcast).
- **Isolation setup:** interleaved DRAM ROW_MAJOR bf16 in and out, real but light compute, from one Tensix core
  up to the full grid. Held equal in both variants: the work split (whole tile-rows of 32 sticks per Tensix core,
  contiguous ranges, row-major cores), bytes moved, NoC transfer size and count (2 KB per transfer whenever
  W ≥ 1024), the compute config (reduce: HiFi4 + fp32 DEST in both). Eltwise: reader and writer are
  byte-identical — the same 32 stick pieces land in the same CB pages; only the compute kernel changes.
- **Why it's kernel-level:** whether the compute kernel converts layout, and how a row reduction walks its
  tiles, are decisions inside the op; the tensor and its layout are fixed inputs.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `tilize_untilize` *(baseline)* | Eltwise: per block of 32 stick pieces × CH ≤ 32 tile-columns: `tilize` → op → `untilize`. Reduce: tilize the tile-row chunk by chunk into a resident tiled block, REDUCE_ROW mean (one output per tile-row), subtract it broadcast down the columns chunk by chunk, untilize each chunk. | — |
| `flat_tiles` | Eltwise: op straight on the staged pages. Reduce: per block of RB whole sticks (RB × Wf ≤ 32 flat tiles), REDUCE_SCALAR mean per row (one output per row), `sub` with a scalar broadcast, pack straight to the writer's CB. Refuses a reduce with W % 1024 != 0 (`ValueError`, `flat_tiles_applicable`). | No tilize / untilize passes (~170 ns per tile, WH B0) and two fewer CBs; for the reduce, a block of a few sticks instead of a whole tiled tile-row (CBs 900 → 266–274 KB per Tensix core at W ≥ 4096). Costs: 32 reduce outputs per tile-row instead of 1. |

Two knobs the reduce lever needs, both chosen by measurement (ablation below), not by taste:
- **Datapath** (`--reduce-algo`, default `auto` = `resolve_reduce_algo`): `flat_tiles` finalizes one mean per
  *row*, so the per-output cost counts 32×. Rows of 1–2 flat tiles use the FPU `reduce_tile` path (cheap per
  output); from 4 flat tiles the pairwise `add_tiles` + SFPU collapse (`accumulate`) wins. The baseline uses
  `accumulate`, its fastest at every W measured.
- **Read / write order** (`--piece-order`, default `interleaved`): a block is RB whole sticks, and a stick lives in
  one DRAM bank. Issuing a stick's Wf transfers back to back (`row`) queues them on one bank; going flat-tile
  column by column across the block's sticks (`interleaved`) spreads consecutive transfers over RB banks.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.row_major_flat_tiles [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,tilize_untilize,flat_tiles}` (one or more) | `all` | which method(s) to run and compare |
| `--trials` | int | `1` | measured passes (median); raise to 3 only if a number looks implausible |
| `--kernel-iters` | int | `1` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | also write the Markdown report to this path |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--case` | `{eltwise,reduce}` | *(predefined sweep)* | one case instead of the sweep |
| `--op` | `{relu,gelu}` | `relu` | eltwise op |
| `--reps` | int | `1` | eltwise: op passes in DEST per tile — the compute-per-element knob |
| `--rows` | ints, comma-separated | `32,512,2048,8192` | H values (sticks), multiples of 32; tile-rows = H / 32 = Tensix cores up to the grid |
| `--widths` | ints, comma-separated | per case | W values (elements), multiples of 32 |
| `--reduce-algo` | `{auto,accumulate,reduce_tile}` | `auto` | reduce datapath, both variants |
| `--piece-order` | `{interleaved,row}` | `interleaved` | reduce `flat_tiles` transfer order |

**Example invocations:**
```bash
# the whole predefined sweep + reduce ablation, median of 3, written to a file
python -m ttnn.operations.examples.row_major_flat_tiles --trials 3 --report /tmp/rmft.md

# your shape: eltwise gelu, 256 sticks of 2048 elements
python -m ttnn.operations.examples.row_major_flat_tiles --case eltwise --op gelu --rows 256 --widths 2048

# does the transfer order matter on a DRAM-bound reduce?
python -m ttnn.operations.examples.row_major_flat_tiles --case reduce --rows 8192 --widths 8192 --variant flat_tiles --piece-order row
```

## Measured result
*WH B0 n150, 8×8 grid, 1000 MHz, N = 3 (median), kernel-iters = 1. Full table: [`report_wh_n150.md`](report_wh_n150.md).*

```
row_major_flat_tiles   box=bgd-lab-22  arch=WH_B0  clock=1000MHz   N=3   kernel-iters=1 (per-launch latency)
  eltwise relu   [32, 4096]    cores=1   tilize_untilize   59 222 ns  ✓   flat_tiles   37 132 ns  ✓  → 1.59×
  eltwise relu   [512, 4096]   cores=16  tilize_untilize   65 822 ns  ✓   flat_tiles   46 270 ns  ✓  → 1.42×
  eltwise relu   [2048, 4096]  cores=64  tilize_untilize  177 552 ns  ✓   flat_tiles  177 787 ns  ✓  → 1.00×  (189 GB/s, DRAM)
  eltwise relu16 [32, 4096]    cores=1   tilize_untilize  444 277 ns  ✓   flat_tiles  422 972 ns  ✓  → 1.05×
  reduce         [32, 8192]    cores=1   tilize_untilize   83 321 ns  ✓   flat_tiles   45 006 ns  ✓  → 1.85×
  reduce         [32, 1024]    cores=1   tilize_untilize   15 421 ns  ✓   flat_tiles   14 331 ns  ✓  → 1.08×
  reduce         [8192, 8192]  cores=64  tilize_untilize 1 393 268 ns ✓   flat_tiles 1 388 814 ns ✓  → 1.00×  (193 GB/s, DRAM)
  reduce         [32, 1536]    cores=1   tilize_untilize   19 864 ns  ✓   flat_tiles  refused (1536 % 1024 != 0)
```

**Reading of the result:**
- **The saving is a fixed ~170 ns per tile on WH B0** — tilize + untilize of one tile, with their init and CB
  handshakes: (59 222 − 37 132) / 128 tiles = 173 ns at W = 4096, (19 021 − 13 682) / 32 = 167 ns at W = 1024
  (eltwise relu, 1 Tensix core). The speedup is therefore that cost over everything else the Tensix core does per
  tile: 1.18× at W = 64 (2-tile blocks, fixed costs dominate both), 1.39× at W = 1024, 1.59× at W = 4096; gelu
  1.35× / 1.52× (its SFPU work is only ~50 ns per tile more than relu).
- **It vanishes when tilize / untilize is a small share.** Sixteen relu passes per tile (~200 ns each):
  1.04–1.09× at every row count. Full grid, DRAM-bound (186–199 GB/s read + write against a ~288 GB/s spec):
  1.00–1.03× — the Tensix cores were waiting on DRAM, not on the unpacker / packer. The gain is largest where
  few Tensix cores carry wide rows (1–16 tile-rows), i.e. where compute, not DRAM, sets the time. Narrow sticks
  on the full grid (W = 64, H = 8192: 128 B transfers, 137–141 GB/s) are flat too: relu 1.00×, gelu 0.96×.
- **The reduce gains with W, because the lever's extra cost is per row and its saving per tile.** One mean per
  row means 32 reduce outputs per tile-row; the baseline has 1. Rows of 1 flat tile: 32 outputs for 32 tiles,
  and the saving barely beats it — 1.08× (with `reduce_tile`; with `accumulate`, 0.60×: 25.8 vs 15.4 µs).
  8 flat tiles per row: 32 outputs for 256 tiles — 1.85×. Full grid: 0.99–1.07× (DRAM-bound).
- **Transfer order is the difference between flat and a loss on DRAM-bound reduces.** With `row` order, a
  block's Wf transfers per stick queue on one bank: [8192, 8192] 1 618 730 ns vs 1 386 175 ns `interleaved`
  (baseline 1 388 305), [2048, 4096] 196 504 vs 176 448 (baseline 189 822). A different block shape than the
  baseline's needs its own transfer order — measure it.
- **Not reproduced: 20–30×.** The isolated mechanism tops out at 1.85× on WH B0 here; a speedup can only reach
  time / (time − tilize-and-untilize share). A 20–30× gain in an op means its baseline spent ≥ 95% of the time in
  something the lever also removed (e.g. tilizing whole tile-rows that are mostly padding); this example does
  not model that and cannot attribute it.

## When it helps / when it does not
Measured by `test_row_major_flat_tiles_regimes` (interleaved A/B, median of 7 trials per variant, 3 runs,
min–max in the test; asserted on WH B0 against a bound derived from these numbers, `perf_test_utils.check_regime`;
logged only on any other board). WH B0 n150 8×8, 2026-11-19.

| Regime | Condition | Shape [H, W] | Tensix cores | Baseline ns | Lever ns | Speedup |
|---|---|---|---:|---:|---:|---|
| helps | eltwise, 1 tile-row, wide row | relu [32, 4096] | 1 | 59 252 | 37 143 | **1.60×** (1.595–1.597) |
| helps | eltwise, 16 tile-rows | relu [512, 4096] | 16 | 65 618 | 46 409 | **1.41×** (1.40–1.41) |
| does not help | eltwise, heavy compute per element | relu ×16 [32, 4096] | 1 | 444 208 | 422 956 | **1.05×** |
| does not help | eltwise, full grid, DRAM-bound | relu [2048, 4096] | 64 | 178 623 | 174 884 | **1.02×** (1.01–1.02) |
| helps | reduce, 8 flat tiles per row | [32, 8192] | 1 | 83 460 | 45 017 | **1.85×** |
| helps | reduce, 4 flat tiles per row, 16 tile-rows | [512, 4096] | 16 | 60 559 | 47 068 | **1.29×** (1.26–1.29) |
| measure | reduce, 1 flat tile per row | [32, 1024] | 1 | 15 470 | 14 292 | **1.08×** |
| does not help | reduce, full grid, DRAM-bound | [8192, 8192] | 64 | 1 380 275 | 1 380 107 | **1.00×** (1.00–1.01) |
| does not apply | reduce, W % 1024 != 0 | [·, 1536] | — | baseline only | refused | — |

Reduce speedup by W and row count (sweep, N = 3): W = 1024: 1.08 / 1.05 / 1.02 / 0.99× at 1 / 16 / 64 / 64
(4 tile-rows each) Tensix cores; W = 2048: 1.16 / 1.20 / 1.02 / 1.01×; W = 4096: 1.39 / 1.30 / 1.07 / 1.02×;
W = 8192: 1.85 / 1.18 / 1.03 / 1.00×. Eltwise relu: W = 64: 1.18 / 1.16 / 1.12 / 1.00×; W = 1024: 1.39 / 1.29 /
1.13 / 1.03×; W = 4096: 1.59 / 1.42 / 1.00 / 1.02×.

Reduce ablation (sweep, N = 3, ns):

| [H, W] | Wf | baseline accumulate | baseline reduce_tile | flat accumulate row | flat accumulate interleaved | flat reduce_tile interleaved |
|---|---:|---:|---:|---:|---:|---:|
| [32, 1024] | 1 | 15 447 | 16 484 | 25 898 | 25 761 | **14 293** |
| [32, 2048] | 2 | 25 266 | 27 215 | 26 364 | 26 301 | **21 654** |
| [32, 4096] | 4 | 44 594 | 48 850 | 32 315 | **32 136** | 36 050 |
| [32, 8192] | 8 | 83 542 | 91 874 | 45 346 | **45 108** | 65 773 |
| [2048, 4096] | 4 | 189 822 | 186 930 | 196 504 | **176 448** | 176 676 |
| [8192, 8192] | 8 | 1 388 305 | 1 382 325 | 1 618 730 | **1 386 175** | 1 402 990 |

Not covered: L1-sharded inputs (the reader would copy local stick pieces instead of DRAM ones; the compute side
is unchanged), fp32, W not a multiple of 32, and Blackhole.

## Run the predefined sweep (regenerates `report_wh_n150.md`)
The sweep is opt-in (`RMFT_SWEEPS=1`; the CLI sets it) and writes a report only to the path in `RMFT_REPORT`:
```bash
RMFT_SWEEPS=1 RMFT_TRIALS=3 RMFT_REPORT=ttnn/ttnn/operations/examples/row_major_flat_tiles/report_wh_n150.md \
    scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py::test_row_major_flat_tiles_device_perf
```

## Code
`program_descriptor_with_inline_kernels.py`: `_ELTWISE_COMPUTE` (`if constexpr (flat)`: one `eltwise_chain` on the
staged pages vs `tilize` → chain → `untilize`); `_REDUCE_BASELINE_COMPUTE` (tilize, `reduce_mean` REDUCE_ROW, `sub`
COL broadcast, untilize) vs `_REDUCE_FLAT_COMPUTE` (`reduce_mean` / `reduce` REDUCE_SCALAR with
`ReduceInputBlockShape::of(1, Wf, RB)` — one batch per row — then `sub` with `BroadcastDim::Scalar` and
`InputTileMapping::Col` over `grid(RB, Wf)`); `_READER` / `_WRITER` mode 0 (32 stick pieces) vs mode 1 (RB whole
sticks, one 2 KB transfer per flat tile, `order`); `flat_tiles_applicable()`, `resolve_reduce_algo()`,
`rows_per_block()`, `cb_tiles()`.
