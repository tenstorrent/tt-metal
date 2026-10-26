# partial_tile_read — read only the bytes of a tile the kernel uses

**Problem type:** data movement below the ceiling: transfer size  ·  **Concept:** bytes per tile of a DRAM operand (whole tile vs only the face-rows the consumer reads)
**First profiled on:** `bh-50-special-adamjanovic-for-reservation-143315` · BH p150b (130 Tensix cores) · 1350 MHz (`CHIP_FREQ`) · 2026-10-26 · `5eac7659e6c`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A kernel takes a row vector that is broadcast over rows — a bias, a scale, a per-column weight. The
vector is stored as a `Layout::TILE` tensor of shape [32, W], so only row 0 of every 32x32 tile holds
data and rows 1-31 are padding. Every Tensix core needs the whole vector (W/32 tiles), and the obvious
reader fetches whole tiles: 2048 B per bf16 tile, of which the broadcast unpacker uses 64 B (row 0 of
face 0 and row 0 of face 1, 32 B each). With many Tensix cores reading the same vector, those padding
bytes are what fills the DRAM bandwidth.

## What this isolates — and how
- **Concept:** how many bytes of each tile the reader fetches. Nothing else changes: same CBs, same
  compute, same work split, same output, same tiles per barrier.
- **Isolation setup:** DRAM read efficiency, multi-core. The vector is in DRAM
  (`TensorMemoryLayout::INTERLEAVED`, `BufferType::DRAM`). The other input and the output are
  height-sharded in L1 (`rows` tile-rows of W/32 tiles per Tensix core), so the vector is the only
  DRAM traffic. Compute is one `add_tiles_bcast_rows` per tile (`out = x + v[0, :]`), cheap but real:
  the exact-match check proves the bytes that matter arrived. Reader on NCRISC, no writer kernel.
- **Why it's kernel-level:** the tensor format is unchanged; the reader decides which bytes of each
  page it fetches. The data type (bf16) is not a knob here.

## The methods being compared
| Variant | What it does | Bytes per tile (BH / WH) | Why it should differ |
|---|---|---|---|
| `whole_tile` *(baseline)* | one `noc_async_read_page` per tile | 2048 / 2048 B | — |
| `row0_only` | two `noc_async_read`s per tile: face 0 row 0 and face 1 row 0, each rounded up to the DRAM read alignment (64 B BH / 32 B WH, from `ttnn.get_dram_alignment()`), landing at the same offsets inside the CB tile slot | 128 / 64 B | 16x fewer DRAM bytes; twice the requests |
| `row0_span` | one `noc_async_read` from the start of face 0 through face 1's first segment | 576 / 544 B | 3.6x fewer bytes than whole tile, half the requests of `row0_only` — tells "bound by bytes" from "bound by requests" |
| `no_read` *(ablation, not a method)* | the reader pushes the CB pages without reading | 0 B | everything except the vector read; output is not checked |

All read variants batch `block` tiles (default 8) per `noc_async_read_barrier` into a CB of 2 × block
tiles, so the baseline is already pipelined. Correctness: each variant runs on its own fresh inputs,
baseline first, and must match `x + v[0, :]` exactly (rows 1-31 of `v` hold large values, so using any
other row fails). A lever that skipped its reads would find the previous vector in the CB and fail;
the test proves this by running `no_read` and asserting it does **not** match.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.partial_tile_read [options]   # from the tt-metal root
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,whole_tile,row0_only,row0_span,no_read}` | `all` | which method(s) to run and compare |
| `--trials` | int | `1` | measured launches per method (median); methods interleaved within each trial |
| `--kernel-iters` | int | `1` | vector streams per launch — **1 = per-launch latency; large = steady state** |
| `--report` | path | *(print only)* | also write the Markdown report |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--cores` | int... or `full` | sweep | Tensix cores reading the vector (row-major from (0,0)); `full` = whole grid |
| `--widths` | int... (elements, multiple of 32) | sweep | vector length W; W/32 tiles per Tensix core |
| `--rows` | int... | sweep | tile-rows of resident input per Tensix core that reuse each vector tile |
| `--block` | int (tiles) | `8` | tiles per read barrier, both variants (the CB holds 2 × block) |

```bash
# full grid, 8192-element vector, 3 trials
python -m ttnn.operations.examples.partial_tile_read --cores full --widths 8192 --trials 3

# few Tensix cores with a deeper reader
python -m ttnn.operations.examples.partial_tile_read --cores 1 8 --widths 8192 --block 32
```

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box. Full table: `report.md`.*

```
partial_tile_read   box=bh-50  arch=BH p150b  CHIP_FREQ=1350MHz  N=3 (median)  kernel-iters=1 (latency)  block=8
  cores=130 (full grid)  placement=row-major  W=8192  method=whole_tile  524288 B/core  209892 ns  324.7 GB/s  ✓
  cores=130 (full grid)  placement=row-major  W=8192  method=row0_only    32768 B/core  107604 ns   39.6 GB/s  ✓  → 1.95×
  cores=130 (full grid)  placement=row-major  W=8192  method=row0_span   147456 B/core   72735 ns  263.5 GB/s  ✓  → 2.89×
  cores=32               placement=row-major  W=2048  method=whole_tile  131072 B/core   18331 ns  228.8 GB/s  ✓
  cores=32               placement=row-major  W=2048  method=row0_only     8192 B/core    7773 ns   33.7 GB/s  ✓  → 2.36×
  cores=1                placement=(0,0)      W=8192  method=whole_tile  524288 B/core   21448 ns   24.4 GB/s  ✓
  cores=1                placement=(0,0)      W=8192  method=row0_only    32768 B/core   21983 ns    1.5 GB/s  ✓  → 0.98×
```

**Reading of the result:** with ≥ 32 Tensix cores the whole-tile reads of all Tensix cores together
run at 210–325 GB/s, close to the DRAM ceiling, and the kernel is read-bound (`no_read` is 6–29×
faster). Fetching only row 0 removes 94% of the bytes and gives **2.0–2.7×**. It does not give 16×:
`row0_only` then runs at a flat ~30–40 GB/s whatever the Tensix-core count or W — it is bound by the
number of 64 B requests (~0.6 requests per ns across the chip), not by bytes. Halving the requests
with one 576 B read per tile (`row0_span`) is faster still: **2.9–3.2×**. With 1–8 Tensix cores each
reader is latency-bound — 8 tiles in flight per barrier take the same round trip whatever their size —
so the bytes do not matter (0.97–1.06×), even though the read is still 60% of the kernel time.

## When it helps / when it does not
Measured by `test_partial_tile_read_regimes` (asserts the direction) and the sweep in `report.md`.
Board: BH p150b, 130 Tensix cores, bf16, block = 8 tiles per barrier, interleaved A/B.

| Regime | Condition | Shape(s) | Baseline ns | Lever ns | Speedup | Trials | Board |
|---|---|---|---|---|---|---|---|
| helps | ≥ 32 Tensix cores read the vector; whole-tile bytes near the DRAM ceiling | 130 Tensix cores, W=8192 (256 tiles) | 209 892 | 107 604 | **1.95×** (span 2.89×) | 3 | BH p150b |
| helps | same, fewer Tensix cores | 32 Tensix cores, W=2048 (64 tiles) | 18 331 | 7 773 | **2.36×** (span 3.03×) | 3 | BH p150b |
| helps | even one tile, if many Tensix cores read it | 130 Tensix cores, W=32 | 9 618 | 3 741 | **2.57×** | 3 | BH p150b |
| shrinks | each vector tile reused by 16 tile-rows of compute per Tensix core | 32 Tensix cores, W=512, rows=16 | 10 916 | 9 105 | 1.20× | 3 | BH p150b |
| does not help | 1 Tensix core: reader latency-bound | 1 Tensix core, W=8192 | 21 448 | 21 983 | **0.98×** | 3 | BH p150b |
| does not help | 8 Tensix cores: aggregate 189 GB/s, still latency-bound per Tensix core | 8 Tensix cores, W=8192 | 22 206 | 22 348 | **0.99×** | 3 | BH p150b |
| loses | 1 Tensix core with a deeper reader (block=32): NCRISC issues 2× the requests | 1 Tensix core, W=8192 | 11 859 | 14 384 | 0.82× (span 1.13×) | 3 | BH p150b |

- **Helps** when the sum of whole-tile bytes over all reading Tensix cores is what limits the DRAM:
  here from 32 Tensix cores up, at every W from 1 tile to 256 tiles. A deeper baseline (block=32,
  365 GB/s at the full grid) shrinks the gain to 1.71× (span 2.68×) — the lever still wins.
- **Does not help** at 1–8 Tensix cores: the time per barrier is a round trip, not a transfer, so
  fewer bytes per tile save nothing; deepen the reader first (`--block`). With a deep reader on one
  Tensix core the extra requests make `row0_only` slower; `row0_span` still gains 1.13×.
- **Shrinks** as compute per vector tile grows (1.20× at 16 tile-rows of reuse): the read is
  mostly hidden behind compute (`no_read` 8 049 ns, so only ~2.9 µs of the baseline is exposed
  read). A configuration where it is fully hidden does not fit
  in L1 here (resident input + output at 16 tile-rows is already 1 MB per Tensix core).
- WH B0 (32 B DRAM read alignment, so 64 B per tile; 12 DRAM banks): not measured.

## Run the predefined sweep (regenerates `report.md`)
```bash
PTR_TRIALS=3 PTR_REPORT=$PWD/ttnn/ttnn/operations/examples/partial_tile_read/report.md \
    scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_partial_tile_read.py::test_partial_tile_read_device_perf
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_partial_tile_read.py
```

## Code
`program_descriptor_with_inline_kernels.py`: `_READER_KERNEL` (`if constexpr (mode == …)`, the only
difference between variants), `segment_bytes()` (face-row size rounded up to `ttnn.get_dram_alignment()`),
`create_program_descriptor()`. The segment for face 1 starts at `tile_bytes / 4` (512 B for bf16).
