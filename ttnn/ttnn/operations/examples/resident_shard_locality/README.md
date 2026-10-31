# resident_shard_locality — split the work by the shard spec when the data is already in L1

**Problem type:** cross-core communication  ·  **Concept(s):** (1) work split = shard spec (every Tensix core works on the sticks it already holds); (2) CB aliased onto the shard (no staging copy at all)
**First profiled on:** `bh-50-special-adamjanovic-for-reservation-143315` · BH, 13×10 = 130 Tensix cores · 1350 MHz · 2026-10-29 · `8104cced9bc`
**Wide-row / partial-width cases added:** same box · 2026-10-30 · `f45246f274b` (appended block in [`report.md`](report.md))

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
An op receives an L1-sharded ROW_MAJOR tensor (sticks of W elements; HEIGHT_SHARDED shard = [sticks per
Tensix core, W], BLOCK_SHARDED shard = [sticks, W / shard columns]) and needs tiles: tilize, compute,
untilize, and write an L1-sharded output. The easy way to write it is the interleaved way: cut the tensor
into tile-rows of 32 sticks, deal them out evenly over the Tensix cores, and fetch each stick through a
sharded `TensorAccessor`. The work split then ignores who owns which sticks, so almost every stick crosses
the NoC twice (in and back out), and the gap to the owner-local version grows with the number of Tensix
cores. A common half-fix only takes the local path when shard heights are multiples of 32, which sends
most real shards (72, 311, 23 sticks…) down the slow path. Two more half-fixes do the same for other real
shards: taking the local path only for whole-tile shard widths (BLOCK [23, 848], [221, 400], [13, 632] fp32
are all partial-width), and only when the whole row fits in L1 staged several times (a 127-tile HEIGHT row
staged 2 deep in 3 CBs needs ~1.5 MB). Both are wrong: pad the last tile-column locally, and chunk W.

## What this isolates — and how
- **Concepts:** (1) *locality*: the work split is the shard spec, so every read and write stays on its own
  Tensix core; a shard height that is not a multiple of 32 pads the last tile-row locally (padding rows are
  tilized but never written back); a shard width that is not a multiple of 32 pads the last tile-column the
  same way; a wide stick is cut into balanced column chunks of ≤ 32 tiles (`chunk_plan`: chunks =
  ⌈width tiles / 32⌉, chunk = ⌈width tiles / chunks⌉) and only the current chunk is staged, so per-Tensix-core
  L1 is 6 × chunk tiles whatever the row width. (2) *aliasing*: when the shard is whole tile-rows and whole tiles wide,
  the input CB **is** the input shard and the output CB **is** the output shard, so the two copies vanish.
- **Isolation setup:** Tensix ↔ Tensix NoC — sharded L1 in and out, no DRAM anywhere, compute held trivial
  (tilize → untilize, a bit-exact copy through the tile format) and identical in every variant; same
  Tensix cores, same column chunks (≤ 32 tiles), same staging CB depth (2 chunks), same one NoC
  transaction per stick and one barrier per chunk. `resplit` vs `local` changes only the work split;
  `local` vs `local_alias` removes only the two copies.
- **Why it's kernel-level:** which Tensix core processes which sticks, and whether a CB is backed by the
  shard, are decisions inside the op's program — the tensor and its shard spec are fixed inputs.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `resplit` *(baseline)* | Work units = 32 sticks × one column chunk, dealt out in contiguous ranges over the shard grid's Tensix cores (row-major). Reader fetches each stick from its owner via the sharded `TensorAccessor` (NoC0), writer sends each untilized stick back to its owner (NoC1). | — |
| `local` | Each Tensix core walks its own shard: local NoC reads (base + row × stick bytes) into a 2-chunk-deep staging CB, tilize, untilize, local NoC writes into its own output shard. | No cross-core traffic, so no NoC contention; cheaper per-stick addressing. Pays a whole tile-row of tilize for a partial last tile-row. |
| `local_alias` | `local` with the input CB globally allocated on the input shard and the output CB on the output shard; the data-movement kernels only push / pop. Legal only for shard height % 32 == 0 and shard width % 32 == 0 (≤ 32 tiles). | Removes the read and write copies; the unpacker tilizes straight from the shard. |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.resident_shard_locality [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,resplit,local,local_alias}` (one or more) | `all` | which method(s) to run and compare; `local_alias` is skipped (shown as n/a) when the shard is not whole tiles |
| `--trials` | int | `1` | measured passes (median); raise to 3 only if a number looks implausible |
| `--kernel-iters` | int | `1` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | also write a report file (the CLI never overwrites the committed `report.md`) |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--layout` | `{height,block}` | `height` | `TensorMemoryLayout::HEIGHT_SHARDED` or `BLOCK_SHARDED` |
| `--cores` | `grid` \| N \| `ROWS,COLS` | `grid` | Tensix cores: whole compute grid, N row-major (height), or a ROWS×COLS rectangle (block) |
| `--shard-rows` | int (sticks) | *(sweep)* | sticks per Tensix core (shard height) |
| `--shard-width` | int (elements) | *(sweep)* | elements per stick (shard width); bytes per stick = this × 2 (bf16) or × 4 (fp32) |
| `--dtype` | `{bf16,fp32}` | `bf16` | element type of the custom shard (fp32: lossless tilize, fp32 Dest) |
| `--suite` | `{all,base,wide}` | `all` | predefined sweep subset: everything, the original cases, or the wide-row / partial-width cases |
| `--swap-nocs` | flag | off | ablation: reader on NoC1, writer on NoC0 |

**Example invocations:**
```bash
# your own HEIGHT shard on the whole grid
python -m ttnn.operations.examples.resident_shard_locality --shard-rows 72 --shard-width 544

# a BLOCK_SHARDED 4x4 grid, median of 3
python -m ttnn.operations.examples.resident_shard_locality --layout block --cores 4,4 --shard-rows 50 --shard-width 96 --trials 3

# a wide HEIGHT row (127 tiles per stick) and a partial-width fp32 BLOCK shard
python -m ttnn.operations.examples.resident_shard_locality --shard-rows 24 --shard-width 4064
python -m ttnn.operations.examples.resident_shard_locality --layout block --shard-rows 13 --shard-width 632 --dtype fp32

# is the resplit penalty NoC direction? (ablation)
python -m ttnn.operations.examples.resident_shard_locality --cores 13 --shard-rows 72 --shard-width 544 --swap-nocs
```

## Measured result
*Illustrative — BH, 13×10 grid, 1350 MHz; re-run the CLI for your box. Full table: [`report.md`](report.md).*

```
resident_shard_locality   box=bh-50  arch=BH  clock=1350MHz   N=1   kernel-iters=1 (per-launch latency)
  cores=130  HEIGHT [72, 544]   method=resplit (baseline)  61 906 ns  ✓
  cores=130  HEIGHT [72, 544]   method=local                5 509 ns  ✓  → 11.24×
  cores=130  BLOCK  [311, 320]  method=resplit (baseline)  60 484 ns  ✓
  cores=130  BLOCK  [311, 320]  method=local               11 541 ns  ✓  →  5.24×
  cores=130  HEIGHT [32, 1024]  method=local                4 250 ns  ✓
  cores=130  HEIGHT [32, 1024]  method=local_alias          2 804 ns  ✓  →  1.52× over local
  cores=128  HEIGHT [1, 8192]   method=resplit (baseline)  11 576 ns  ✓
  cores=128  HEIGHT [1, 8192]   method=local               17 391 ns  ✓  →  0.67× (resplit wins)

  wide rows / partial widths (2026-10-30, N=3 median)
  cores=130  HEIGHT [24, 4064]  method=resplit (baseline)  80 170 ns  ✓
  cores=130  HEIGHT [24, 4064]  method=local               10 231 ns  ✓  →  7.84×   (4 chunks × 32 tiles)
  cores=130  HEIGHT [17, 5120]  method=resplit (baseline) 134 527 ns  ✓
  cores=130  HEIGHT [17, 5120]  method=local               11 987 ns  ✓  → 11.22×   (5 chunks × 32 tiles)
  cores=130  BLOCK  [13, 632]   fp32  resplit → local      11 847 →  5 590 ns  ✓  →  2.12×
  cores=130  BLOCK  [221, 400]  bf16  resplit → local      56 368 →  9 880 ns  ✓  →  5.71×
  cores=130  BLOCK  [16, 848]   bf16  resplit → local      13 030 →  3 116 ns  ✓  →  4.18×
```

**Reading of the result:**
- **The resplit's cost is cross-core NoC reads under contention.** Stubbing the copies at HEIGHT [72, 544]
  on 130 Tensix cores (one-off ablation): no reads 24 µs, no writes 57 µs, neither 4.9 µs, of 60 µs. The
  penalty grows with the number of Tensix cores (`report.md`, mechanism table): 1.03× on 1, 1.29× on 2,
  6.6× on 13 (one grid row), 10.9× on 130. In the row-major deal-out every owner sits at or after its
  reader, so the NoC0 read responses run the long way round the torus; swapping the NoCs cuts one grid
  row to **1.31×** but still leaves **8.2×** on 130 Tensix cores, where flows span rows and wrap either
  way. `local` has no cross-core traffic, so its time is flat in Tensix cores (5.35–5.53 µs).
- **When the shard is whole tile-rows, `resplit` lands on the owners by accident** (one or more tile-rows per
  Tensix core, dealt out evenly): every read is local and the remaining 1.14–1.59× is per-stick addressing —
  sharded `TensorAccessor` lookup vs base + offset — largest for small sticks (256 B: 1.59×; 2048 B: 1.14×),
  where the reader is issue-bound. Shift the shard height off a multiple of 32 and the 10× cliff returns.
- **Field evidence said 3–4.5× (HEIGHT) and 2–4× (BLOCK).** Measured here: HEIGHT 2.6–11.2×, BLOCK
  5.2–6.4×. The baseline's cost depends on route length and Tensix-core count (1.3× on one grid row with
  the read NoC along the owner direction, 11× on the full grid), so a baseline that happens to read along
  the NoC direction, or on fewer Tensix cores, lands in the field's range. Measure your own placement.
- **Wide rows and partial widths need no fallback.** HEIGHT [24, 4064] (127 tiles per stick) and [17, 5120]
  (160) run `local` at 7.8× and 11.2× over `resplit`, with 384 KB of CBs per Tensix core (4 or 5 chunks of 32
  tiles, 2 deep, 3 CBs) where whole-row staging would need 1 524 KB and 1 920 KB — more than BH's L1. Partial-width
  BLOCK shards on the 10×13 grid win 2.1–6.3× (fp32 [13, 632] 2.1×: 13 sticks per Tensix core and a single
  65-unit resplit — the smallest cross-core traffic of the set). Field evidence for these shapes: a reference
  that ran them locally with ≤ 32-tile chunks was 2.5–3.1× faster than the re-split in a real op (with real
  compute in both); this isolated copy shows the full gap.
- **The chunk rule matters.** Taking the widest divisor of the width in tiles that is ≤ 32 degrades a prime
  width to 1-tile chunks: [24, 4064] (127 tiles) then moves 64 B per NoC transaction and `local` takes 68.6 µs
  instead of 10.2 µs (2.1× over `resplit` instead of 7.9×). Balance the chunks and let the last one carry the
  padding.
- **Aliasing** removes the two copies: 1.52–1.58× over `local` at one tile-row per Tensix core (nothing to
  overlap the copies with), 1.21–1.27× at 2–3 tile-rows (the 2-chunk staging CB already hides most of the
  copy behind compute).

## When it helps / when it does not
Measured on device by `test_resident_shard_locality_regimes` (interleaved A/B, median of 5; asserts the
direction). Conditions in portable units: sticks per Tensix core, bytes per stick, Tensix cores.

| Regime | Condition | Shape(s) | Baseline ns | Lever ns | Speedup | Trials | Board |
|---|---|---|---|---|---|---|---|
| helps | ≥ 1 tile-row per Tensix core, shard height not a multiple of 32, whole grid | HEIGHT [72, 544], 130 Tensix cores | 59 221 (`resplit`) | 5 498 (`local`) | **10.77×** | 5 | BH 13×10 |
| helps | BLOCK_SHARDED, whole grid | BLOCK [311, 320], 10×13 | 63 061 (`resplit`) | 11 537 (`local`) | **5.47×** | 5 | BH 13×10 |
| helps | whole tile-rows, ≤ 32 tiles wide (aliasing legal) | HEIGHT [32, 1024], 130 Tensix cores | 4 247 (`local`) | 2 720 (`local_alias`) | **1.56×** | 5 | BH 13×10 |
| helps | wide rows, W chunked (127 tiles per stick) | HEIGHT [24, 4064], 130 Tensix cores | 81 129 (`resplit`) | 10 222 (`local`) | **7.94×** | 5 | BH 13×10 |
| helps | wide rows, W chunked (160 tiles per stick) | HEIGHT [17, 5120], 130 Tensix cores | 132 890 (`resplit`) | 11 976 (`local`) | **11.10×** | 5 | BH 13×10 |
| helps | partial-width BLOCK, fp32 | BLOCK [13, 632] fp32, 10×13 | 11 939 (`resplit`) | 5 596 (`local`) | **2.13×** | 5 | BH 13×10 |
| helps | partial-width BLOCK | BLOCK [221, 400], 10×13 | 55 783 (`resplit`) | 9 850 (`local`) | **5.66×** | 5 | BH 13×10 |
| helps | partial-width BLOCK | BLOCK [23, 848] / [16, 848], 10×13 | 21 676 / 13 248 (`resplit`) | 3 453 / 3 116 (`local`) | **6.28× / 4.25×** | 5 | BH 13×10 |
| does not help | 1 stick per Tensix core, 16 KB sticks | HEIGHT [1, 8192], 128 Tensix cores | 12 074 (`resplit`) | 17 423 (`local`) | **0.69×** | 5 | BH 13×10 |

Sub-tile-row crossover (sweep, single pass, 128 Tensix cores): local costs one tile-row per column chunk
whatever the stick count (~68 ns per tile of tilize + untilize: [·, 8192] = 256 tiles ≈ 17.5 µs flat for 1–8
sticks), while `resplit` packs full tile-rows onto fewer Tensix cores (32 of 128 at 1 stick).

| Sticks per Tensix core | 2 KB sticks (32 tiles) | 8 KB sticks (128 tiles) | 16 KB sticks (256 tiles) |
|---:|---:|---:|---:|
| 1 | 1.52× | **0.68×** | **0.60–0.81×** |
| 2 | 1.58× | 0.98× | 0.92–0.95× |
| 3 | — | 1.37× | 1.69–1.74× |
| 4 | 1.97× | — | 2.14–2.26× |

Local loses only for sub-tile-row shards of wide sticks: ≤ 2 sticks per Tensix core at ≥ 8 KB per stick
(crossover ≈ 2 sticks); at 2 KB sticks it wins even at 1 stick. The boundary is sticks per Tensix core, not
tiles per stick or row width: at ≥ 3 sticks per core `local` wins at every stick width measured once W is
chunked (3 × 16 KB 1.74×, 17 × 10 KB 11.2×, 24 × 8 KB 7.9×). `local_alias` has no measured loss in its
legal domain; its gain shrinks as tile-rows per Tensix core grow (1.21× at 3).

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py
```

## Code
`program_descriptor_with_inline_kernels.py`: `chunk_plan()` (balanced ≤ 32-tile column chunks) and
`local_l1_bytes()` (CB bytes per Tensix core, chunked vs whole row); `_LOCAL_READER` / `_LOCAL_WRITER` (own shard, base + offset)
vs `_RESPLIT_READER` / `_RESPLIT_WRITER` (unit → tile-row, page column, chunk; `TensorAccessor` per stick),
the `resplit` unit deal-out in `create_program_descriptor()`, and the `local_alias` branch
(`cb_descriptor_from_sharded_tensor` with tile-sized pages + `_ALIAS_READER` / `_ALIAS_WRITER`). The compute
kernel `_COMPUTE` is shared by all three variants (fp32: lossless tilize with fp32 Dest and
UnpackToDestFp32, `_compute_config()`).
