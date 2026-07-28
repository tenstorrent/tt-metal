# index_staging — resolve random per-index access in SRAM vs. one remote read each

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** index-driven access (`out[w] = src[idx[w]]`) — one bulk read + SRAM-local indexing vs. one remote NoC transaction per index
**First profiled on:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181` · WH (wormhole_b0) · ~1000 MHz · 2026-07-22 · `a9c0d008bde`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You need `out[w] = src[idx[w]]` where `idx` is an arbitrary index list — an indexed
select over some indexable dimension in DRAM. The obvious kernel issues one remote NoC read per
index. But you cannot read 2 bytes from an arbitrary DRAM offset: the read granularity
is a whole aligned line (32 B here), and every read command carries a fixed issue +
round-trip cost. So a per-index read pulls 16× the bytes you want and, more importantly,
adds one more transaction to a NoC-issue-bound loop. When the index list touches most of
a row, it is cheaper to bulk-read the whole indexable dimension into L1 once and resolve
every index locally.

## What this isolates — and how
- **Concept:** how the random per-index access is serviced — W remote reads, or one bulk
  read + W SRAM-local extracts.
- **Isolation setup:** the *DRAM-read* row of the isolation rule. There is no compute
  (pure dataflow: reader selects by index, writer writes); the source is a single interleaved DRAM
  page per row; single core, placement held constant. **Both variants run the identical
  W-element local extract loop**, so that cost is common — the only difference is the read
  strategy, and any delta is attributable to it.
- **Why it's kernel-level:** the author chooses the NoC access pattern (many small remote
  reads vs. one bulk read + local indexing). It is not a model/dtype choice — dtype,
  shape, page size, and cores are identical across variants.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `remote_per_index` *(baseline)* | For each of the W indices, issue a separate remote NoC read of the whole aligned 32-byte line containing element `idx[w]`, then extract the 2 wanted bytes. All W reads pipelined under one barrier. | W transactions, each moving 16× the useful bytes. The NCRISC issue loop is the bottleneck; cost scales with transaction **count**. |
| `l1_staged` | One bulk contiguous read of the whole source row into L1, then extract every element locally in SRAM. | 1 transaction moving exactly the useful bytes; the random access is SRAM-local. Collapses W issues to 1. |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.index_staging [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,remote_per_index,l1_staged}` | `all` | which method(s) to run/compare (via `IS_*` env; default runs all) |
| `--trials` | int | `20` | measured trials; report shows the average ns/op |
| `--iters` | int | `1` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | pytest writes the committed report; the CLI prints the table |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--shape` | `ROWS,W` | `8,512` | rows processed on the core × W indices per row (W must be a multiple of 16) |
| `--width` | int | *(from --shape)* | override just W, keeping ROWS |
| `--dist` | comma list of `{sorted,shuffled}` | `sorted,shuffled` | index distribution: `sorted` monotonic, `shuffled` the same multiset in random order |
| `--cores` | int | `1` | cores running the (independent) pipeline |

**Example invocations:**
```bash
# A/B both methods on a wide row, steady-state
python -m ttnn.operations.examples.index_staging --shape 8,2048 --iters 50

# just the candidate, single-shot latency
python -m ttnn.operations.examples.index_staging --variant l1_staged --iters 1
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

```
index_staging   box=bgd-lab-t3002...  arch=wormhole_b0   trials=20 (avg ns/op)   iters=1 (per-launch)
  rows=8  W=512  cores=1  src_row=1024 B  baseline_read=16384 B  placement=single core
    dist=sorted    method=remote_per_index (baseline)  ...  264902 ns  ✓
    dist=sorted    method=l1_staged                    ...  124177 ns  ✓  → 2.13×
    dist=shuffled  method=remote_per_index (baseline)  ...  264968 ns  ✓
    dist=shuffled  method=l1_staged                    ...  124136 ns  ✓  → 2.13×
```

**Reading of the result:** l1_staged is ~2.1–2.2× faster across every W measured
(128 → 8192) and holds in steady state (iters=50). The win is "same work, faster":
one bulk read of exactly the useful bytes replaces W remote reads that each move a
full 32-byte line. **The win is bounded, not unbounded** (~2.2×, not 16×) because both
variants share the same W-element local-extract loop, which becomes the common floor
as W grows — so the ratio is roughly constant in W rather than diverging.

**Index distribution had no measurable effect** (sorted ≡ shuffled at every W). This is
the real finding: when the entire indexable dimension is one interleaved DRAM page (the
very setup that makes a single bulk read possible), the per-index reads are pipelined
into one open DRAM row and the loop is bottlenecked on **NCRISC transaction issue**, not
DRAM locality. Transaction *count* is what matters — and count is order-independent, so
coalescing/row-buffer sensitivity does not appear here. Staging wins by collapsing the
count, not by improving locality.

## Run the predefined sweep (regenerates report.md)
```bash
scripts/run_safe_pytest.sh ttnn/ttnn/operations/examples/index_staging/test_index_staging.py
```
(Correctness: `test_index_staging_correctness`; measurement: `test_index_staging_device_perf`.)
