# transfer_alignment — the hidden NoC alignment-residue tax on sub-page span reads

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** NoC read alignment-residue congruence; the over-read + local L1 realign a non-congruent sub-page read forces
**First profiled on:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181` · WH · ~1000 MHz · 2026-07-22 · `b54bcdedf74`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A reader extracts many `width`-byte row-spans from a DRAM tensor, each starting at an
arbitrary byte offset inside its row, straight into a circular buffer. A `noc_async_read`
can only move a range when the **source byte address and the destination byte address are
congruent modulo the alignment window** (the DRAM/L1 alignment granule — 32 B DRAM on this
box, queried at runtime, never hard-coded). When the span start's residue does not match the
destination residue, you **cannot express the read as one transfer**: you must round the
source down to an alignment boundary, over-read `width + residue` bytes into an aligned
scratch buffer, then do a local L1 realign pass to move the useful `width` bytes into place.
The subtle part is that most authors never notice they are on this path — the kernel silently
pays an over-read plus a per-span L1 copy on every span.

## What this isolates — and how
- **Concept:** the alignment-residue congruence requirement of a NoC read, and the over-read
  + local realign a non-congruent sub-page read forces — versus the direct read you get for free
  when the residues match.
- **Isolation setup:** *DRAM-read efficiency* row of the isolation rule. Compute is identity
  (there is no compute kernel); the writer is byte-identical across both variants (the reader
  always lands the useful span at a residue-0 CB offset, so the span→DRAM write is congruent
  either way); single core, same placement; `width`, span count, page size, and CB depth are
  all held constant. The **only** difference is whether the span start is alignment-congruent —
  so the measured delta is purely the read strategy.
- **Why it's kernel-level:** the kernel author chooses the span offset and the CB write-pointer.
  Arranging them so `(src_off % align) == (dst_off % align)` is a decision made in the kernel,
  not a model or dtype choice.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `misaligned` *(baseline)* | Span start has a non-zero residue. Rounds the source down to the alignment boundary, over-reads `width + residue` bytes into an aligned scratch CB, then `memmove`s the useful `width` bytes into the destination. | An over-read plus one extra per-span CPU pass over L1 — the alignment-residue tax. |
| `aligned` | Span start arranged congruent with the destination residue. One direct `noc_async_read` of exactly `width` bytes; no scratch, no realign. | The whole transfer stays a single NoC read on the fast datamover — no second pass. |

Both extract a correct sub-page span (at their own, per-variant offset); a constant-source
control asserts the two paths produce byte-identical output.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.transfer_alignment [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,aligned,misaligned}` | `all` | which method(s) to run and compare |
| `--trials` | int | `10` | measured trials; report shows the average |
| `--iters` | int | `1` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | the pytest report is written to `report.md` |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--width` | int (bytes), comma-list | `64,256,1024,4096` | span payload bytes per row (multiple of 2) |
| `--spans` | int, comma-list | `16,64,256` | number of spans / rows `N` |
| `--align` | `{aligned,misaligned}` | *(both)* | run just one variant |
| `--cores` | int | `1` | cores running the (independent) pipeline |

**Example invocations:**
```bash
# A/B both methods across the default width x count grid
python -m ttnn.operations.examples.transfer_alignment

# one width, sweep the span count, steady-state
python -m ttnn.operations.examples.transfer_alignment --width 1024 --spans 16,64,256 --iters 50

# just the dodge, single-shot latency
python -m ttnn.operations.examples.transfer_alignment --align aligned --width 4096
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

```
transfer_alignment  box=bgd-lab-...33181  arch=WH  clock=~1000MHz  N=10 (avg)  iters=1 (per-launch latency)
  cores=1  placement=single core  align_window=32B  residue=16B
    N=16   width=  64B   misaligned  12024.5 ns   aligned   7096.9 ns  ✓  → 1.69×
    N=16   width=1024B   misaligned  55025.7 ns   aligned   8137.8 ns  ✓  → 6.76×
    N=16   width=4096B   misaligned 192829.0 ns   aligned  10529.9 ns  ✓  → 18.31×
    N=256  width=4096B   misaligned 3075847.6 ns  aligned 161684.7 ns  ✓  → 19.02×
```

**Reading of the result:** the dodge wins everywhere (1.7×–19×), and — counter to the naive
"the over-read amortizes with width" intuition — the win **grows** with span width. The tax is
not the over-read (only +16 B, negligible even at 4 KB); it is the forced CPU `memmove` that
realigns the useful bytes out of the aligned scratch. That copy runs off the NoC on a scalar
RISC and scales with `width`, while the direct read is a single NoC transfer nearly independent
of `width` in the single-core latency regime — so the gap widens as spans grow. Across `N` the
ratio is flat (both variants scale linearly with independent per-span work); the absolute ns
saved grows with `N` because the per-span realign is paid `N` times. Dodge it by arranging the
span offset / CB write-pointer so `(src_off % align) == (dst_off % align)`.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_transfer_alignment.py
```
