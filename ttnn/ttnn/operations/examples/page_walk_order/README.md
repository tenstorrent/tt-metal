# page_walk_order — the order one reader walks its DRAM page indices

**Difficulty:** ⭐ T1  ·  **Concept(s):** page-index walk order → DRAM bank-level parallelism (a temporal access-order decision, independent of core placement)
**First profiled on:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181` · WH (wormhole_b0) · ~1000 MHz · 2026-07-22 · `6041ca1350b`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You have one reader core streaming N pages from an interleaved-DRAM tensor, and an index
mapping makes it walk the page indices with a large constant stride — the author picked a
loop order without noticing the stride collides with the DRAM bank count. It is mysteriously
slower than a plain contiguous read of the same pages. Interleaved DRAM round-robins
consecutive page indices across the banks: page `p` lives in bank `p % num_banks`. So a
**unit-stride** walk (`p, p+1, p+2, …`) sends consecutive reads to *different* banks, while a
stride equal to `num_banks` (or a multiple) sends every read to the *same* bank — serializing
them at that one bank's controller with no cross-bank parallelism.

## What this isolates — and how
- **Concept:** the temporal ORDER in which a single reader walks its source page indices, and
  how that order maps onto DRAM banks. Nothing about *where* the core sits — this is purely
  an access-order decision the kernel author makes.
- **Isolation setup:** the *DRAM-read* row of the isolation rule, **read-isolation**. The
  source is interleaved DRAM; a **single reader core** walks every page once and computes a
  **negligible** per-page checksum (one halfword per page, a handful of adds), so the measured
  time is the READ walk, not the compute. Page count, page size, block (reads-per-barrier),
  and core are held identical across variants — **the only difference is the walk stride.**
  Every variant reads the identical set of pages, so the checksum is identical.
- **Why it's kernel-level:** the stride between consecutive page reads is a loop-order choice
  the kernel author controls. It is not a model/dtype/shape decision — all of those are fixed.

## The methods being compared
The stride for each named walk is derived at runtime from the **queried** bank count
(`num_dram_banks`, 12 on this box) — never hard-coded.

| Variant | Stride | What it does | Why it should differ |
|---|---:|---|---|
| `bank_stride` *(baseline / the trap)* | `num_banks` (12) | Every read in a block lands on the **same** bank. | That bank serializes them; no cross-bank parallelism, poor row-buffer locality. |
| `unit_stride` | `1` | Consecutive reads march across all banks. | Up to `num_banks`-way bank parallelism + row-buffer locality. |
| `coprime_stride` | `num_banks + 1` (13) | Coprime to the bank count → bank index steps by 1 (mod banks). | Also touches all banks — isolates **spread** from contiguity. |

The walk is a general coset enumeration `idx = (base + k·stride) mod N` over `gcd(stride, N)`
cosets, so it visits every page exactly once for any stride — the read multiset (and checksum)
is stride-independent; only the order changes.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.page_walk_order [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,bank_stride,unit_stride,coprime_stride}` | `all` | which named walk order(s) to run/compare |
| `--trials` | int | `20` | measured trials; report shows the average ns/op |
| `--iters` | int | `1` | in-kernel repeat of the full walk — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | pytest writes the committed report; the CLI prints the table |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--pages` | int | `1536` | requested page count N (rounded up to a multiple of the queried bank count) |
| `--page-size` | int (bf16 elems) | `1024` | elements per page; page bytes = 2× this (multiple of 16) |
| `--strides` | `auto` or comma list | `auto` | `auto` = the named variants' strides from the bank count; or explicit integer strides |
| `--block` | int | `0` (auto = 2×banks) | reads issued per barrier — how many are kept in flight |

**Example invocations:**
```bash
# A/B all three walk orders on 4 KB pages (bigger transactions -> bigger gap)
python -m ttnn.operations.examples.page_walk_order --page-size 2048

# custom raw strides on your own page count
python -m ttnn.operations.examples.page_walk_order --pages 3072 --strides 1,12,24,13
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

```
page_walk_order   box=bgd-lab-t3002...  arch=wormhole_b0   trials=20 (avg ns/op)   iters=1 (per-launch)
  num_banks=12  pages=1536  page_bytes=2048  block=24  cores=1  placement=single core
    walk order        stride        ns/op   read GB/s   vs baseline
    bank_stride           12     178639.0     17.61     (baseline)
    unit_stride            1     141767.0     22.19     → 1.26×
    coprime_stride        13     141781.0     22.19     → 1.26×
```

**Reading of the result:** the trap is real but **bounded**: spreading the walk across banks
is ~1.26–1.31× faster (2–4 KB pages), not the ~12× the bank count might suggest. A single
reader core is limited by NCRISC transaction-issue rate and one NoC port — it tops out around
26 GB/s (vs ~200 GB/s DRAM), so it can never keep all 12 banks busy; the walk order only sets
how efficiently that limited concurrency is spread across banks. Two things sharpen the story:
`coprime_stride` ≡ `unit_stride` (so it is bank **spread**, not contiguity, that matters), and
the gap **grows with page size** — from ~1.0× at 512 B (issue-bound, bank irrelevant) to 1.31×
at 4 KB (DRAM-service-bound, same-bank serialization bites). See `report.md` for the full
page-size and block sweeps.

## Run the predefined sweep (regenerates report.md)
```bash
scripts/run_safe_pytest.sh ttnn/ttnn/operations/examples/page_walk_order/test_page_walk_order.py
```
(Correctness: `test_page_walk_order_correctness`; measurement: `test_page_walk_order_device_perf`.)
