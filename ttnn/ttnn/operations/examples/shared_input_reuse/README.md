# shared_input_reuse — read once, share many (shared-input DRAM-read amortization via multicast)

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept(s):** redundant-DRAM-read elimination via NoC multicast (the `mcast_pipe` helper)
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH · Blackhole · 2026-07-13

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A grid of worker cores all need the **same** multi-MB input to do their work — here a large shared
matrix `[R, C]` (≈2.4 MB), which is far larger than L1, so it is streamed through L1 in fixed-size
chunks (a common stream-through-L1 loop). The obvious way is to let *every* core read the whole stream
from DRAM. With `N` cores that is `N ×` the unique bytes — the same input fetched from DRAM `N` times —
and at scale the DRAM interface saturates.

## What this isolates — and how
- **Concept:** how the cores obtain a shared, re-read stream — each reads it from DRAM, vs. one core
  reads each chunk once and NoC-multicasts it to the rest (via the `mcast_pipe` `SenderPipe`/`ReceiverPipe`
  helper with a per-chunk semaphore handshake — the production forwarding pattern).
- **Isolation setup:** *DRAM-read / NoC-contention* row of the rule — the input is DRAM interleaved, the
  per-core job is trivial and identical in both variants, and only the read path differs. Geometry is a
  fixed **2 × grid_x** rectangle used by both variants, so core placement is not a hidden variable
  (contrast `reader_placement`, which isolates exactly *where* a fixed-work line sits).
- **The job (identical, both variants):** fold the whole streamed input into one running **tile-sum** using
  `add_tiles(acc_to_dest=true)` — a genuine **fp32** accumulation of **bf16** data (the running sum stays
  in the fp32 DEST adder and never round-trips through a bf16 Src register). Output is **one tile per
  core** — negligible next to the multi-MB read — so the kernel stays **read-bound**.
- **Why it's kernel-level:** read-per-core vs read-once-and-broadcast is a dataflow-kernel decision.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `per_core_dram` *(baseline)* | every core streams the whole input from DRAM itself | `N ×` the unique bytes; contends for the DRAM interface at scale |
| `mcast` | the top-left injector reads each chunk once and `mcast_pipe`-broadcasts it to the rest (self-excluded); receivers ack per chunk (double-buffered flow control) | DRAM sees the stream once; the copies travel core-to-core over the NoC |

`cb_in` is double-buffered (2 chunks) so the injector/reader fetches chunk `c+1` while compute folds `c`.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.shared_input_reuse [--chunk-rows 16] [--d-cols 4] [--chunks 19] [--trials 10]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--chunk-rows` | int | 16 | tile-rows per chunk |
| `--d-cols` | int | 4 | input width in tiles (tile-cols) |
| `--chunks` | int | 19 | number of chunks (rows / chunk-rows) |
| `--trials` | int | 10 | profiled rounds; report shows median |

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box.* Shared input = 19×16×4 = 1216
tiles ≈ 2.4 MB bf16, streamed in 19 chunks, on 22 cores (2×11).

```
shared_input_reuse  box=bh-qb-11…  arch=Blackhole  cores=22 (2x11)  injector=top-left
  variant          ns/op      vs per_core_dram
  per_core_dram    135415     1.00x
  mcast             78987     1.71x
```

**Reading of the result:** reading a shared stream from DRAM on every core is `N ×` redundant; reading
each chunk once on the injector and NoC-broadcasting it wins **1.71×** here at the realistic ~2.4 MB / 22-
core operating point. Note the device-time win is *smaller than the DRAM-read-count reduction* — reading
each chunk once on the injector instead of on every core cuts the per-tile DRAM read count by roughly the
multicast fan-out (~11× on this 22-core grid), but the device time is capped because the single injector
reads the whole stream serially and the bytes still cross the NoC (the mcast fan-out isn't free). It grows
with core count and with more concurrent injectors (one per independent stream in the real full-grid case).

## Correctness (both gates pass)
- **delivery** (full 2.4 MB stream): `mcast` output == `per_core_dram` output — proves the multicast hands
  every core the exact bytes the DRAM read would. (This is the meaningful gate; both do identical compute.)
- **structural** (small depth vs torch): the tile-sum matches the torch reference. All-ones input sums to
  the exact tile count (1216) — the fp32 accumulation does not saturate (a naive bf16-Src accumulator
  stalls at 256; `add_tiles(acc_to_dest)` keeps the sum in the fp32 DEST adder, so it's exact).

```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_delivery
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_structural
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_device_perf
```
