# shared_input_reuse — read once, share many (shared-input DRAM-read amortization via multicast)

**Problem type:** redundant reads  ·  **Concept(s):** redundant-DRAM-read elimination via NoC multicast (the `mcast_pipe` helper)
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH P150 · 2026-07-13
**Measured on:** WH B0 n150 (regimes + chunk sweep, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

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
  (contrast `noc_placement`, which isolates exactly *where* a fixed-work line sits).
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
*Illustrative — re-run the CLI for your box.* Shared input = 19×16×4 = 1216 tiles ≈ 2.4 MB bf16,
streamed in 19 chunks, on 22 cores (2×11).

### P150 reference

The original run did not record busy AICLK, enabled GDDR count, or firmware.

```
shared_input_reuse  box=bh-qb-11…  arch=Blackhole  cores=22 (2x11)  injector=top-left
  variant          ns/op      vs per_core_dram
  per_core_dram    135415     1.00x
  mcast             78987     1.71x
```

### P100a, seven enabled GDDR banks

`bh-43-special-sjovic-for-reservation-97381` · 1350 MHz busy AICLK · firmware 19.12.0 · 2026-09-23

```
shared_input_reuse  box=bh-43-...  arch=Blackhole  cores=22 (2x11)  injector=top-left
  variant          ns/op      vs per_core_dram
  per_core_dram    161936     1.00x
  mcast             88861     1.82x
```

**Reading of the result:** reading a shared stream from DRAM on every core is `N ×` redundant; reading
each chunk once on the injector and NoC-broadcasting it wins **1.71× on P150 and 1.82× on this
seven-bank P100a** at the realistic ~2.4 MB / 22-core operating point. The P100a's redundant-DRAM
baseline is 19.6% slower than the P150 result, while the multicast path is 12.5% slower. Note the
device-time win is *smaller than the DRAM-read-count reduction* — reading
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

## When it helps / when it does not
**Rule:** multicast the shared stream only when each chunk (one multicast, one handshake) is large enough to
amortize the per-chunk ack-and-broadcast round trip; crossover **~32 tiles per chunk on WH B0 n150**
(16 Tensix cores), **between 8 and 16 tiles per chunk on BH p150b** (24 Tensix cores).

Same 1216-tile bf16 stream (d_cols = 4 tiles) in every row unless noted; a page is one 2048-byte tile; only
the chunk size changes. The worker rectangle is 2 × grid_x Tensix cores on every board, injector at (0,0).

| Regime (chunk size · `chunk_rows × d_cols × chunks`) | WH B0 n150 · 2×8 = 16 Tensix cores | BH p150b · 2×12 = 24 Tensix cores | BH P150 · 2×11 = 22 Tensix cores | BH P100a · 2×11 = 22 Tensix cores, 7 DRAM banks |
|---|---|---|---|---|
| **Does not help:** 4 tiles (8 192 B) · `1 × 4 × 304` | 253 944 → 353 620 ns **0.72×** | 169 285 → 247 015 ns **0.69×** | not measured | not measured |
| **Does not help:** 8 tiles (16 384 B) · `2 × 4 × 152` | 214 339 (213 654–215 695) → 262 332 (262 167–262 390) ns **0.82×** | 148 567 (148 167–148 897) → 154 494 (154 410–154 698) ns **0.96×** | not measured | not measured |
| **Does not help:** 16 tiles (32 768 B) · `4 × 4 × 76` | 205 610 → 221 500 ns **0.93×** | 149 415 → 113 451 ns **1.32× — helps on BH** | not measured | not measured |
| **Boundary (WH):** 32 tiles (65 536 B) · `8 × 4 × 38` | 208 500 → 200 334 ns **1.04×** (flat) | 150 358 → 90 532 ns **1.66× — helps on BH** | not measured | not measured |
| **Helps:** 64 tiles (131 072 B) · `16 × 4 × 19` (CLI default) | 255 592 (251 866–258 115) → 191 246 (190 730–191 830) ns **1.34×** | 150 228 (149 554–150 601) → 77 777 (77 707–77 864) ns **1.93×** | 135 415 → 78 987 ns **1.71×** | 161 936 → 88 861 ns **1.82×** |
| **Helps:** 128 tiles (262 144 B) · `32 × 4 × 10` = 1280 tiles | 287 511 → 198 901 ns **1.45×** | 164 538 → 76 613 ns **2.15×** | not measured | not measured |
| **Boundary** (smallest chunk where `mcast` wins) | ~32 tiles per chunk | between 8 and 16 tiles per chunk | not measured | not measured |

Cells are `per_core_dram → mcast` DEVICE KERNEL DURATION (parentheses: min–max over the 10 windows). Stream length matters much less than chunk size
once chunks are large (WH B0 n150, 64-tile chunks: 1 chunk 1.17×, 4 chunks 1.23×, 19 chunks 1.34×,
40 chunks 1.33×; not measured on BH).

**Why the boards differ:**
- The per-chunk handshake costs the same on both boards: the `mcast` time grows by ~0.5–0.65 µs per extra
  chunk between 4- and 64-tile chunks on WH B0 n150 and on BH p150b alike (slope of the sweep).
- The `per_core_dram` baseline is DRAM-bound on both boards, but BH p150b's DRAM ceiling is ~2× higher:
  24 readers × 2.49 MB (1216 tiles × 2048 B = 2 490 368 B) = 59.8 MB in ~150 µs ≈ 400 GB/s
  (BH p150b ceiling ~415–430 GB/s, 8 DRAM banks) vs 16 × 2.49 MB = 39.8 MB in 205–256 µs ≈ 155–195 GB/s (WH B0 ceiling ~190 GB/s, 12 DRAM banks).
- The single injector's stream is ~2.7× faster on BH p150b (estimated from the sweep after subtracting the
  handshake: ~65 µs vs ~180 µs for the 2.49 MB stream, ~38 vs ~14 GB/s; 64-byte vs 32-byte NoC flit,
  1350 vs 1000 MHz AICLK). So BH has ~85 µs of handshake budget under the baseline (~150 chunks → 8-tile
  chunks), WH ~25 µs (~45 chunks → ~27-tile chunks): the crossover moves from ~32 tiles to 8–16 tiles.

- **Helps** when each chunk is large: the per-chunk handshake (injector waits for all N − 1 receiver acks,
  then broadcasts) is amortized, and DRAM sees 1/N of the reads (N = Tensix cores in the rectangle).
- **Does not help** when chunks are small: the fixed per-chunk round trip is paid once per chunk and exceeds
  the DRAM time saved, so `mcast` is slower than every Tensix core reading for itself. If you control the
  chunk size, enlarge it (CB depth and L1 permitting) before choosing between the variants.
- The gain grows with the number of Tensix cores sharing the stream (and with the board's single-injector
  stream rate): at 64-tile chunks 1.34× with 16 (WH B0 n150), 1.71–1.82× with 22 (BH P150 / P100a), 1.93×
  with 24 (BH p150b).

*How measured:*
- WH B0 n150: `test_shared_input_reuse_regimes` (helps 64-tile / does-not-help 8-tile rows) plus a
  same-stream chunk sweep with the same code; bf16; median of 10 interleaved windows × 5 launches; 2026-10-12.
  Ranges in [`report.md`](report.md).
- BH p150b: the same regimes test (64-tile helps, 16-tile `measure`, 8-tile does-not-help) plus the 4-, 32- and
  128-tile points from a probe with the same measurement code (10 × 5 launches);
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19.
- BH P150 / P100a: `test_shared_input_reuse_device_perf` (64-tile chunks only), boxes and dates under
  *Measured result* above.

Reproduce (same regimes on any board; asserts only what held on every measured board, `measure` regimes only log):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_regimes
```
