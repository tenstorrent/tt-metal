# shared_input_reuse — read once, share many (shared-input DRAM-read amortization via multicast)

**Problem type:** redundant reads  ·  **Concept(s):** redundant-DRAM-read elimination via NoC multicast — (1) a streamed shared input, chunk by chunk with a handshake (the `mcast_pipe` helper); (2) a one-shot broadcast of a small window every Tensix core keeps resident in L1 ([second situation](#second-situation--a-resident-window-one-shot-broadcast))
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH P150 · 2026-07-13
**Measured on:** WH B0 n150 (regimes + chunk sweep, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19) · BH p150b 13×10 grid (`bh-50-special-adamjanovic-for-reservation-143315`, 2026-10-30: streamed re-measure + resident window)

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
The streamed mode runs `test_shared_input_reuse_device_perf`, which rewrites `report.md`; set `SIR_REPORT=<path>` to
keep the committed report. For the resident window, see the [second situation](#cli-resident-window).

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
(16 Tensix cores), **between 8 and 16 tiles per chunk on BH p150b** (24 Tensix cores), **between 2 and 4 tiles per
chunk on the 13×10-grid BH p150b** (26 Tensix cores, slower baseline).

Same 1216-tile bf16 stream (d_cols = 4 tiles) in every row unless noted; a page is one 2048-byte tile; only
the chunk size changes. The worker rectangle is 2 × grid_x Tensix cores on every board, injector at (0,0).

| Regime (chunk size · `chunk_rows × d_cols × chunks`) | WH B0 n150 · 2×8 = 16 Tensix cores | BH p150b · 2×12 = 24 Tensix cores | BH P150 · 2×11 = 22 Tensix cores | BH P100a · 2×11 = 22 Tensix cores, 7 DRAM banks | BH p150b, 13×10 grid · 2×13 = 26 Tensix cores |
|---|---|---|---|---|---|
| **Does not help:** 1 tile (2 048 B) · `1 × 1 × 1216` | 632 497 → 905 974 ns **0.70×** | not measured | not measured | not measured | 542 976 → 844 086 ns **0.64×** |
| **Does not help:** 4 tiles (8 192 B) · `1 × 4 × 304` | 253 944 → 353 620 ns **0.72×** | 169 285 → 247 015 ns **0.69×** | not measured | not measured | 270 407 → 256 299 ns **1.06×** |
| **Does not help:** 8 tiles (16 384 B) · `2 × 4 × 152` | 214 339 (213 654–215 695) → 262 332 (262 167–262 390) ns **0.82×** | 148 567 (148 167–148 897) → 154 494 (154 410–154 698) ns **0.96×** | not measured | not measured | 242 398 → 161 552 ns **1.50× — helps here** |
| **Does not help:** 16 tiles (32 768 B) · `4 × 4 × 76` | 205 610 → 221 500 ns **0.93×** | 149 415 → 113 451 ns **1.32× — helps on BH** | not measured | not measured | 232 619 → 114 680 ns **2.03×** |
| **Boundary (WH):** 32 tiles (65 536 B) · `8 × 4 × 38` | 208 500 → 200 334 ns **1.04×** (flat) | 150 358 → 90 532 ns **1.66× — helps on BH** | not measured | not measured | 188 303 → 91 767 ns **2.05×** |
| **Helps:** 64 tiles (131 072 B) · `16 × 4 × 19` (CLI default) | 255 592 (251 866–258 115) → 191 246 (190 730–191 830) ns **1.34×** | 150 228 (149 554–150 601) → 77 777 (77 707–77 864) ns **1.93×** | 135 415 → 78 987 ns **1.71×** | 161 936 → 88 861 ns **1.82×** | 177 524 → 78 360 ns **2.27×** |
| **Helps:** 128 tiles (262 144 B) · `32 × 4 × 10` = 1280 tiles | 287 511 → 198 901 ns **1.45×** | 164 538 → 76 613 ns **2.15×** | not measured | not measured | not measured |
| **Boundary** (smallest chunk where `mcast` wins) | ~32 tiles per chunk | between 8 and 16 tiles per chunk | not measured | not measured | between 2 and 4 tiles per chunk |

Cells are `per_core_dram → mcast` DEVICE KERNEL DURATION (parentheses: min–max over the 10 windows). Stream length matters much less than chunk size
once chunks are large (WH B0 n150, 64-tile chunks: 1 chunk 1.17×, 4 chunks 1.23×, 19 chunks 1.34×,
40 chunks 1.33×; not measured on BH).

**Why the boards differ:**
- BH p150b with the 13×10 grid (2×13 = 26 Tensix cores): `mcast` is within 1–5% of the 24-core p150b, but the
  `per_core_dram` baseline is slower and no longer flat in chunk size (177.5 µs = 365 GB/s at 64-tile chunks,
  242 µs = 267 GB/s at 8-tile chunks; the 24-core board held ~400 GB/s). Each Tensix core has one chunk in flight
  per read barrier, and on this board that makes the baseline latency-bound, so the crossover drops to 2–4 tiles per chunk.
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
- BH p150b, 13×10 grid: the regimes test (64-tile helps, 16- and 8-tile `measure`, 1-tile does-not-help) plus 2-, 4-
  and 32-tile `test_shared_input_reuse_device_perf` probes (10 × 5 launches);
  `bh-50-special-adamjanovic-for-reservation-143315`, 2026-10-30.

The 8-tile regime (`measure_chunk8tiles_1216tiles`, formerly `does_not_help_chunk8tiles_1216tiles`) is log-only
since the 26-core p150b measured 1.50×; `does_not_help_chunk1tile_1216tiles` (1-tile chunks, `d_cols` = 1: WH B0
n150 0.70×, 26-core BH p150b 0.64×) carries the does-not-help assertion.

Reproduce (same regimes on any board; asserts only what held on every measured board, `measure` regimes only log):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_regimes
```

---

## Second situation — a resident window, one-shot broadcast

### The problem
Every Tensix core streams its **own** rows of a large input from DRAM and writes its own output rows (a DRAM-bound
read + write over the grid), and every Tensix core **also** needs the same small vector — a bias, a scale, a
window — resident in L1 for the whole run. Here: `out[r, c] = x[r, c] * w[c]`, x `[R, W]` split by tile-rows,
w `[1, W]`. Reading w on every Tensix core is `N ×` the window's bytes from the same DRAM pages. One Tensix core can
read it once and multicast it instead. Unlike the streamed case, the window's L1 is written once per launch, so the
multicast needs no "slot free" handshake.

### What this isolates — and how
- **Concept:** how w reaches each Tensix core's L1. x, the split, the compute (`mul_tiles_bcast_rows`, row 0 of w
  broadcast down each tile) and the writer are identical in every variant.
- **Setup:** x, w, out are `Layout::TILE` bf16 in DRAM, `TensorMemoryLayout::INTERLEAVED`. x's tile-rows go to a
  rectangle of Tensix cores (contiguous tile ids per core); the reader (NoC 0) and writer (NoC 1) move 8-tile
  blocks, 2 blocks deep. w is W/32 tiles (2048 bytes per tile, row 0 valid) in a circular buffer that is never
  popped. The window cannot be smaller than one 2048-byte tile in `Layout::TILE`.
- **Portable unit:** `r` = window bytes ÷ x bytes per Tensix core = 1 ÷ (tile-rows of x per Tensix core).

| Variant | What it does | Why it should differ |
|---|---|---|
| `per_core` *(baseline)* | every Tensix core reads all of w from DRAM on its reader, after its first x block | `N ×` the window from DRAM, competing with the stream |
| `oneshot_mcast` | the top-left Tensix core reads w in 32-tile chunks; after each chunk it multicasts the data and then a semaphore *set* (chunks landed) to the rectangle; receivers push each chunk to compute as it lands | DRAM sees the window once; no handshake |
| `lockstep_mcast` | the same multicast with an ack-then-broadcast handshake per chunk, and the whole window fetched before the first x block | adds one round trip per chunk |

<a id="cli-resident-window"></a>
### CLI
```bash
python -m ttnn.operations.examples.shared_input_reuse --window [--cores 16,32,64,full] \
    [--window-tiles 1,8,32,128,256] [--stream-mb 16,64] [--window-chunk 32] \
    [--variants per_core,oneshot_mcast,lockstep_mcast] [--repeats 3] [--trials 3]
```

| Flag | Default | Meaning |
|---|---|---|
| `--window` | off | run the resident-window sweep (`test_shared_input_reuse_window_sweep`) |
| `--cores` | `16,32,64,full` | Tensix cores (the squarest rectangle from (0,0); `full` = the whole compute grid) |
| `--window-tiles` | `1,8,32,128,256` | w width in tiles (2048 B each) = x width |
| `--stream-mb` | `16,64` | MiB of x (raised to one tile-row per Tensix core when smaller) |
| `--window-chunk` | 32 | window tiles per DRAM batch / multicast (and per handshake in `lockstep_mcast`) |
| `--variants` | all three | the first one is the baseline |
| `--repeats` / `--trials` | 3 / 3 | DRAM placements per point / interleaved windows (× 5 launches) per placement |

### Measured result — BH p150b, 13×10 = 130 Tensix cores
`bh-50-special-adamjanovic-for-reservation-143315`, 2026-10-30; median over 3 DRAM placements × 3 × 5 launches
(regimes: 10 × 5). Full sweep in [`report.md`](report.md).

| Regime | Tensix cores · window · x | r | `per_core` → `oneshot_mcast` |
|---|---|---|---|
| **Helps** (regime, assert ≥ 1.10×) | 130 · 128 tiles (256 KB) · 65 MiB | 0.5 | 437 387 → 364 948 ns **1.20×** |
| **Helps** | 130 · 128 tiles · 32.5 MiB | 1.0 | 271 336 → 179 437 ns **1.51×** |
| **Helps** | 130 · 32 tiles (64 KB) · 16 MiB | 0.5 | 111 065 → 91 589 ns **1.21×** |
| **Helps** | 64 · 128 tiles · 16 MiB | 1.0 | 139 440 → 103 124 ns **1.35×** |
| **Helps** | 32 · 128 tiles · 16 MiB | 0.5 | 130 046 → 110 874 ns **1.17×** |
| **Helps** | 16 · 256 tiles (512 KB) · 16 MiB | 0.5 | 187 858 → 152 786 ns **1.23×** |
| **Does not help** (regime, assert ≤ 1.05×) | 130 · 8 tiles (16 KB) · 16 MiB | 0.125 | 92 383 → 92 609 ns **1.00×** |
| **Does not help** (regime, assert ≤ 1.05×) | 32 · 128 tiles · 64 MiB | 0.125 | 442 016 → 442 592 ns **1.00×** (sweep 0.96×) |
| **Does not help** | 130 · 32 tiles · 64 MiB | 0.125 | 373 774 → 358 502 ns **1.04×** |
| **Does not help** | 64 · 1 tile (2 KB) · 16 MiB | 0.008 | 93 771 → 97 776 ns **0.96×** |
| **Does not help** (regime, `lockstep_mcast`, 1-tile chunks) | 130 · 32 tiles · 16 MiB | 0.5 | 124 342 → 141 805 ns **0.88×** |

**Rule:** broadcast the window once when it is at least half of each Tensix core's own x bytes (r ≥ 0.5):
1.21–1.51× on the full grid, 1.06–1.35× on 16–64 Tensix cores. At r ≤ 0.125 it gains nothing (0.96–1.07×,
whatever the core count), and windows ≤ 16 KB (8 tiles) never gain on ≥ 64 Tensix cores (0.96–1.03×). Use
large chunks; never add a per-chunk handshake.

**Why:**
- `per_core` adds one window of DRAM reads per Tensix core to that core's x read + out write, so the expected gain is
  ≈ 1 + r/2 while the stream is DRAM-bound (r = 1 → 1.5×; measured 1.40–1.51× on the full grid). The time saved matches
  the DRAM ceiling: 130 × 256 KB = 34 MB saved 77 µs (≈ 440 GB/s).
- A small Tensix-core count does not stop the lever by itself; a long stream does. 32 Tensix cores lose (0.96×) at
  8 tile-rows each and gain 1.17× at 2 tile-rows each with the same 128-tile window.
- Every Tensix core's first tile-row needs the whole window, so the stream waits for the full window in every
  variant. Waiting is therefore free (`lockstep_mcast` with one 32-tile chunk equals `oneshot_mcast`). What costs is
  each ack-then-broadcast round: ≈ 1.2–1.3 µs on 130 Tensix cores, so 32 handshakes cost +40 µs (0.88× of
  `per_core`) and 256 cost +300 µs (0.77×).
- Surprise, not a rule: on 16–32 Tensix cores with a 1-tile-wide x, `per_core` was 1.18–1.32× slower than
  `oneshot_mcast` although the window is 2 KB. A `per_core` build without the window read ran as fast as
  `oneshot_mcast`, and ±1 tile-row of stream swung the ratio 0.83–1.33×. The read shifts each core's start, and
  the latency-bound narrow stream amplifies it. This is not a byte-count effect.

### Correctness
`test_shared_input_reuse_window_correctness` checks all three variants against torch `x * w` (PCC ≥ 0.999 and
`allclose` at 2e-2): full grid with a 40-tile window (a partial chunk), 16 Tensix cores with a 1-tile window,
32 Tensix cores with a 256-tile window, and 64 Tensix cores with a 33-tile window in 8-tile chunks. Every split
is uneven, and each case runs two launches so the semaphores and window restart. Every sweep and regime point
also gates each variant's output before timing it.

```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_window_correctness
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_regimes
```
