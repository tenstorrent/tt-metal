# width_split — filling the grid on a wide, short tensor

**Problem type:** grid not filled  ·  **Concept(s):** work distribution — splitting along width vs. height; for a per-row reduction, the cross-core combine that split costs (and, on `Layout::ROW_MAJOR` sticks, the k-fold DRAM transfer count of splitting each stick)
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` · WH B0 · 8×8=64 grid · 2026-07-24 · `6f13238bd69`
**Measured on:** WH B0 n150 (`bgd-lab-16-special-dstoiljkovic-for-reservation-44175`, 2026-07-24; regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19) · reduce case: BH p150b 13×10 (`bh-50-special-adamjanovic-for-reservation-143315`, 2026-10-29; `Layout::ROW_MAJOR` reduce case 2026-10-30)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You have a **wide, short** tensor — few tile-rows, many tile-columns (extreme case: a single
tile-row tall, e.g. `[32, 8192]`). The obvious way to spread a tile op over the grid is to give
each core some tile-**rows**. But a one-tile-row-tall tensor has exactly one row to hand out, so
*one* core does all the work and the other 63 sit idle. The op then runs at single-core speed no
matter how wide the tensor is — and wide-short tensors are common (folded activations, flat feature
maps). The fix is to distribute along the **width** instead.

## What this isolates — and how
- **Concept:** work distribution — parallelizing a wide-short tensor along its **width**
  (tile-columns) instead of its height (tile-rows).
- **Isolation setup:** *work distribution / multi-core grid* — compute is a trivial per-tile op
  (one relu), inputs and outputs are interleaved DRAM, and the tensor is fixed at **one tile-row
  tall (H=32)**. The reader/compute/writer kernels are byte-identical across variants; only the
  per-core `(start_page, num_pages)` — and therefore how many cores run — changes. So the measured
  delta is purely *how the Wt tiles are spread across cores*, nothing else.
- **Why it's kernel-level:** how you assign work-units to cores (the core grid + per-core runtime
  args) is a decision the kernel/program author makes, not a model choice.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `single_core` *(baseline)* | all `Wt` tiles on ONE core | what a tile-row split degenerates to for a 1-tile-row tensor — no width parallelism, runs serially |
| `width_split` | the `Wt` tiles spread contiguously across `min(Wt, grid)` cores (each core owns a `WT_CHUNK`-bounded column range) | fills the whole compute grid; per-core work drops to `Wt / cores`, so it should be ~`cores`× faster until the grid saturates |
| `row_split_reduce` *(baseline, reduce case)* | `y = x − mean_W(x)` per row; whole tile-rows per Tensix core over `min(R_t, grid)` Tensix cores | no cross-core traffic; but only `R_t` Tensix cores work, each serially over `Wt` tiles |
| `width_split_reduce` | each tile-row split into `k` slices of `Wt/k` tiles over a group of `k` neighbouring Tensix cores; the group exchanges 1-tile partials once per row (symmetric all-gather), then each member subtracts the mean from its own slice | `k`× more Tensix cores and `k`× less work each, minus one combine round per row (~0.5 µs at `k = 2`, growing with `k`) |
| same two, `Layout::ROW_MAJOR` input/output | pages are sticks of `W` bf16 datums (2·W bytes); each Tensix core reads its 2·W/k byte piece of each of a tile-row's 32 sticks, tilizes, reduces/combines/applies, untilizes, writes 32 pieces back | as above, but a Tensix core still issues 32 reads + 32 writes per tile-row at any `k`: the split multiplies the grid's DRAM transfer count by `k` and shrinks each transfer by `k` |

## Measured result (WH B0, 64-core grid, bf16, H=32)
| W | Wt | single_core (ns) | width_split (ns) | cores | speedup |
|--:|--:|--:|--:|--:|--:|
| 32 | 1 | 1817 | 1822 | 1 | **1.00×** (nothing to split) |
| 256 | 8 | 4292 | 1919 | 8 | **2.24×** |
| 1024 | 32 | 9613 | 2263 | 32 | **4.25×** |
| 2048 | 64 | 16655 | 2666 | 64 | **6.25×** |
| 4096 | 128 | 30885 | 4203 | 64 | **7.35×** |
| 8192 | 256 | 59261 | 7637 | 64 | **7.76×** |

**On WH B0 n150 the crossover is at `Wt=2`–`3`:** with a single tile-column there's nothing to spread, so
`width_split` correctly matches the baseline (both use 1 Tensix core); at `Wt=2` it is still flat on WH B0 (0.89–0.99×),
while on BH p150b `Wt=2` already gains 1.18× (see [When it helps / when it does not](#when-it-helps--when-it-does-not)). From `Wt=8` up it scales with the grid — the
speedup tracks the number of cores it can fill (8→2.24×, 32→4.25×, 64→6.25×) — and past the 64-core
saturation point the gap keeps widening because `single_core` grows linearly with width while
`width_split` grows at 1/64 the rate. Full table + method in [`report.md`](report.md).

## CLI — measure your own widths
```bash
python -m ttnn.operations.examples.width_split [--widths 32,256,1024,2048,4096,8192]
                                               [--variant all|single_core|width_split]
                                               [--dtype bfloat16|float32|bfloat8_b]
                                               [--iters K] [--trials N]
```
`H` is fixed at 32 (one tile-row — the wide-short case); only the width sweeps. `--iters 1` measures
per-launch latency; large `--iters` measures steady-state throughput.

Reduce case (per-row mean subtracted from the row):
```bash
python -m ttnn.operations.examples.width_split --reduce [--rows 1,16,64,128] [--wts 4,8,16,32,64]
                                               [--ks 1,2,4,8,16,32,64] [--max-oversub 2]
                                               [--iters K] [--trials N]
```
Add `--layout row_major` for the same op on `Layout::ROW_MAJOR` sticks (defaults `--rows 1,4,16,47,64,128
--wts 2,4,8,16,32,64 --ks 1,2,4`; the table adds stick bytes, piece bytes per DRAM transfer, the transfer count
and transfers per second).
`--rows` is the tile-row count `R_t`, `--wts` the tile-columns `Wt`, `--ks` the split factors (a `k` is
kept when it divides `Wt` and `R_t · k ≤ max-oversub × grid`). The table prints Tensix cores, tiles per
slice, ns, DRAM GB/s (input + output bytes ÷ time) and the speedup vs `k = 1`.

## Takeaway for your own kernels
If a tensor is **wider than it is tall in tiles** (`Wt > nt_h`, and especially `nt_h` small), do not
split work by tile-rows — you will strand it on `nt_h` cores. Split along the width (assign each
core a contiguous tile-column range, capped by a `WT_CHUNK` constant so per-core L1 stays bounded)
so the whole grid runs in parallel. The benefit grows with width, up to ~grid× at saturation; it
does not pay for `Wt ≤ 2` tiles on WH B0 n150 and for `Wt = 1` tile on BH p150b (see below — the earlier
WH claim of "only `Wt ≤ 1`" was corrected by measurement there, and holds on BH p150b).
**If the op reduces along the row** (norm, softmax, centering), the split also costs a per-row combine
between the Tensix cores sharing the row: it pays only while the tile-rows leave most of the grid idle and
each slice keeps ≥ 4 tiles — see [Reduce case](#reduce-case--when-the-row-needs-a-cross-core-combine).
On **`Layout::ROW_MAJOR` sticks** the split also cuts every stick into `k` smaller DRAM transfers without
reducing any Tensix core's transfer count; it needs ≥ 1–2 KB sticks and loses or is flat (0.38–1.08×) once the tile-rows
cover ~1/3 of the grid — see [ROW_MAJOR](#reduce-case-on-row_major-sticks--splitting-a-stick-multiplies-the-transfers).

## When it helps / when it does not
**Rule:** split along the width once the single Tensix core's baseline work is more than the launch floor can
hide — from **Wt ≥ 3 tiles on WH B0 n150** and from **Wt ≥ 2 tiles on BH p150b**; at Wt = 1 tile there is nothing
to split on either board. bf16, H = 32 (one tile-row), interleaved DRAM (`TensorMemoryLayout::INTERLEAVED`) in
and out; baseline = `single_core`, lever = `width_split` (1 tile per Tensix core up to the grid).

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** Wt = 64 tiles → 64 Tensix cores × 1 tile `[32, 2048]` | 16 832 → 2 720 ns **6.19×** | 11 092 → 1 536 ns **7.22×** |
| **Does not help:** Wt = 2 tiles → 2 Tensix cores × 1 tile `[32, 64]` | 2 076 → 2 105 ns **0.99×** (runs 0.89–0.99×) | 1 217 → 1 031 ns **1.18× — helps on BH** |
| **Does not help:** Wt = 1 tile, both variants on 1 Tensix core `[32, 32]` | 1 824 → 1 828 ns **1.00×** | 1 025 → 1 010 ns **1.02×** |
| Sweep: Wt = 3 tiles `[32, 96]` | 2 513 → 1 934 ns **1.30×** | **1.36×** |
| Sweep: Wt = 4 tiles `[32, 128]` | 2 873 → 1 966 ns **1.46×** | **1.54×** |
| Sweep: Wt = 8 tiles `[32, 256]` | 4 305 → 1 932 ns **2.23×** | 2 480 → 1 145 ns **2.17×** |
| Sweep: Wt = 32 tiles `[32, 1024]` | 9 613 → 2 263 ns **4.25×** | 6 160 → 1 240 ns **4.97×** |
| Sweep: Wt = 128 tiles `[32, 4096]` | 64 Tensix cores × 2 tiles: 30 885 → 4 203 ns **7.35×** | 120 Tensix cores: 20 886 → 1 867 ns **11.19×** |
| Sweep: Wt = 256 tiles `[32, 8192]` | 64 Tensix cores × 4 tiles: 59 261 → 7 637 ns **7.76×** | 120 Tensix cores: 40 518 → 2 961 ns **13.68×** |
| **Boundary** | Wt ≈ 3 tiles | Wt = 2 tiles |

**Why the boards differ:**
- **Launch floor.** A 1-tile launch costs ~1.8 µs on WH B0 n150 and ~1.0 µs on BH p150b (AICLK 1000 vs
  1350 MHz). The second tile adds ~280 ns to `single_core` on WH B0 and ~190 ns on BH p150b; on WH B0 the
  second Tensix core's own startup cancels that saving, on BH p150b it does not, so the split pays one
  tile earlier.
- **Grid size.** Past Wt = 64 tiles WH B0 n150 has no more Tensix cores and each one takes 2–4 tiles (7.35–7.76×),
  while BH p150b spreads up to 120 Tensix cores and keeps scaling (11.19× at Wt = 128 tiles, 13.68× at 256 tiles).

- **Helps:** `single_core` reads, relus and writes every tile serially (~235 ns per tile on WH B0 n150,
  ~160 ns on BH p150b, on top of the launch floor), while `width_split` gives every Tensix core one tile, so
  the device time collapses to about one tile's round trip plus the launch floor.
- **Does not help (WH B0 n150, Wt = 2 tiles):** the single Tensix core moves both tiles in **one** 8-tile read
  block (one NoC barrier), ~2.1 µs of which ~1.8 µs is the launch floor; a second Tensix core cannot shorten
  that floor and only adds its own startup, so the split is flat to ~10% slower.
- **Limitation:** the example's API fixes H at one tile-row (`nt_h = 1`), so the "tile-row split vs width
  split as `nt_h` grows toward `Wt`" crossover cannot be measured here; the portable rule is about **tiles per
  Tensix core**: the split pays once the busiest Tensix core's baseline work is ≳3 tiles above the launch
  floor on WH B0 n150, ≳2 tiles on BH p150b.

*How measured:* `DEVICE KERNEL DURATION [ns]`, baseline → lever.
- **WH B0 n150:** Wt = 64 and Wt = 2 tiles from `test_width_split_regimes` (median of 12 trials
  per variant, a trial = mean of 5 launches, variants interleaved),
  2026-10-12; Wt = 1, 3, 4, 8 tiles from that day's `test_width_split_device_perf` crossover sweep, Wt = 32, 128,
  256 tiles from the 2026-07-24 sweep on `bgd-lab-16-…` (mean of 20 launches each; [`report.md`](report.md)).
- **BH p150b:** Wt = 64 and Wt = 2 tiles from `test_width_split_regimes` (same method; Wt = 2
  re-measured 1.17× in a probe); Wt = 1, 8, 32, 128, 256 tiles from `test_width_split_device_perf` (mean of 20
  launches); Wt = 3, 4 tiles from a probe of the same interleaved A/B (ratios only).
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).


## Reduce case — when the row needs a cross-core combine

The elementwise numbers above are the best case: a width split costs nothing because no tile needs
another Tensix core's data. A **per-row reduction followed by a per-row apply** (a norm, a softmax, a
centering) is different: the `k` Tensix cores that share a tile-row each hold only a partial, so every row
pays a **combine round** before any of them can apply the result. This case measures that trade.

**The op:** `y = x − mean_W(x)` per row, bf16 `Layout::TILE`, `TensorMemoryLayout::INTERLEAVED` DRAM in
and out, shape `[32·R_t, 32·Wt]`. Compute is one HiFi4 `REDUCE_ROW` (scaler `1/W`, fp32 DST) per tile and one
`sub_tiles_bcast_cols` per tile; the slice stays in its CB between the two (no DRAM re-read). Same three
kernels for both variants (`kernels/ws_reduce_*.cpp`); `k` is a runtime arg and `k = 1` skips the combine.

**The combine (what was chosen and why):**
- **Symmetric all-gather, one round.** Member `q` of a `k`-member group writes its partial into slot `q`
  of every member's gather CB (its own included) and bumps every member's semaphore; each member waits for
  `k` arrivals, sums the `k` slots in member order, and applies. No root, no second (send-the-result-back)
  round; for `k = 2` it is a plain swap. Every member sums the same tiles in the same order, so all slices
  of a row subtract a bit-identical mean.
- **Partials as small as the tile format allows.** A `REDUCE_ROW` result is column 0 of the tile, which
  lives in faces 0 and 2; only those two 16×16 faces are sent (**2 writes × 512 B = 1 KB** per partial per
  member, half the 2048 B bf16 tile). Measured alternative: one 1536 B write of faces 0–2 is within ~1% at
  `k ≤ 4` but **14% slower at `k = 16` and 27% slower at `k = 64`** — each member ingests `k` partials, so
  bytes per partial matter once `k` is large.
- **Signal after flush, not barrier.** The data writes and the semaphore increment go to the same
  destination on the same NoC and unicast-write VC, which delivers them in order; flushing (writes left
  this Tensix core) before the increment is enough.
- **Sum in pairs with the FPU accumulating into DST** (`add_tiles` with `acc_to_dest`). A dest-reuse add
  moves the running sum DST → SrcA every step and truncates it to SrcA's 19-bit format: a bias of ~1 ulp per
  add that reached 0.07 absolute error at `k = 32`.
- Groups are `k` consecutive Tensix cores in row-major grid order (neighbours; a few groups wrap a grid row).

### Measured (BH p150b, 13×10 = 130 Tensix cores, bf16, mean of 20 launches)

Speedup of `width_split_reduce` over `row_split_reduce` (`k = 1`); in brackets: Tensix cores, tiles per slice.

| R_t (tile-rows) | Wt | `k = 1` ns (Tensix cores) | k = 2 | k = 4 | k = 8 | k = 16 | k = 32 | k = 64 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4 | 2 083 (1) | 0.91× (2, 2) | 0.86× (4, 1) | | | | |
| 1 | 8 | 3 018 (1) | **1.17×** (2, 4) | 1.17× (4, 2) | 0.97× (8, 1) | | | |
| 1 | 16 | 4 900 (1) | 1.39× (2, 8) | **1.71×** (4, 4) | 1.48× (8, 2) | 0.89× (16, 1) | | |
| 1 | 32 | 8 567 (1) | 1.61× (2, 16) | 2.18× (4, 8) | **2.40×** (8, 4) | 1.50× (16, 2) | 0.85× (32, 1) | |
| 1 | 64 | 16 004 (1) | 1.77× (2, 32) | 2.77× (4, 16) | **3.45×** (8, 8) | 2.63× (16, 4) | 1.54× (32, 2) | 0.82× (64, 1) |
| 16 | 4 | 2 543 (16) | 0.95× (32, 2) | 0.89× (64, 1) | | | | |
| 16 | 8 | 3 742 (16) | **1.13×** (32, 4) | 1.10× (64, 2) | 0.91× (128, 1) | | | |
| 16 | 16 | 5 869 (16) | 1.11× (32, 8) | **1.19×** (64, 4) | 1.14× (128, 2) | 0.52× (128, 1)¹ | | |
| 16 | 32 | 9 674 (16) | **1.21×** (32, 16) | 1.19× (64, 8) | 1.16× (128, 4) | 0.77× (128, 2)¹ | | |
| 16 | 64 | 17 834 (16) | **1.28×** (32, 32) | 1.24× (64, 16) | 1.20× (128, 8) | 1.00× (128, 4)¹ | | |
| 64 | 4 | 3 659 (64) | 0.81× (128, 2) | 0.69× (128, 1)¹ | | | | |
| 64 | 8 | 7 217 (64) | 1.03× (128, 4) | 1.05× (128, 2)¹ | | | | |
| 64 | 16 | 13 534 (64) | 0.98× (128, 8) | 1.03× (128, 4)¹ | | | | |
| 64 | 32 | 25 725 (64) | 1.01× (128, 16) | 1.00× (128, 8)¹ | | | | |
| 64 | 64 | 50 875 (64) | 1.09× (128, 32) | 1.08× (128, 16)¹ | | | | |
| 128 | 4 / 8 / 16 / 32 / 64 | 7 049 / 14 123 / 25 310 / 46 022 / 90 295–122 388 (128) | 0.86× / 1.01× / 0.99× / 0.97× / 1.07× (130 Tensix cores, 1–2 tile-rows per group)¹ | | | | | |

¹ `R_t · k` > 130, so a group processes 2 tile-rows in turn (2 combines per Tensix core).

**Combine cost, isolated** (same 4 tiles per Tensix core, so the extra time is the combine):
`k = 1, Wt = 4` 2 083 ns → `k = 2, Wt = 8` 2 571 ns (**+0.49 µs**) → `k = 4, Wt = 16` 2 859 ns (+0.78 µs) →
`k = 8, Wt = 32` 3 567 ns (+1.48 µs) → `k = 16, Wt = 64` 6 089 ns (+4.0 µs). With 1-tile slices the combine
reaches ~10 µs at `k = 32` and ~19 µs at `k = 64`: every member receives `k` partials.

**DRAM rate of the baseline** (input + output bytes ÷ time): 8–16 GB/s at `R_t = 1`, 100–235 GB/s at
`R_t = 16`, **287–330 GB/s at `R_t = 64`** (64 of 130 Tensix cores), 274–365 GB/s at `R_t = 128` — against
~415–430 GB/s achievable read + write on this board.

### When it helps / when it does not (reduce case)

**Rule:** split a row-reduction across `k` Tensix cores only when (a) the tile-rows leave most of the grid
idle (`R_t` ≤ ~1/8 of the grid; the baseline's DRAM rate well below the ceiling), (b) each slice keeps
**≥ 4 tiles** (2 tiles is break-even-to-small-win, 1 tile always loses), and (c) **`k` ≤ 8** — so `Wt` ≥ 8 for
`k = 2`, `Wt` ≥ 16 for a real gain, best at `Wt` ≥ 32. The saved per-core work, ~(Wt − Wt/k) × ~230 ns per
tile on BH p150b, must beat the combine (~0.5 µs at `k = 2`, ~1.5 µs at `k = 8`).

| Regime (BH p150b, 130 Tensix cores) | row_split_reduce → width_split_reduce | speedup |
|---|---|--:|
| **Helps:** `R_t = 1, Wt = 32, k = 8` (1 → 8 Tensix cores, 4 tiles per slice) | 8 715 → 3 690 ns | **2.34–2.37×** |
| Helps (sweep): `R_t = 1, Wt = 64, k = 8` (8 tiles per slice) | 16 004 → 4 633 ns | 3.45× |
| Helps, little: `R_t = 16, Wt = 64, k = 2` (16 → 32 Tensix cores) | 17 834 → 13 987 ns | 1.28× |
| **Does not help:** `R_t = 64, Wt = 16, k = 2` (64 → 128 Tensix cores, 8 tiles per slice) | 13 581 → 13 759 ns | **0.98–0.99×** |
| Edge: `R_t = 64, Wt = 8, k = 2` (4 tiles per slice) | 7 157 → 6 913 ns | 1.03–1.04× |
| Does not help: `Wt = 4`, any `R_t`, any `k` | e.g. `R_t = 1, k = 2` 2 083 → 2 299 ns | 0.69–0.95× |
| Does not help: 1-tile slices (`k = Wt`) | e.g. `R_t = 1, Wt = 32, k = 32` 8 567 → 10 091 ns | 0.52–0.97× |

- **Why it helps at `R_t = 1`:** one Tensix core streams the whole row at ~230 ns per tile (reduce + apply +
  DRAM round trip); `k` Tensix cores each stream `Wt/k` tiles and add one ~0.5–1.5 µs combine.
- **Why it does not help once the tile-rows occupy ~half the grid:** at `R_t = 64` the 64 baseline Tensix
  cores already move 290–330 GB/s, ~70–80% of the DRAM read + write ceiling. Doubling the Tensix cores
  halves per-core work but not the bytes, so the time stays (0.98–1.09×) — the combine is not what
  stops it here, the DRAM ceiling is.
- **Why small slices and large `k` lose:** the combine grows with `k` (each member sends and receives `k`
  partials) while the per-core work it saves shrinks to a tile or two.
- **If your combine is heavier** (a root that sums and sends the result back = two rounds, a full write
  barrier per signal, whole-tile partials) the break-even moves to larger slices: measure your per-row
  combine as `T(k, Wt) − T(1, Wt/k)` (same tiles per Tensix core) and compare it with the per-core time saved.
- **On `Layout::ROW_MAJOR` sticks** the split also multiplies the DRAM transfer count by `k`: see
  [the next section](#reduce-case-on-row_major-sticks--splitting-a-stick-multiplies-the-transfers).

*How measured:* `DEVICE KERNEL DURATION [ns]`. Sweep: `test_width_split_reduce_device_perf`, mean of 20 launches
per cell after 5 warmup launches. Regimes: `test_width_split_regimes` (12 trials per variant interleaved, a
trial = mean of 5 launches, medians; ranges over 2–5 runs). `bh-50-special-adamjanovic-for-reservation-143315`,
BH p150b, AICLK max 1350 MHz, 2026-10-29, base commit `8104cced9bc`. Full table in [`report.md`](report.md).

Reproduce:
`scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes`

## Reduce case on ROW_MAJOR sticks — splitting a stick multiplies the transfers

Same op, same two variants, same combine, but the input and output are bf16 `Layout::ROW_MAJOR`,
`TensorMemoryLayout::INTERLEAVED` DRAM: one page = one **stick** of `W` datums = 2·W bytes. Per tile-row a
Tensix core reads its piece of each of the 32 sticks (**32 reads of 2·W/k bytes**, one barrier), tilizes the
`Wt/k`-tile slice, reduces / combines / applies as above, untilizes, and writes 32 pieces back. Kernels:
`kernels/ws_reduce_*.cpp` built with `WS_ROW_MAJOR` (the slice width is a compile-time define for the
tilize / untilize helpers).

What changes against the TILE case — the cost side of the same "does the width split pay" question:
- **A Tensix core's transfer count does not shrink with `k`**: 32 + 32 per tile-row at any `k`; only the
  bytes per transfer do. With a tile page, halving the slice halves the reads.
- **The grid's transfer count grows `k`-fold**, and each small DRAM transfer costs the grid ~1 ns (BH p150b):
  pieces below ~400 B are count-bound, not byte-bound, against the ~415 GB/s DRAM ceiling.

### Measured (BH p150b, 13×10 = 130 Tensix cores, bf16, mean of 20 launches)

Speedup of `width_split_reduce` over `row_split_reduce` (`k = 1`); bytes per DRAM transfer in brackets.

| R_t (tile-rows) | W = 256 (512 B sticks) | W = 512 (1 KB) | W = 1 024 (2 KB) | W = 2 048 (4 KB) |
|--:|---|---|---|---|
| 1 | k=2 1.01× (256 B) · k=4 1.02× | k=2 1.09× (512 B) · k=4 1.15× (256 B) | k=2 1.22× · k=4 1.36× | 12 626 ns → k=2 1.51× · **k=4 1.88×** (1 KB) |
| 4 | 1.01× · 1.02× | 1.08× · 1.15× | 1.21× · 1.37× | 1.38× · 1.70× |
| 16 | 0.90× · 0.71× | 1.05× · 0.90× | 1.16× · 1.22× (512 B) | 1.25× · 1.38× |
| 47 | **0.69×** (94 Tensix cores) · 0.44× | 0.82× · 0.61× | 0.98× · 0.83× | 1.08× (2 KB) · 1.02× |
| 64 | 8 424 → 12 902 ns **0.65×** (64 → 128 Tensix cores) · 0.43× | 0.82× · 0.61× | 0.95× · 0.83× | 1.07× · 1.02× |
| 128 | k=2 0.72× (130 Tensix cores) | 0.86× | 0.94× | 1.08× |

128 and 256 B sticks (W = 64, 128) lose at every `R_t` and `k` (0.38–0.94×). The field shapes `2048 × 256`
(R_t = 64) and `4 × 8 × 47 × 256` (R_t = 47, 512 B sticks): **k = 2 is 1.54× and 1.45× slower**.

**Transfer cost, isolated** (BH p150b): bytes held constant, `R_t = 64, W = 128`, k = 1 / 2 / 4 = 4 096 /
8 192 / 16 384 transfers → 6 936 / 10 929 / 18 484 ns (+0.92–0.97 ns per added transfer); per-core work held
constant, `W = 64, k = 1`, R_t = 64 → 128 → 6 493 → 10 600 ns (+1.0 ns per added 128 B transfer). Per Tensix core,
`R_t = 1, W = 256`: k = 1 / 2 / 4 = 4 976 / 4 910 / 4 891 ns although its tiles drop 8 → 4 → 2.

### When it helps / when it does not (ROW_MAJOR)

**Rule:** split a stick only where the TILE rule already allows a split (tile-rows on ≤ ~1/8 of the grid)
**and** the sticks are long: from **1 KB sticks** (W ≥ 512) when the tile-rows fill ≤ ~4 Tensix cores, from
**2 KB sticks** (W ≥ 1 024) at R_t = 16 of 130, and keep each piece **≥ 512 B** (≥ 256 B only at R_t ≤ 4).
With the tile-rows on ≥ ~1/3 of the grid (R_t ≥ 47 of 130), do not split a stick: pieces ≤ 1 KB give 0.38–1.02×,
2 KB pieces (4 KB sticks, k = 2) 1.07–1.08×.

| Regime (BH p150b, 130 Tensix cores) | row_split_reduce → width_split_reduce | speedup |
|---|---|--:|
| **Helps:** `R_t = 1, W = 2 048, k = 4` (4 KB sticks, 1 KB pieces, 1 → 4 Tensix cores) | 12 803 → 6 872 ns | **1.86–1.88×** |
| **Does not help:** `R_t = 64, W = 256, k = 2` (512 B sticks, 256 B pieces, 64 → 128 Tensix cores) | 8 566 → 13 032 ns | **0.65–0.66×** |
| Edge: `R_t = 64, W = 2 048, k = 2` (2 KB pieces) | 50 471 → 46 953 ns | 1.07× |

- **Why it loses where the TILE path is flat:** with the tile-rows on half the grid the TILE split already
  buys nothing (DRAM near its ceiling); on sticks it also doubles the transfers at ~1 ns each, so
  `R_t = 64, W = 256` goes from 4 096 transfers in 8.4 µs to 8 192 in 12.9 µs.
- **Why it needs longer sticks at small `R_t` than the TILE path needs tiles:** each Tensix core keeps its
  64 transfers per tile-row whatever `k`, so only the tilize / reduce / untilize work (~150 ns per tile) and the
  bytes split; with 512 B sticks there is too little of either to pay for the combine.
- **Check in your own kernel:** count DRAM transfers per launch (sticks × pieces per stick, read + write) and
  multiply by ~1 ns (BH p150b); if that exceeds bytes ÷ ~400 GB/s, the transfers are count-bound and a split
  that adds pieces costs time, not saves it.

*How measured:* `DEVICE KERNEL DURATION [ns]`. Sweep: `test_width_split_reduce_rm_device_perf`, mean of 20 launches
per cell after 5 warmup launches (R_t = 1, 4 from a second run whose R_t = 16, 64 cells matched within 1%).
Regimes: `test_width_split_regimes` (12 trials per variant interleaved, a trial = mean of 5 launches, medians).
`bh-50-special-adamjanovic-for-reservation-143315`, BH p150b, 2026-10-30, base commit `30351ce33f5`. Full
table in [`report.md`](report.md).

Reproduce:
`scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes`
