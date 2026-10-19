# mcast_topology — 1-D multicasts vs. redundant per-core DRAM reads on a 2-D work split

**Problem type:** redundant reads  ·  **Concept(s):** Tensix↔Tensix multicast topology on a 2-D core grid
**First profiled on:** `bh-49-special-mstaletovic-for-reservation-60064` · BH P150b · 2026-08-13 · `45770842250`
**Measured on:** WH B0 n150 (regimes + sweep, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem

You have split a tiled `C[M,N] = A[M,K] @ B[K,N]` two ways at once — grid **rows** carry `M`, grid
**columns** carry `N` — because neither axis alone is long enough to fill the grid. Core `(x=c, y=r)`
now owns output block `[M_r, N_c]` and needs two operand slices: `A[M_r, :]` and `B[:, N_c]`.

The obvious implementation has every core fetch its own two slices from DRAM. That is redundant *by
construction*, and the 2-D split is what makes it so: all `Gc` cores in a grid row want the **same**
`A[M_r, :]`, and all `Gr` cores in a grid column want the **same** `B[:, N_c]`. On an 8×8 grid DRAM
serves each A slice 8 times and each B slice 8 times, for one slice's worth of unique bytes.

This is the moment where "a 2-D split costs `P ×` the operand traffic" gets written down as a reason
to reject the 2-D split — and it is only true if you keep every core reading for itself.

## What this isolates — and how

- **Concept:** the on-chip broadcast topology that a 2-D work split requires — and the DRAM
  redundancy it removes.
- **Isolation setup:** *Tensix↔Tensix NoC* — **there is no compute kernel at all.** The cores fetch
  the operand slices and stop; the matmul is never performed, so nothing about the Matrix Unit,
  DEST budget, subblock shape or math fidelity can leak into the number. The **work split is
  identical in both variants** (same `Gr × Gc` rectangle, same cores, same slices, same bytes
  resident per core), so the measured delta is purely *how the bytes arrive*.
- **Why it's kernel-level:** which core sources a shared operand, and whether the copies travel over
  the NoC or are re-fetched from DRAM, is entirely the kernel author's decision. Nothing about the
  tensors, dtypes or shapes changes between variants.

Correctness without computing anything: each core writes three **probe tiles** straight out of the
operand CBs it filled (`A[m0,0]`, `B[0,n0]`, `B[Kt-1, n0+Nloc-1]`), and the test asserts them
**tile-exactly** against the slices that core was supposed to receive. Delivery is a routing
question, so a probe either is the right tile or it is not — no PCC. Inputs carry a distinct constant
per tile, so a misdelivered block is unmistakable rather than plausibly close. Three tiles per core
is negligible next to the transferred operands, so the kernel stays delivery-bound.

## The methods being compared

| Variant | What it does | Why it should differ |
|---|---|---|
| `per_core_dram` *(baseline)* | Every core reads its own `A[M_r, :]` and `B[:, N_c]` from DRAM. No semaphores, no cross-core traffic. | — |
| `mcast_1d_pair` | An operand is broadcast along the axis it does **not** vary with. `A` is invariant along a grid ROW → column 0 of each row reads it once and multicasts across the row (an `Mcast` with row-major groups). `B` is invariant down a grid COLUMN → row 0 of each column reads it once and multicasts down (an `Mcast` with column-major groups). | Each slice crosses DRAM **once per line** instead of once per core: `8×` fewer DRAM reads on an 8×8 grid. The copies travel core-to-core instead. |

### The topology inverts, and that is the counter-intuitive part

A **2-D** (block-sharded) work split needs **1-D** multicasts — one source per line, each feeding
only its own row or column. A **1-D** work split (cut `M` only, every core needing the whole of `B`)
is the one that needs a **2-D** multicast — a single injector feeding a whole rectangle.

"More sharded" therefore does *not* mean "bigger broadcast". It means each operand travels a
shorter, narrower path, and the grid dimension an operand does not travel along is exactly the one
carrying the other operand. Two independent `Mcast` channels ride the same grid on disjoint
semaphore ids, and each core is a sender on one, both, or neither.

## CLI — measure your own shapes/params

```bash
python -m ttnn.operations.examples.mcast_topology [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,per_core_dram,mcast_1d_pair}` | `all` | which method(s) to run and compare |
| `--trials` | int | `5` | measured rounds (median of 5-launch windows) |
| `--mt` | int | `8` | output tile-rows (`M` in tiles) |
| `--nt` | int | `32` | output tile-cols (`N` in tiles) |
| `--kt` | int | `4` | contraction tiles (`K` in tiles) |

The grid rectangle is derived from the device: `Gr` = largest divisor of `Mt` that fits the grid
height, `Gc` = largest divisor of `Nt` that fits the width. There is no `--kernel-iters`: one launch
delivers the operands once, which is the quantity of interest.

```bash
# A/B both delivery methods on your shape
python -m ttnn.operations.examples.mcast_topology --mt 8 --nt 32 --kt 4

# one variant only
python -m ttnn.operations.examples.mcast_topology --variant mcast_1d_pair
```

## Measured result

*Illustrative — re-run the CLI for your box. The same 11×10 worker grid does not imply the same
DRAM configuration: P150b and P100a boards can expose different enabled GDDR counts.*

### P150b reference

The original run did not record busy AICLK, enabled GDDR count, or firmware.

```
mcast_topology  box=bh-49-...  arch=BLACKHOLE  grid=11x10 (110 cores)  M=8t N=32t K=4t
                delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores=64/110 (58%)  per-core DRAM reads               8512 ns ±0.3%  ✓
  mcast_1d_pair  split=8x8  cores=64/110 (58%)  2x Mcast (RowMajor + ColumnMajor) 4450 ns ±1.1%  ✓  → 1.91×
```

### P100a, seven enabled GDDR banks

`bh-43-special-sjovic-for-reservation-97381` · 1350 MHz busy AICLK · firmware 19.12.0 · 2026-09-23

```
mcast_topology  box=bh-43-...  arch=BLACKHOLE  grid=11x10 (110 cores)  M=8t N=32t K=4t
                delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores=64/110 (58%)  per-core DRAM reads              10958 ns ±0.7%  ✓
  mcast_1d_pair  split=8x8  cores=64/110 (58%)  2x Mcast (RowMajor + ColumnMajor) 4683 ns ±0.4%  ✓  → 2.34×
```

**Reading of the results:** delivering the same operands to the same 64 cores is **1.91× faster on
P150b and 2.34× faster on this seven-bank P100a** when each slice is read once per line and
broadcast. The DRAM tile-read count is unchanged across boards and drops `1280 → 160` (**8×**) — 64
cores × (4 A-tiles + 16 B-tiles) versus 8 row-senders × 4 + 8 column-senders × 16. The P100a's
redundant-DRAM baseline is 28.7% slower than the P150b result while its multicast path is only 5.2%
slower, consistent with the baseline putting much more pressure on this seven-bank DRAM system.

The device-time win is much smaller than the read-count reduction (8×), and that gap is the
honest part of the result: each line's sender reads its slice **serially** before broadcasting, and
the bytes still have to cross the NoC. Expect the win to grow with the line length (a longer row or
column shares one read further) and to shrink toward nothing as the slices get small enough that the
per-chunk semaphore handshake dominates the transfer. At a 1×1 grid there is no line to share along
and both variants are the same program.

## When it helps / when it does not
**Rule:** `mcast_1d_pair` pays when **both** grid dimensions are long (lines of ≥ ~5–8 Tensix cores, so each
slice is shared ≥ 5-fold in both directions) and grows with K-tiles per slice; lines of ≤ 4 Tensix cores are
flat on WH B0 n150 and **lose on BH p150b**. Boundary between the 4×4 split (16 Tensix cores) and the 8×8
split (64 Tensix cores) on WH B0; between 4×4 and 5×12 (60 Tensix cores) on BH p150b.

`Gr × Gc` split = largest divisors of `Mt` / `Nt` that fit the grid (`layout()`), so the same shape can give a
different split on each board; the split and shape are in every cell where they differ.

| Regime (split · `Mt × Nt × Kt` tiles) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores | BH P150b (bh-49) · 11×10 = 110 Tensix cores | BH P100a · 11×10 = 110 Tensix cores, 7 DRAM banks |
|---|---|---|---|---|
| **Helps:** 8×8 = 64 Tensix cores · `8 × 32 × 4` | 16 204 (16 077–16 504) → 8 677 (8 602–8 774) ns **1.87×** | 8 698 (8 653–8 821) → 4 327 (4 268–4 374) ns **2.01×** | 8 512 → 4 450 ns **1.91×** | 10 958 → 4 683 ns **2.34×** |
| **Helps:** 8×8, more K · `8 × 32 × 16` | 56 780 → 21 236 ns **2.67×** | **3.27×** | not measured | not measured |
| **Helps:** 8×8, more K · `8 × 32 × 64` | 261 544 → 75 342 ns **3.47×** | **3.90×** | not measured | not measured |
| **Helps:** 10×12 = 120 Tensix cores · `10 × 48 × 4` | not reachable (grid is 8×8) | 14 171 → 5 849 ns **2.42×** | not measured | not measured |
| **Helps:** 10×12, more K · `10 × 48 × 16` | not reachable | **4.18×** | not measured | not measured |
| **Helps:** 5×12 = 60 Tensix cores · `5 × 24 × 4` | not reachable (12 columns) | **1.44×** | not measured | not measured |
| **Does not help:** 4×4 = 16 Tensix cores, lines of 4 · `4 × 4 × 4` | 4 186 (4 105–4 303) → 4 056 (3 963–4 129) ns **1.03×** (flat) | 1 791 (1 765–1 818) → 2 252 (2 241–2 283) ns **0.80× — loses on BH** | not measured | not measured |
| **Does not help:** 2×2 = 4 Tensix cores · `2 × 2 × 4` | 2 288 (2 164–2 653) → 3 120 (3 088–3 174) ns **0.73×** | 1 261 (1 248–1 293) → 1 933 (1 909–1 953) ns **0.65×** | not measured | not measured |
| **Does not help:** single row 1×8 (only A shared) | `1 × 8 × 4` (1 output tile-col per Tensix core): **0.84×** | `1 × 32 × 4` (4 output tile-cols per Tensix core): **0.75×** | not measured | not measured |
| **Does not help:** 2×8 = 16 Tensix cores | `2 × 8 × 4`: **0.86×** | `2 × 32 × 4`: **0.91×** | not measured | not measured |
| **Boundary** | between 4×4 (1.03×) and 8×8 (1.87×); 4×4 needs Kt = 32 to win (1.23×) | between 4×4 (0.80×) and 5×12 (1.44×) | not measured | not measured |

Cells are `per_core_dram → mcast_1d_pair` DEVICE KERNEL DURATION (parentheses: min–max over the 10 windows); speedup-only cells come from a sweep whose
ns were not kept. More WH B0 n150 points (8×1, 8×2, 16×64, 56×56, …) are in [`report.md`](report.md).

**Why the boards differ:**
- Small grids lose on BH p150b because the baseline shrinks more than the handshake: at 4×4 `per_core_dram`
  is 2.3× faster on BH p150b than on WH B0 n150 (1 791 vs 4 186 ns: faster DRAM, 64-byte NoC flit, ~1.0 vs
  ~1.8 µs launch floor) while `mcast_1d_pair` is only 1.8× faster (2 252 vs 4 056 ns), so the two per-line
  semaphore handshakes plus the sender's serial read no longer break even.
- Large grids gain more on BH p150b: the 12×10 grid allows lines of 10 and 12 Tensix cores (2.42× at Kt = 4,
  4.18× at Kt = 16), and at the same 8×8 split the speedup is higher at every Kt (Kt = 4: 2.01× vs 1.87×;
  Kt = 16: 3.27× vs 2.67×; Kt = 64: 3.90× vs 3.47×).

- **Helps** when lines are long and each Tensix core receives more than a handful of tiles: DRAM reads drop
  line-length-fold, and `per_core_dram` with 64+ readers is DRAM-bound; the gain grows with Kt and per-core
  tile count.
- **Does not help** when the grid is small (≤ 16 Tensix cores, lines of ≤ 4: only ≤ 4× fewer DRAM reads) or
  one grid dimension is 1–2 (1×8, 2×8: the `B` slice travels down a column of 1–2 Tensix cores, so it is
  shared at most 2-fold); the semaphore handshakes plus the sender's serial read then cost more than they
  save. On WH B0 n150 4×4 is break-even and 2×2 is 1.4× slower; short Kt on a small grid also loses
  there (4×4, Kt = 1: 0.92×) and only 4×4 Kt = 32 wins (1.23×). On BH p150b 4×4 already loses (0.80×).

*How measured:*
- WH B0 n150: `test_mcast_topology_regimes` (8×8 helps, 4×4 / 2×2 does-not-help) plus a sweep with the same
  code; bf16; median of 10 interleaved windows × 5 launches; 2026-10-12.
- BH p150b: the same regimes test (8×8 helps, 4×4 / 2×2 does-not-help), the 10×12 point from a since-dropped BH-only
  regime, plus a probe with the same measurement code for the other rows; `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19.
- BH P150b (bh-49) / P100a: `test_mcast_topology_device_perf`, boxes and dates under *Measured result* above.

Reproduce (same regimes on any board; asserts only what held on every measured board, `measure` regimes only log):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_mcast_topology.py::test_mcast_topology_regimes
```
