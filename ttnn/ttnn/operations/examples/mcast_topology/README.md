# mcast_topology — 1-D multicasts vs. redundant per-core DRAM reads on a 2-D work split

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** Tensix↔Tensix multicast topology on a 2-D core grid
**First profiled on:** `bh-49-special-mstaletovic-for-reservation-60064` · BH · 2026-08-13 · `45770842250`

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
| `mcast_1d_pair` | An operand is broadcast along the axis it does **not** vary with. `A` is invariant along a grid ROW → column 0 of each row reads it once and multicasts across the row (`Mcast1D(PerRow)`). `B` is invariant down a grid COLUMN → row 0 of each column reads it once and multicasts down (`Mcast1D(PerColumn)`). | Each slice crosses DRAM **once per line** instead of once per core: `8×` fewer DRAM reads on an 8×8 grid. The copies travel core-to-core instead. |

### The topology inverts, and that is the counter-intuitive part

A **2-D** (block-sharded) work split needs **1-D** multicasts — one source per line, each feeding
only its own row or column. A **1-D** work split (cut `M` only, every core needing the whole of `B`)
is the one that needs a **2-D** multicast — a single injector feeding a whole rectangle.

"More sharded" therefore does *not* mean "bigger broadcast". It means each operand travels a
shorter, narrower path, and the grid dimension an operand does not travel along is exactly the one
carrying the other operand. Two independent `Mcast1D` families ride the same grid on disjoint
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

*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

```
mcast_topology  box=bh-49-...  arch=BLACKHOLE  grid=11x10 (110 cores)  M=8t N=32t K=4t
                delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores=64/110 (58%)  per-core DRAM reads               8512 ns ±0.3%  ✓
  mcast_1d_pair  split=8x8  cores=64/110 (58%)  2x Mcast1D (PerRow + PerColumn)   4450 ns ±1.1%  ✓  → 1.91×
```

**Reading of the result:** delivering the same operands to the same 64 cores is **1.91× faster**
when each slice is read once per line and broadcast. The DRAM tile-read count drops `1280 → 160`
(**8×**) — 64 cores × (4 A-tiles + 16 B-tiles) versus 8 row-senders × 4 + 8 column-senders × 16.

The device-time win (1.91×) is much smaller than the read-count reduction (8×), and that gap is the
honest part of the result: each line's sender reads its slice **serially** before broadcasting, and
the bytes still have to cross the NoC. Expect the win to grow with the line length (a longer row or
column shares one read further) and to shrink toward nothing as the slices get small enough that the
per-chunk semaphore handshake dominates the transfer. At a 1×1 grid there is no line to share along
and both variants are the same program.
