# eltwise_l1_vs_dest_accumulate — where the running accumulator lives

**Problem type:** compute-bound: L1 round-trips  ·  **Concept(s):** the accumulate mechanism in a reduction loop — L1↔DEST round-trip vs. packer L1-accumulation vs. DEST-resident accumulation
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH · Blackhole · 2026-07-15
**Measured on:** WH B0 n150 (regimes + step sweep, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
Many kernels build a **running accumulator** by summing a stream of tiles into it. Addition can
happen at **three distinct points** in the compute pipeline, and a running sum can be built at any:
- the **FPU adder** — `add_tiles(A, B)` computes `A + B` into DEST (in fact any FPU binary op);
- the **DEST accumulator** — `acc_to_dest` folds each FPU result into a held DEST tile
  (`DEST += FPU_op(in1, in2)`), with no repack between adds;
- the **L1 accumulator** — `pack_reconfig_l1_acc` folds the packed DEST onto the resident L1 tile
  (`L1 += <whatever DEST holds>`) at pack time.

What you accumulate is incidental to the mechanism. The dominant cost is **how much L1 traffic the
accumulator itself pays**: every time you re-read `acc` into the FPU and write it back is a round-trip
through the packer/unpacker. This example uses a plain per-tile add to isolate that, and measures
three ways to combine those adders, each removing more accumulator traffic, on the identical fp32 result.

## The three methods
Each combines the three adders differently (`acc` is the running accumulator):

| Variant | Adders used | What it does | L1 traffic on `acc` |
|---|---|---|---|
| `rmw` *(baseline)* | FPU only | one streaming `ckl::add<cb_out, cb_in, cb_out>(tiles(B-1))` fold — each iter unpacks `acc` **and** the next block into DEST, adds on the FPU, packs the result back to `acc` | `acc` **unpacked and packed every tile** (B steps) |
| `pack_l1_acc` | FPU + **L1 accumulator** | read **two** tiles per step: one `BinaryFpu` puts `X[2k]+X[2k+1]` in DEST, then the packer folds that onto `acc` in place (`pack_reconfig_l1_acc`) | `acc` **only packed, never unpacked**, once per pair (B/2 steps) |
| `dest_acc` | FPU + **DEST accumulator** | the running sum stays in a **sticky DEST** tile the whole reduction (`acc_to_dest`); pack to L1 **once at the end** | `acc` **never touches L1** until the final pack |

`rmw` uses only the FPU adder, so the accumulator must round-trip L1 every step; `pack_l1_acc` adds
the L1 accumulator (never unpacks `acc`); `dest_acc` adds the DEST accumulator (keeps `acc` off L1
entirely). All three keep `acc` in fp32, produce the identical sum, and go through the `eltwise_chain`
helper. The progression is monotonic — each step strips more accumulator L1 traffic:

- **`rmw`** is the naive read-modify-write. A single streaming `ckl::add` with `cb_out` as both an
  operand and the output does the honest fold: the per-iteration `cb_out` push→wait both *drives* the
  L1 round-trip and *provides the PACK→UNPACK sync* so the next step reads the freshly written `acc`.
  Nothing is retained in DEST between steps.
- **`pack_l1_acc`** reads two tiles at a time from a **caller-managed** input (so one add can read two
  distinct tiles of the same CB — `srcA = tile 2k`, `srcB = tile 2k+1` — via per-operand `TileAddressing`
  + `InputTileMapping::Scalar`), sums them in DEST, and packs with `OutputLifecycle::L1AccumulationCallerManaged`
  so the **packer** adds DEST onto the resident `acc` tile. `acc` is never unpacked; the binary init is
  hoisted once (`SetupOwner::Caller`).
- **`dest_acc`** keeps the running sum in one **sticky DEST** tile (`BinaryFpu<…, DestAccumulation::Enabled>`
  → `OutputLifecycle::DestAccumulation`) and packs the final `acc` exactly once. Because the sticky
  DEST must live across all pairs (one DEST acquire), the two operands are the two **halves** of the
  stream (`srcA = tile i`, `srcB = tile i+N`) rather than adjacent pairs.

## What this isolates — and how
- **Isolation setup:** *pure compute* row of the rule. Input and accumulator are all sharded in L1 on
  a **single Tensix core** (no DRAM in the fast path); the resident input is re-exposed block-by-block,
  so the measured delta is purely the per-step accumulator handling. fp32 both ways → identical, correct sum.
- **Why it's kernel-level:** where the accumulator lives (L1 vs DEST) and how it's updated is entirely
  the compute-kernel author's choice.
- **Caveat on `dest_acc`:** it *camps* the accumulator in DEST for the whole reduction, which is only
  possible when DEST is otherwise free. A real accumulation loop usually needs DEST for per-step work
  (a matmul that produces the block, an exp, a reduce), which is exactly why `acc` lives in L1 in the
  first place. So read `dest_acc` as the **upper bound** ("if you *could* keep it in DEST"), and
  `pack_l1_acc` as the realistic win when the accumulator must stay resident in L1.

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.eltwise_l1_vs_dest_accumulate [--blocks 64] [--iters 100] [--trials 10]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--blocks` | int | 64 | accumulation steps (single-tile blocks summed; must be even) |
| `--iters` | int | 100 | in-kernel repeat (steady-state) |
| `--trials` | int | 10 | profiled rounds; report shows median |

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box.*

```
eltwise_l1_vs_dest_accumulate  box=bh-qb-11…  arch=Blackhole  cores=1  single-core sharded-L1  blocks=64  tiles/block=1  iters=100
  method          ns/op        vs rmw
  rmw           975665.2        (base)
  pack_l1_acc   191808.8         5.09x
  dest_acc       92149.7        10.59x
```

**Reading of the result:** the win tracks exactly how much L1 traffic each method spends on the
accumulator. `rmw` round-trips `acc` through unpack + math-add + pack **every tile** — the most
traffic, the baseline. `pack_l1_acc` only ever **packs** `acc` (the packer folds DEST onto it in
place, so `acc` is never unpacked) and does it once per **pair** → **5.09×**. `dest_acc` keeps the
running sum in DEST and touches L1 exactly once, at the end → **10.59×** — the upper bound. With one
tile per step and no independent math to hide the accumulator handling, the differences are at their
clearest; they shrink when heavy per-step math overlaps the unpack/add.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_device_perf
```
Correctness (all three variants compute the identical fp32 sum):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_correctness
```

## When it helps / when it does not
**Rule:** folding the accumulator with the packer (`pack_l1_acc`) instead of an L1 round-trip (`rmw`) pays at every
step count tried (≥ 2 single-tile steps) on both boards; keeping the sum in DEST (`dest_acc`) over `pack_l1_acc`
pays from 4 steps and grows with the step count B — to 1.30× at 64 steps on WH B0 n150 but 2.01× on BH p150b.

| Regime (same definition on every board; 1 Tensix core, sharded L1, fp32, 1 tile per step) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** accumulator round-trips L1 every step — `rmw → pack_l1_acc`, B = 64 steps | 23 784 → 5 769 ns **4.12×** | 9 772 → 1 857 ns **5.26×** |
| **Helps:** many steps to keep in DEST — `pack_l1_acc → dest_acc`, B = 64 steps | 5 768 → 4 447 ns **1.30×** (step sweep) | 1 857 → 922 ns **2.01×** |
| **Does not help:** only one pair to fold — `pack_l1_acc → dest_acc`, B = 2 steps | 367 → 370 ns **0.99×** | 155 → 155 ns **1.00×** |
| Step sweep `pack_l1_acc → dest_acc`, B = 2 / 4 / 8 / 16 / 64 | 0.99× / 1.09× / 1.17× / 1.25× / 1.30× | 1.00× / 1.15× / 1.40× / 1.65× / 2.01× |
| Step sweep `rmw → pack_l1_acc`, B = 2 / 4 / 8 / 16 / 64 | 2.03× / 2.75× / 3.36× / 3.72× / 4.10× | 2.25× / 3.15× / 3.89× / 4.58× / 5.27× |
| **Boundary** (`dest_acc` over `pack_l1_acc`) | B = 4 steps (1.09×); ≥ 1.15× from B = 8 | B = 4 steps (1.15×) |

**Why the boards differ:** the marginal cost per extra step (ns at B = 64 minus B = 2, over 62 steps) is ~87 ns for
`pack_l1_acc` and ~66 ns for `dest_acc` on WH B0 n150, but ~27 ns and ~12 ns on BH p150b. Both paths got faster than
the 1.35× AICLK ratio (1350 vs 1000 MHz) alone explains, but the DEST-resident add (unpack + FPU add into DEST) got
much faster than the packer's L1 accumulate, so on BH p150b every step left in `pack_l1_acc` costs ~2.2× a
`dest_acc` step instead of ~1.3×. (Per-step costs derived from the step sweeps; the per-unit split is inferred, not
profiled per zone.)

- **Helps** because `rmw` unpacks and packs `acc` every tile (B steps) while `pack_l1_acc` only packs it, once per pair (B/2 steps).
- **Does not help (for `dest_acc` over `pack_l1_acc`)** at B=2: both are one add plus one pack. Their gap opens only
  as B grows — slowly on WH B0 n150 (B=4: 1.09x, B=8: 1.17x, B=16: 1.25x, B=64: 1.30x), fast on BH p150b
  (B=4: 1.15x, B=8: 1.40x, B=16: 1.65x, B=64: 2.01x). On BH p150b, prefer `dest_acc` for any run of 4+ steps.
  Note `pack_l1_acc` over `rmw` was NOT flat at any step count tried on either board (WH B0 n150 B=2: 2.03x … 64: 4.10x;
  BH p150b B=2: 2.25x … 64: 5.27x), so there is no small-B regime where `pack_l1_acc` stops paying for this single-tile fold.

*How measured:* WH B0 n150 rows — `test_eltwise_l1_vs_dest_accumulate_regimes` (1 Tensix core, sharded L1, fp32,
one tile per step, kernel-iters 100, DEVICE KERNEL DURATION per accumulation pass, 11 interleaved trials, median;
spread 23653..23791 / 5735..5808 and 366..368 / 365..373 ns), 2026-10-12; step sweeps from
`test_eltwise_l1_vs_dest_accumulate_device_perf` on the same board. BH p150b rows — the same regimes test plus
step sweeps on that board, `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19 (details in `report.md`). The
illustrative bh-qb-11 table above (5.09× / 10.59× at 64 steps) is an older Blackhole run, kept.
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).

## Syntax — summing one stream with `DestAccumulation` (moved here from `master.md`)

`DestAccumulation` lives on `BinaryFpu`, which takes TWO CB
inputs, so accumulating ONE stream looks impossible and the tempting fix is to pair every tile against a
zero tile. Don't: that zero tile has to be filled, and the fill is never free. Instead point BOTH operands
at the SAME CB and offset the second by half the run — the operands are the two **HALVES** of the stream
(`x[i] + x[i + N/2]`), not adjacent pairs — so N tiles fold in N/2 FPU steps with no identity operand:
```cpp
constexpr uint32_t N = B / 2;                       // B tiles to sum; the SECOND operand starts at N
ckl::eltwise_chain<ckl::SetupOwner::Caller>(
    ckl::EltwiseShape::tiles(N),
    ckl::BinaryFpu<cb_in, cb_in, Add, None, ..., D0,
                   InputTileMapping::Block, InputTileMapping::Block,
                   ckl::TileAddressing::Direct, ckl::TileAddressing::Offset,
                   ckl::DestAccumulation::Enabled>{0, N},        // <- {A base, B base} IS the trick
    ckl::PackTile<cb_out, ckl::OutputLifecycle::DestAccumulation, ..., D0>{});
```
`tiles(...)` is one contiguous shape, so the accumulation scope is `WholeShape`/`Enabled` (`PerRow` is
rejected there); a 2D walk uses `grid(H, W)` with `TileAddressing::Strided` + a `StridedTileRange{base,
row_stride}` per operand. **Odd N does not tile into halves.** For the specific case of summing tiles that
are ALREADY reduced (per-core partials from an earlier `REDUCE_ROW`), `reduce_helpers_compute.hpp` does it
for you — `ReduceAlgorithm::AccumulateViaAdd` + `ReduceWithinTile::Skip`.
