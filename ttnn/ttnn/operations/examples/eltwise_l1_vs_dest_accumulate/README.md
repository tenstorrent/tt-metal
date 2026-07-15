# eltwise_l1_vs_dest_accumulate — where the running accumulator lives

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** the accumulate mechanism in a reduction loop — L1↔DEST round-trip vs. packer L1-accumulation vs. DEST-resident accumulation
**First profiled on:** `bh-qb-11-special-dnijemcevic-for-reservation-42432` · BH · Blackhole · 2026-07-15

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
  distinct tiles of the same CB — `srcA = tile 2k`, `srcB = tile 2k+1` — via per-operand `TileOffset`
  + `OperandKind::Scalar`), sums them in DEST, and packs with `OutputLifecycle::L1AccumulationCallerManaged`
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
