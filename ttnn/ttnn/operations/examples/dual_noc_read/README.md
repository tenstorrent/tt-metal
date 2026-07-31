# dual_noc_read — two independent DRAM operands, two read engines instead of one

**Difficulty:** ⭐⭐ T2 Intermediate (⭐⭐⭐ for the semaphore form) · **Concept(s):** (1) splitting two
*independent* DRAM input streams across the two data-movement RISC-Vs; (2) reads in flight per
barrier (`block`), the axis that decides how much that split is worth.
**First profiled on:** `bh-qb-13-special-dnijemcevic-for-reservation-52900` · BH · 2026-07-31 · `4a1d6a97ca9`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

This is a **second-tensor sibling of the split-reader idea**: same underlying move (put the other
data-movement RISC-V to work), but applied across *two different tensors* rather than by cutting one
tensor's transactions in half.

## The problem

A fused kernel often needs two operands that have **no dependency on each other**. The shape to look
for is a kernel that computes something like

```
    A op1 B          e.g. two projections of the same activation,
    A op2 C                or two independent inputs to one fused chain
```

`B` and `C` are different tensors, both in DRAM, and **nothing says they must be fetched in order** —
they could be read at the same time. The natural way to write it is one reader that fetches `B`, then
`C`, then hands both to compute. That is one line shorter and it works.

But a Tensix core has five RISC-V processors, two of which move data: **NCRISC** (conventionally the
"reader", NoC 0) and **BRISC** (conventionally the "writer", NoC 1). If the reader fetches both
streams, BRISC does nothing during the whole read phase. In a read-heavy phase — before compute has
produced anything for the writer to drain — that second data-movement processor is free capacity
sitting idle, and everything queues behind one of them.

Taking that capacity is **not free**, and the cost is not obvious: one RISC is one NoC, so handing an
operand to BRISC also moves it onto NoC 1, which is the *worse* route for DRAM reads. Whether the
extra engine outweighs that is a measured question with a sign that changes —
see [*Where this wins and where it does NOT*](#where-this-wins-and-where-it-does-not--core-count--transaction-size).

## What this isolates — and how

- **Concept:** how many data-movement RISC-Vs fetch two independent operand streams (1 vs 2), and how
  many reads each keeps in flight per barrier.
- **Isolation setup** (the *DRAM read efficiency* row of the isolation rule):
  - inputs `A` and `B` are **DRAM interleaved** — the reads under study;
  - the output `C = A*B` is **L1 height-sharded on the same single core** with its CB *aliased to the
    tensor*, so compute packs straight into the output buffer and **no kernel drains it to DRAM**. The
    measured kernel moves DRAM bytes in one direction only, and BRISC is free to be a pure second
    *read* engine instead of splitting its time with writes;
  - compute is **one `mul_tiles` per tile pair, byte-identical in every variant** — dummy math, there
    only to consume the operands;
  - **one core**, so the number is about this core's own data-movement capacity, not cross-core
    contention.
- **Why it's kernel-level:** which RISC-V issues which read, and how many reads it batches per
  barrier, are decisions the kernel author makes. Nothing about the tensors or the op's semantics
  changes.

**Fairness:** the baseline gets the *good* single-RISC implementation — it issues all `2*block` reads
before its single barrier, so it has **more** transactions in flight than either candidate. It is not
handicapped with extra barriers.

## The methods being compared

| Variant | What it does | Why it should differ |
|---|---|---|
| `one_riscv` *(baseline)* | NCRISC reads both operands. BRISC issues nothing. | Everything queues behind one data-movement processor. |
| `two_riscv` | NCRISC reads A; BRISC reads B. **Each RISC owns its own CB** (its own reserve/push), so plain CB semantics order everything — no semaphore. | Two processors issue concurrently. Cheapest correct form. |
| `two_riscv_sem` | Same split, but BRISC fills a slot the **reader** owns, and two local same-core semaphores order them (`go` = slot reserved, `done` = bytes landed). | Same concurrency, plus one handshake per block. Needed when the reader must *do something* to the block after it lands (forward it, multicast it, re-page it) and so cannot give up CB ownership. |

BRISC's kernel is present in **all three** variants (it just returns in the baseline) so kernel count
and launch shape are identical.

A fourth variant, `one_riscv_brisc`, is a **diagnostic** rather than a candidate: BRISC reads *both*
operands, so all reads ride NoC 1 with the same single-RISC issue load as the baseline. Because a
RISC and its port move together, that is the only way to see the port/route contribution on its own.
It is used in the multi-core tables below.

## Measured result (BH P150, single core, bf16, 128 tiles/operand)

Full tables and noise figures: [`report.md`](report.md). Max spread across all cells was **0.5%**.

| block | full op `C=A*B` | payload-ablated (pure read) |
|---|---|---|
| 1 | 1.14× | 1.14× |
| 4 | 1.23× | 1.24× |
| **8** | **1.34×** ← best full-op | 1.37× |
| 16 | 1.27× | 1.49× |
| **32** | 1.14× | **1.59×** ← best read (61.2 → 97.3 GB/s) |

`two_riscv` vs `one_riscv`. The payload ablation removes `mul_tiles` while keeping every CB handshake
and the pack cycle.

**Three things to take away:**

1. **On one core the split wins at every block size measured**, for zero extra L1. (Multi-core is a
   different story — see *Where this wins and where it does NOT* below.)
2. **The read-side win grows monotonically with `block`, but the full-op win peaks at `block=8` and
   then decays.** The ablation says why: the measured FPU cost rises from ~0 at `block≤2` to ~4.1 µs
   at `block=32`. Once the reads get fast enough the FPU becomes the critical path, so a faster read
   has less and less left to uncover. The trick does not stop working at large blocks — the *op* stops
   being read-bound. Always run the ablation before concluding where the win "stops".
3. **Two engines buy ~1.6×, not 2×** at this transaction size. Don't budget 2×.

**On the semaphore form:** `two_riscv_sem` is a **loss** at `block=1` (0.97× — 128 same-core
handshakes), breaks even around `block=2`, and converges to within ~1% of `two_riscv` by `block≥16`.
Prefer the handshake-free form; pay for semaphores only when the reader must keep ownership of the
block, and then use a large block so the cost amortizes.

## What is actually being relieved — issue rate, not saturated links

"Two NoC ports instead of one" is the intuitive story, but it is mostly **wrong**. The bottleneck
being relieved is predominantly **how fast a single RISC-V can issue NoC commands**.

**One RISC is one NoC**, so there is nothing to disentangle by A/B. Each data-movement processor is
bound to a single port (NCRISC → NoC 0, BRISC → NoC 1), and firmware initializes only that port's
per-RISC state (`noc_local_state_init(noc_index)`). "Add a RISC" and "add a port" are the same knob:
there is no 2×2 of RISCs × ports to sweep, so the two contributions cannot be separated by varying
one and holding the other.

The mechanism is therefore established by measurement instead, two independent ways.

**(a) On device: hold bytes fixed, scale the command count.** Table [3] in `report.md` keeps the total
bytes constant and shrinks the transaction size, which issues proportionally more NoC commands for the
same traffic:

| txn bytes | commands | 1-RISC ns | read GB/s | 2-RISC win |
|---|---|---|---|---|
| 2048 | 256 | 13,696 | 38.3 | 1.36× |
| 1024 | 512 | 22,876 | 22.9 | 1.58× |
| 512 | 1024 | 38,661 | 13.6 | 1.75× |
| 256 | 2048 | 66,814 | 7.8 | **1.86×** |

Same bytes, **4.9× more time** for 8× the commands, and achieved bandwidth *collapses* 38.3 → 7.8
GB/s. If bytes on the wire were the constraint, time would be roughly flat. And the two-engine win
**grows toward the ideal 2×** as command count rises — exactly what parallelizing issue work predicts.

**(b) Independently, from a NoC-only model.** Feeding the same transfer set to a NoC performance
estimator (which models links/NIUs and has **no** notion of RISC-V issue cost) gives, for the
one-engine case, a bound of **60.8 GB/s** at 2048 B and **32.1 GB/s** at 256 B — and reports average
link utilisation of only **~11%**. Compare to measured:

| txn bytes | NoC-model bound | measured (1 RISC) | fraction of bound |
|---|---|---|---|
| 2048 | 60.8 GB/s | 38.3 | 63% |
| 1024 | 60.7 | 22.9 | 38% |
| 512 | 42.6 | 13.6 | 32% |
| 256 | 32.1 | 7.8 | **24%** |

The kernel falls further and further below the NoC's own bound as commands multiply. The links are
nowhere near saturated (~11% utilisation); the shortfall is per-command cost upstream of the NoC.

**The honest caveat:** the same model *does* predict a clean ~2.0× from spreading the transfers over
two NoCs even with all issue cost removed, so the port's injection path is a real serializer too.
Since a RISC and its port move together, the two contributions cannot be split apart. The defensible
summary: **issue rate dominates — it is what the measurements track — link bandwidth is not the
limit, and the port's injection path contributes a smaller share.**

Practical consequence: the smaller your transactions, the more this trick is worth. With large
contiguous reads you are closer to the byte side and should expect less.

## Where this wins and where it does NOT — core count × transaction size

The single-core numbers above do **not** generalize. `two_riscv` vs baseline, payload-ablated,
`block=8`, per-core work held constant (32 tiles/operand/core) — full tables in
[`report.md`](report.md):

| cores | grid | 2048 B (bf16 tile) | 1024 B | 512 B | 256 B |
|---|---|---|---|---|---|
| 1 | 1×1 | 1.32× | 1.58× | 1.74× | **1.85×** |
| 8 | 8×1 | **0.68× ← regression** | 0.94× | 1.34× | 1.74× |
| 64 | 8×8 | 1.14× | 1.18× | **1.46×** | **1.66×** |

Two forces compete, and their balance sets the sign.

**Force 1 — issue-rate relief** (what the split buys). Cost per NoC command is roughly fixed, so this
grows as transactions shrink: 1.32× → 1.85× down the single-core row.

**Force 2 — the NoC-1 read-route penalty** (what the split costs). **Reads want NoC 0.** The two NoCs
are dimension-ordered in opposite directions — NoC 0 east→south, NoC 1 north→west — so NoC 0
*disperses* column-localized DRAM read traffic while NoC 1 *concentrates* it onto the DRAM columns.
Writes are the mirror image, which is exactly why the reader-NoC0 / writer-NoC1 default pairing
exists. [`noc_placement`](../noc_placement/README.md) isolates and measures this directly: reads on
NoC 0 / writes on NoC 1 is **2.5–4.8× faster** than the reverse for spread core placements.

**That is what the regression is.** Giving operand B to BRISC necessarily puts it on NoC 1 (one RISC
is one NoC), so the split moves **half the read traffic onto the worse route** — and buying a second
issue engine does not pay for it at a 2 KB tile page.

Measured here in isolation by the `one_riscv_brisc` diagnostic, which keeps the baseline's
single-RISC issue load but puts *all* reads on NoC 1:

| cores | grid | all reads on NoC 1, vs all on NoC 0 |
|---|---|---|
| 1 | 1×1 | 0.98× — routing is irrelevant on one core |
| 8 | 8×1 | **0.34×** |
| 16 | 8×2 | 0.40× |
| 32 | 8×4 | 0.54× |
| 64 | 8×8 | 0.62× |

Consistent with `noc_placement`, and it explains the shape of the first table: the penalty is absent
at one core, worst on a single row of cores, and eases as the grid fills out.

**Why 64 cores recovers to ~1.1×:** DRAM saturates. Aggregate read bandwidth plateaus at **~340–410
GB/s from 16 cores up** while per-core bandwidth collapses 37 → 5.6 GB/s. Once the read is DRAM-bound,
neither force has much left to express, so both the win and the penalty flatten out.

**Why small transactions flip the sign:** they push both forces the same way — issue cost rises *and*
the NoC-1 penalty fades (0.34× → 1.05× at 8 cores, because per-command cost starts to dominate
routing). By ~512 B the split is a win at every core count tested.

**The rule: this is a small-transaction optimization, not a "more cores" one.**

- **Use it** when reads are command-limited — many small transactions. Block-float weight pages sit
  exactly there (a bfp4 tile page is 576 B, bfp8 1088 B), and at ~512 B the split wins at **every**
  core count tested: 1.34× at 8 cores, 1.46× at 64, 1.74× on one core.
- **Don't** use it for large, tile-sized reads spread over a moderate core count — at bf16 2048 B on
  8 cores it is a **0.68× regression**, because you pay the NoC-1 read-route penalty for nothing.
- On a **single core** it always wins, and routing does not matter.
- If you are already DRAM-bandwidth-bound, expect little either way.
- The shape above is not specific to this grid width; it reproduced on both 8-wide and wider
  rectangles, with the regression always landing on the single-row placement.

## Gist — how to apply it

1. Move the second operand's `noc_async_read` into the **writer** kernel (`WriterConfigDescriptor`
   lands it on BRISC).
2. Let that kernel **own** that operand's CB — its own `cb_reserve_back` / `cb_push_back`. Compute
   already waits on both CBs independently, so no semaphore is needed and compute is unchanged.
3. Keep `block ≥ 8` so each engine has enough reads in flight to be worth splitting.
4. Only if the reader must post-process the block: keep CB ownership in the reader and add the two
   semaphores (`two_riscv_sem`) — and note the writer must then **reconstruct the live CB slot
   itself**, because a CB write pointer is per-RISC and the writer's never advances. That is why this
   example requires `total_tiles % block == 0`: a short tail block would advance the reader's pointer
   by less than a full slot and desynchronize the arithmetic.

## CLI — measure your own shapes/params

```bash
python -m ttnn.operations.examples.dual_noc_read [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--shape` | `H,W` | `1024,128` | shape of each bf16 tiled operand (128 tiles). Total tiles must divide every `--blocks` value. |
| `--blocks` | comma list | `1,2,4,8,16,32` | reads-per-barrier, per RISC, to sweep |
| `--variant` | `all` \| comma list | `all` | which candidates to run; the baseline is always measured so speedups have a reference |
| `--trials` | int | `5` | independent profiler windows; report shows median ± spread |
| `--iters` | int | `10` | launches averaged inside each window |

Multi-core sweeps run through the test directly (the grids and per-core tile count are env knobs):
`DNR_GRIDS=1x1,8x1,8x8`, `DNR_TPC=32` (tiles per operand per core), `DNR_TXN_GRIDS`, `DNR_TXNS`.

There is deliberately **no `--kernel-iters`**: the output is L1-resident and nothing drains it, so a
launch performs exactly one pass over the tiles. Amortize launch overhead with `--shape` instead.
The transaction-size sweep behind table [3] is set with `DNR_TXNS` (default `2048,1024,512,256`).

```bash
python -m ttnn.operations.examples.dual_noc_read
python -m ttnn.operations.examples.dual_noc_read --shape 2048,128 --blocks 8,16,32 --variant two_riscv
```

## Tests

```bash
# correctness — the only pass/fail: every variant x block produces A*B
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_correctness

# the single-core sweep (all three tables)
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_device_perf

# multi-core scaling, and core count x transaction size
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_multicore_perf \
    tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_multicore_txn_probe
```

## Caveats

- **Perf is measured, never asserted.** Correctness (`C == A*B`) is the only pass/fail.
- **The win is bounded by how read-bound the op is** — see takeaway 2; run the ablation before
  drawing conclusions about where it "stops".
- The output being L1-resident is what makes this a clean read-only measurement. A real op writes its
  result, and if the writer must do that *concurrently* with reading the second operand, this is no
  longer free. The trick is free precisely when the writer would otherwise be idle.
- **Grids here are rectangular and anchored at (0,0)**, with work split row-major. Different
  placements route differently (that is force 2 above), so re-measure rather than extrapolating to a
  different core layout.
- Only `bfloat16` operands are wired up; `txn_bytes` is what stands in for smaller pages. A real
  block-float tensor also changes the page size seen by the accessor, so treat the 512 B column as
  indicative of a bfp4-sized read, not identical to one.
