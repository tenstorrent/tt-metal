# trid_double_issue — keeping DRAM reads in flight ACROSS the barrier (transaction ids)

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** barrier **granularity** on a DRAM read stream — a drain-everything `noc_async_read_barrier()` vs. a per-transaction-id wait · how that trades off against block size
**First profiled on:** `bh-50-special-dstoiljkovic-for-reservation-88042` · Blackhole · 2026-09-10 · `dfc0dae18e4`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You already know not to read one page at a time: you issue a **block** of async reads and take
**one** barrier for the whole block. That is the right shape, and it is what almost every reader
looks like. But read what `noc_async_read_barrier()` actually promises — it waits for **every
outstanding read on this NoC**. The instant it returns, *nothing is in flight*. The next block is
not issued until after that wait, so the NoC goes completely idle once per block and you pay a
full DRAM round trip with an empty pipe underneath it.

Batching amortized that round trip over `block` reads. It never removed it. Reads and waits still
strictly alternate, and the bigger you make `block` the more L1 you burn to hide a cost you could
simply stop paying.

## What this isolates — and how
- **Concept:** *which barrier retires a block* — a drain-everything wait
  (`noc_async_read_barrier()`) versus a wait scoped to one transaction id
  (`noc_async_read_barrier_with_trid()`), which leaves the next blocks' reads on the wire while
  you wait for this one.
- **Isolation setup:** DRAM-read efficiency, so compute is held at nothing. The op is an identity
  copy with **no compute kernel at all**: reader (NCRISC, NoC0) fills a circular buffer, writer
  (BRISC, NoC1) drains it straight back to DRAM. The tensor is interleaved in DRAM.
- **What is held constant:** the writer is **byte-identical** in every cell, `cb_blocks` (CB
  depth) is **fixed across the whole table**, and the page order, core count and dtype never
  change. So neither the drain side nor L1 depth can explain the delta.
- **The issue call is the same in both variants.** Both call plain `noc_async_read`. A transaction
  id lives in the read command buffer's `NOC_PACKET_TAG` register; setting the trid writes that
  register and nothing else, while an ordinary `noc_async_read` writes the address, length and
  control registers and never touches the tag. So a tag set once rides along on every subsequent
  read. **Only the barrier differs** — which is also why this works on ordinary **interleaved**
  DRAM, where consecutive pages live in different banks.
- **The tag/wait pair comes from the shared kernel helper library**
  (`dataflow_kernel_lib::set_read_trid` / `::async_read_barrier_with_trid`), not raw NoC calls.
  Same two primitives on this RISC-V's default NoC, plus the watcher's transaction-id sanitizer
  under `--dev`.
- **Why it's kernel-level:** barrier placement and transaction-id bookkeeping are lines of kernel
  code the author writes; nothing about the tensor, the dtype or the work split changes.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `full_barrier`, `ahead=1` *(naive baseline)* | issue `block` reads → `noc_async_read_barrier()` → push | the barrier drains **every** outstanding read, so the next block starts from zero in flight — one exposed DRAM round trip per block |
| `full_barrier`, `ahead=N` *(strong baseline)* | issue `N` blocks → **one** barrier → push them individually | spends the spare CB to keep `N × block` reads in flight. **This is the strongest a non-trid reader can be**, and it is the baseline the headline is quoted against |
| `trid_double_issue`, depth `T` | tag block *k* with id *k mod T*; once `T` blocks are outstanding, retire the **oldest** with a per-id barrier | reads stay on the wire *across* the wait, and the writer still gets one block at a time |

**Why the strong baseline can go no further.** The only global completion signal is a **count**
(`NIU_MST_RD_RESP_RECEIVED` vs `noc_reads_num_issued`), and read responses take **dynamically
assigned VCs** so they can land out of order. A count never proves a *specific earlier* block
arrived, so nothing may be pushed until the barrier has drained everything.
`NIU_MST_REQS_OUTSTANDING_ID(trid)` is the only per-group completion signal — precisely the gap
transaction ids fill.

**The ring needs no counters.** Block *k* and block *k−T* carry the same id, so the id you are
about to tag with is exactly the one whose slot must be freed — one barrier both retires the old
block and makes the id reusable, and every quantity falls out of the loop index. No issue/wait
cursors, no in-flight counter:

```
trid       = (k % T) + 1
if k >= T  -> barrier_with_trid(trid); push the block it retires
in_flight  = min(k, T - 1)                  // slots the CB pointer lags by
drain      = the last min(full_blocks, T) ids, in issue order
finally    -> set_read_trid(noc, 0)         // restore untagged
```

(This is the formulation the helper library's own docstring gives, and what the fastest in-tree
gathers use.)

**The trap is the landing slot.** A circular buffer's write pointer only advances on
`cb_push_back`, so while `in_flight` blocks are issued-but-not-yet-pushed the CB pointer *lags*
them. The landing address for a new block is therefore `get_write_ptr() + in_flight` slots,
wrapped by hand at the end of the CB region — and you must
`cb_reserve_back((in_flight + 1) * block)` so that slot is genuinely free rather than one the
writer still owns. Get it wrong and you silently overwrite a block that is still in flight, which
is why the correctness test demands a **bitwise-exact** copy on every cell (both variants).

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.trid_double_issue [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--shape` | `H,W` | `512,512` | tile-aligned tensor (256 tiles) |
| `--cores` | int | `1` | cores running the copy (each independent) |
| `--blocks` | int list | `1,2,4,8,16` | pages-per-barrier sweep |
| `--trids` | int list | `2,3,4` | pipeline depth sweep (blocks in flight) |
| `--ahead` | int list | `1,2,3,4` | baseline strength: blocks issued per barrier (1 = naive loop, >1 = strongest non-trid reader) |
| `--cb-blocks` | int | `6` | CB depth in blocks; held FIXED across the table so it is not a confound (must be ≥ max trid depth) |
| `--dtype` | `bfloat8_b\|bfloat16\|float32` | `bfloat16` | tile format = transfer size (~1088 / 2048 / 4096 B) |
| `--iters` | int | `1` | in-kernel repeat of the page range (1 = latency, large = steady-state) |
| `--trials` | int | `10` | profiled launches per case (averaged) |

```bash
# the headline table
python -m ttnn.operations.examples.trid_double_issue

# is the saving a latency or bandwidth? sweep the transaction size
python -m ttnn.operations.examples.trid_double_issue --dtype float32 --blocks 1,4 --trids 2,4

# does it still pay with more cores?
python -m ttnn.operations.examples.trid_double_issue --cores 6 --blocks 1,4,16 --trids 2,4
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

**CB allocation is identical in every cell of a block row** (`cb_blocks × block × page_bytes`,
`cb_blocks=6`). `slots` is what the reader actually reserves at its peak, so a `trid ×N` row is
**iso-L1** against the `ahead=N` row.

**Tune the baseline first — `ahead=3` is its optimum here.** With `cb_blocks=6` that splits the CB
evenly between the reader's window and the writer's lag; `ahead=4` leaves the writer 2 slots and
regresses. A sweep that skips 3 badly understates the baseline (mine did, at first).

```
cores=1  [512,512] bf16 (256 tiles)  cb_blocks=6 allocated in every cell        GB/s
  block   base a=1   base a=2   base a=3   base a=4 |  trid x2   trid x3   trid x4
      1       10.1       17.6      *24.2       23.5 |     19.2      29.0     *37.3
      2       17.9       30.2      *39.4       35.6 |     34.9      51.7     *64.3
      4       32.7       49.2      *62.2       53.9 |     62.1     *84.9      84.5
      8       52.1       73.6      *85.1       68.4 |     97.2     *99.4      97.5
     16       76.9       93.2      *97.8       77.3 |    *108.8    104.2     100.1
                                    (* = best in its family)
```

| | best base | best trid | gain |
|---|---:|---:|---:|
| block 1 | 24.2 | 37.3 | **1.54×** |
| block 2 | 39.4 | 64.3 | **1.63×** |
| block 4 | 62.2 | 84.9 | **1.36×** |
| block 8 | 85.1 | 99.4 | **1.17×** |
| block 16 | 97.8 | 108.8 | **1.11×** |

**The honest headline is 1.11–1.63×** over a properly tuned non-trid reader at the same allocated
CB. Against the *naive* one-block-per-barrier loop it reads up to 3.7×, but batching alone recovers
most of that — so the naive number is not the one to quote.

**Depth is what pays, not tagging.** At `block=4`, `trid ×2` (62.1) exactly *ties* the tuned
baseline (62.2). A shallow ring over a well-tuned baseline buys nothing; the wins are at depth 3–4.

**What it actually buys is L1, not peak bandwidth.** Widening `block` buys the same depth and costs
L1. Sweep it and everything converges to the ~122 GB/s single-core ceiling — at a 768 KB CB the
*simplest possible reader* (one block, one barrier, no ring, no ids) is within **3%** of the best
number in the study:

| L1 for the CB | simplest loop | best baseline | best trid | trid vs best base |
|---:|---:|---:|---:|---:|
| 96 KB | 52.7 | 89.8 | **107.1** | 1.19× |
| 192 KB | 79.1 | 111.4 | **119.5** | 1.07× |
| 384 KB | 102.9 | 120.0 | **123.9** | 1.03× |
| 768 KB | 118.7 | 120.9 | **122.2** | 1.01× |

trid at 96 KB (107.1) needs a baseline somewhere between 96 and 192 KB to match; trid at 192 KB
(119.5) matches the baseline's 384 KB. So the saving is **~1.5–2× less L1**, shrinking as the
budget grows.

**It is a latency, not bytes** — 1.54–1.59× at block=1 across bfp8 → fp32, a 3.8× byte range — and
**per core**, holding essentially undiminished to 6 cores at 346 GB/s (1.54× → 1.48× at block=1).

**Where it does not apply:** if L1 is free, widen the block and write the trivial loop — it reaches
the ceiling on its own. If you have not yet tuned `ahead`, do that first; it is the bigger and
simpler win. Trids are for a block capped by a shard size, a wide tensor or co-resident buffers,
where you still want the depth. The fully DRAM-saturated grid was not measured here.

**This is also why the two code paths here are the same size** (17 vs 23 lines): the complexity is
not the ids, it is keeping N blocks in flight in a circular buffer at all, and both pay it. The
trid-specific part is four lines — so once you are already batching, adding ids is nearly free.

## Run the predefined sweep
```bash
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_trid_double_issue.py::test_trid_double_issue_correctness
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_trid_double_issue.py::test_trid_double_issue_device_perf
```

## Code
- [`kernels/trid_reader.cpp`](kernels/trid_reader.cpp) — the kernel under study; both barrier
  disciplines live here behind `if constexpr (num_trids == 0)`, so the surrounding scaffolding is
  provably identical, and everything the two share — ring addressing, issuing a block, the
  sub-block tail — is factored into three lambdas written once. Uses `dataflow_kernel_lib` for the
  trid tag/wait pair.
- [`kernels/trid_writer.cpp`](kernels/trid_writer.cpp) — held constant across variants.
- [`trid_double_issue.py`](trid_double_issue.py) — program descriptor, CB sizing, work split.
