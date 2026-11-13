# tensix_all_reduce_row_vector — per-row-vector all-reduce: bytes on the wire and the return path

**Problem type:** cross-core communication  ·  **Concept(s):** for a per-row-vector payload, bytes per member on the wire (both directions) and the return path
**First profiled on:** `bh-50-special-adamjanovic-for-reservation-143315` · BH p150b 13×10 grid · AICLK 1350 MHz · 2026-10-31 · `fc9611c72e8`
**Wormhole B0 numbers:** `bgd-lab-22-special-adamjanovic-for-reservation-165935` · WH B0 n150 8×8 grid · AICLK 1000 MHz · 2026-11-13 · `b31c663a467` ([`report_wh_n150.md`](report_wh_n150.md))

> Reading order: [`../master.md`](../master.md) → [`../tensix_all_reduce/README.md`](../tensix_all_reduce/README.md) (the whole-tile all-reduce this specializes) → **this file** → run the CLI, and read the code only if you need to.

## Bytes on the wire and the return path

**The special case.** [`tensix_all_reduce`](../tensix_all_reduce/README.md) all-reduces whole tiles elementwise. Very often each member only
contributes **one value per row**: a row reduction (sum, max, the statistics of a norm) split across K
Tensix cores with the input `TensorMemoryLayout::WIDTH_SHARDED`. Each member row-reduces its slice
(`REDUCE_ROW`, float32 DST), which leaves its partial in **column 0** of one float32 tile per tile-row;
one **owner** per tile-row adds the K partials, and every member needs the whole-row result back.
`row_sum_all_reduce()` does exactly that: input bf16 `[b·32, K·w·32]` WIDTH_SHARDED over K members
(b tile-rows, w tiles per member per tile-row), output float32 `[b·32, K·32]` over the same members,
column 0 = sum over the whole row, the other 31 columns don't-care. The result stays the
**un-expanded column-0 tile**; a consumer applies it with a column broadcast (`mul_tiles_bcast_cols`).

The row reduce and the owner placement (owners on a diagonal of the members' rectangle, so no two share a
grid column) are identical in every variant. Two kernel-level choices vary:

| Choice | Option | Bytes per member per tile-row | What the owner / root does |
|---|---|---:|---|
| Partial on the wire (member -> owner) | `tile_partial_*` | 4096 B | adds K tiles (paired `add_tiles`, float32 DST) |
| | `faces_partial_*` (faces 0/2 hold column 0) | 2 × 1024 B | adds K tiles; faces 1/3 are don't-care |
| | `row_partial_*` (member transposes; 32 datums = row 0) | 2 × 64 B | row m % 32 of gather tile m // 32; column-reduces ⌈K/32⌉ tiles, transposes back |
| Return payload (owner -> members) | whole column-0 tile (default) | 4096 B | |
| | `*_faces_return_*` | 2 × 1024 B | |
| Return path | `*_unicast` | K writes per owner | every owner writes all K members, then bumps K semaphores |
| | `*_mcast` | 1 multicast | owners write the root (member 0); the root multicasts b tiles to the members' rectangle, then the semaphore |

Rows of the last gather tile that no member writes are zero-filled with NoC reads of the hardware zeros
region (RISC-V stores would cost more than the whole combine at small K). The member's gather kernel
waits for this iteration's result before shipping the next partial, so an owner's gather slot is never
overwritten while it is being read.

**Measured** (BH p150b 13×10 grid, `bh-50`, 1350 MHz, w = 1, kernel-iters 1, median of 5 interleaved
trials, ns per launch; full tables and the 17 regimes in [`report.md`](report.md)):

| K (members) | b | `tile_partial_unicast` (baseline) | `faces_partial_unicast` | `faces_partial_mcast` | `row_partial_mcast` | `faces_partial_faces_return_unicast` | fastest vs baseline |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1 | 1 867 | **1 838** | 2 179 | 2 479 | 1 844 | faces + unicast 1.02× |
| 4 | 1 | 2 161 | 2 079 | 2 260 | 2 513 | **2 067** | faces + unicast (either return) 1.05× |
| 8 | 1 | 2 707 | 2 524 | **2 396** | 2 549 | 2 570 | faces + mcast 1.13× |
| 12 | 1 | 3 266 | 3 021 | **2 563** | 2 577 | 3 055 | faces/row + mcast 1.27× |
| 16 | 1 | 4 007 | 3 725 | 3 152 | **3 035** | 3 817 | row + mcast 1.32× |
| 64 | 1 | 10 948 | 9 314 | 5 156 | **3 570** | 9 827 | row + mcast 3.07× |
| 128 | 1 | 20 012 | 16 869 | 7 858 | **4 321** | 17 834 | row + mcast 4.63× |
| 4 | 4 | 2 832 | 2 716 | 2 967 | 3 328 | **2 463** | faces + faces return, unicast 1.15× |
| 8 | 4 | 4 483 | 3 859 | 3 444 | **3 353** | 3 425 | row + mcast 1.34× (3 within 3%) |
| 12 | 4 | 6 065 | 5 086 | 3 919 | **3 428** | 4 397 | row + mcast 1.77× |
| 16 | 4 | 6 210 | 5 276 | 4 225 | **3 930** | 4 595 | row + mcast 1.58× |
| 64 | 4 | 14 899 | 12 973 | 6 487 | **4 530** | 10 989 | row + mcast 3.29× |
| 128 | 4 | 25 546 | 21 733 | 9 355 | **5 424** | 19 622 | row + mcast **4.71×** |

**Reading - one mechanism per lever:**
- **Whole tile never wins.** Faces 0/2 halves the bytes into the owner for the same number of writes:
  1.02–1.19× with unicast return, up to 1.42× with multicast (K = 128, b = 4: 13 261 → 9 355 ns), never below
  1.01×. Below ~4 KB, fewer NoC commands matter more than fewer bytes, so the gain is small at small K.
- **Row partial (128 B) pays two transposes, a transpose back and the zero-fill** for 16× fewer bytes into
  the owner and ⌈K/32⌉ instead of K tile-adds. That only pays once the owner's inbound bytes dominate: it
  loses to faces at K ≤ 8, b = 1 (0.83–0.94×), breaks even at K = 12–16, b = 1 (0.99–1.04×), wins at
  K = 12–16 with b = 4 (1.07–1.14×) and at K ≥ 32 at any b (1.12–1.82×).
- **Multicast return** replaces each owner's K writes of 4096 B (b·K writes in all, on b owners) with one
  owner -> root write each and one root multicast. It loses at K ≤ 4 (0.84–0.93×: an extra hop through the
  root), is marginal at K = 8 (1.05–1.13×), and wins from K = 12 (1.15–3.22×, growing with K).
- **Faces 0/2 on the return** helps only on a unicast return with b = 4 (1.10–1.18×: b·K writes of
  4096 B become 2·b·K writes of 1024 B on a byte-bound owner). It is null at b = 1 (0.95–1.01×: two 1 KB writes
  cost about one 4 KB write) and always loses under multicast (0.67–0.81× at b = 4, 0.93–0.97× at b = 1):
  2b small multicasts instead of one b × 4096 B multicast.

**Rule:** never ship the whole tile. K ≤ 8: faces 0/2 + unicast return (K = 8: multicast is already
1.05–1.13×; at b ≥ 4 trim the unicast return to faces 0/2). K = 12–16: faces or row at b = 1, row at
b ≥ 4, multicast return. K ≥ 32: row + multicast. Return the un-expanded column-0 tile; trim it to
faces 0/2 only on a unicast return with b ≥ 4, never under multicast.

**Measured on WH B0 n150** (8×8 grid, 1000 MHz, w = 1, kernel-iters 1, median of 5 interleaved trials, ns per
launch; full tables in [`report_wh_n150.md`](report_wh_n150.md); K = 128 needs 128 Tensix cores):

| K (members) | b | `tile_partial_unicast` (baseline) | `faces_partial_unicast` | `faces_partial_mcast` | `row_partial_mcast` | `faces_partial_faces_return_unicast` | fastest vs baseline |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1 | 2 943 | 2 805 | 3 058 | 3 552 | **2 682** | faces + faces return, unicast 1.10× |
| 4 | 1 | 3 753 | 3 469 | 3 397 | 3 588 | **3 172** | faces + faces return, unicast 1.18× |
| 8 | 1 | 5 310 | 4 760 | 3 980 | **3 625** | 4 194 | row + mcast 1.46× |
| 12 | 1 | 7 179 | 6 391 | 5 125 | **4 252** | 5 522 | row + mcast 1.69× |
| 16 | 1 | 8 823 | 7 674 | 5 658 | **4 282** | 6 527 | row + mcast 2.06× |
| 32 | 1 | 15 198 | 12 777 | 8 075 | **4 496** | 10 561 | row + mcast 3.38× |
| 64 | 1 | 27 992 | 23 252 | 12 775 | **5 037** | 18 649 | row + mcast 5.56× |
| 4 | 4 | 5 937 | 5 207 | 5 353 | 5 627 | **4 473** | faces + faces return, unicast 1.33× |
| 8 | 4 | 10 465 | 8 426 | 6 809 | **5 657** | 6 884 | row + mcast 1.85× |
| 12 | 4 | 11 696 | 9 542 | 7 877 | **6 527** | 7 838 | row + mcast 1.79× |
| 16 | 4 | 14 784 | 11 962 | 8 611 | **6 582** | 9 264 | row + mcast 2.25× |
| 32 | 4 | 22 895 | 18 977 | 11 379 | **6 818** | 14 674 | row + mcast 3.36× |
| 64 | 4 | 37 362 | 30 989 | 16 291 | **7 348** | 23 467 | row + mcast 5.08× |

**Where WH B0 differs from BH:** the crossovers sit at smaller K. Row + multicast is already the fastest at
K = 8 (b = 1: 1.46×; BH 0.94× row over faces), and row over faces gains 1.10× at K = 8 and 1.20× at K = 12, b = 1
(BH 0.94× / 0.99×). Multicast return is null at K = 4 (tile partial 1.02×, BH 0.93×) instead of losing. Faces 0/2
on a unicast return also pays at b = 1 (K = 64: 1.24×; BH 0.95×), and still loses under multicast (K = 16,
b = 4: 0.86×). **WH rule:** never ship the whole tile; K ≤ 4: faces 0/2 + faces 0/2 unicast return; K ≥ 8: row +
multicast.

**Code:** `_RV_GATHER_KERNEL` (`FACES_PARTIAL`: two 1024 B face writes; `ROW_PARTIAL`: two 64 B face-row
writes into row m % 32), `_RV_COMPUTE_KERNEL` (`transpose_tiles()`, `REDUCE_COL` over ⌈K/32⌉ gather
tiles vs paired `add_tiles` over K), `_RV_RETURN_KERNEL` (`MCAST_RETURN` owners → root →
`noc_async_write_multicast`; `FACES_RETURN` two pieces per tile; the zeros-region fill), host side
`create_row_vector_program_descriptor()`, `row_owner_indices()`, `build_member_layout()`.

## CLI - measure your own shapes and parameters

```bash
python -m ttnn.operations.examples.tensix_all_reduce_row_vector [options]
```

The CLI runs `test_tensix_all_reduce_row_vector_sweep` (an opt-in measurement sweep; the CLI sets its
`AR_RV_SWEEP=1` switch).

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `recommended` \| `all` \| any row-vector variant name | `recommended` | `recommended` = the 8 measured ones, `all` = all 12 combinations |
| `--members` | `K,K,...` | `2,4,8,12,16,32,64,128` | Tensix cores the row is split across |
| `--tile-rows` | `b,b,...` | `1,4` | tile-rows per member (b ≤ K, one owner per tile-row) |
| `--width-tiles` | `w,w,...` | `1` | bf16 tiles per member per tile-row |
| `--trials` | int | `5` | interleaved profiler trials per variant |
| `--kernel-iters` | int | `1` | all-reduces per kernel launch; `1` measures latency, larger values measure steady state |
| `--report` | path | print only | also append the Markdown tables to this file |

```bash
# What to ship and how to return it, for a row split over 8 and 64 Tensix cores.
python -m ttnn.operations.examples.tensix_all_reduce_row_vector \
  --members 8,64 --tile-rows 2 --variant tile_partial_unicast faces_partial_unicast row_partial_mcast
```

## Reproduce

The 17 regimes were measured on BH p150b, and the 15 that fit 64 Tensix cores also on WH B0 n150 (min–max of
the regime test's median over 10 runs). Each asserts against a bound derived from the speedup measured on the
board it runs on (`perf_test_utils.regime_bound`, with one confirmation re-measure) and is skipped on a grid
with fewer than K Tensix cores. Three flip between the boards (a loss or null on BH, a gain on WH) and are
`measure` regimes that only log: `row_over_faces_measure_K8_b1_mcast` (WH 1.08–1.11×, BH 0.94×),
`row_over_faces_measure_K12_b1_mcast` (WH 1.19–1.21×, BH 0.99×), `faces_return_measure_K64_b1_unicast`
(WH 1.24–1.25×, BH 0.95×).

```bash
# correctness (all 12 partial x return x path combinations, exact vs torch) and the 17 regimes
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_row_vector.py
# the sweep behind the table (opt-in; AR_RV_REPORT=<path> appends it), or the CLI above
AR_RV_SWEEP=1 AR_RV_TILE_ROWS=1 scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_row_vector.py::test_tensix_all_reduce_row_vector_sweep
```

## Code

The complete host descriptor and inline dataflow/compute kernels are in
`program_descriptor_with_inline_kernels.py`; `__main__.py` provides the CLI.
