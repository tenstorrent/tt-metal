# noc_placement — placement × NoC × operation

**Measured on:** WH B0 n150 (`bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`, 2026-07-17; regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

Two kernel-level decisions govern how interleaved DRAM traffic contends on the Network-on-Chip,
and this example lets you switch and measure both:

1. **Where you place the line of cores** — spreading it across the DRAM-facing axis (a **row** or
   **diagonal**) beats stacking it on one axis (a **column**): same kernel, same work, ~2.2–2.9× less time on both WH B0 n150 and BH p150b.
   A column line is exactly what `split_work_to_cores(..., row_wise=False)` (the default) hands you.
2. **Which NoC a stream uses** — **reads want NoC0, writes want NoC1** (the tt-metal default) for the
   spread placements you actually use. It is not that NoC0 is intrinsically faster: NoC0's routing
   *disperses* column-localized DRAM traffic while NoC1's *concentrates* it (mirror-image for writes).

**Op:** `noc_placement(t, *, op, noc, placement, num_cores=N, block=B, kernel_iters=K)` — interleaved DRAM
data movement over a line of `N` cores. The kernels are byte-identical across placements; only the
physical cores and the chosen NoC(s) move, so any measured difference is pure NoC routing.

## The switch space (cases are switchable)
```python
from ttnn.operations.examples.noc_placement import noc_placement, OPS, NOCS, PLACEMENTS
# OPS=("read","write","copy")  NOCS=("noc0","noc1")  PLACEMENTS=("column","row","diagonal")

noc_placement(t, op="read",  noc="noc0", placement="row")       # DRAM->L1 read bench on NoC0
noc_placement(t, op="write", noc="noc1", placement="diagonal")  # L1->DRAM write bench on NoC1
noc_placement(t, op="copy",  noc="noc0", placement="column")    # identity copy: reads NoC0, writes NoC1
```
- `op="read"` / `op="write"` isolate a **single stream** (one kernel, a fixed L1 scratch, no CB handshake
  and no partner kernel) so the NoC-for-reads vs NoC-for-writes question is answered without cross-contention.
- `op="copy"` is the realistic identity DRAM→DRAM copy: reads on `noc`, writes on the **other** NoC
  (so the two streams never share links). `noc="noc0"` = the canonical reader-NoC0 / writer-NoC1 pairing.
- Reads are issued in **blocks** (`block` pages per NoC barrier) so many transactions are in flight — this
  makes the movement **bandwidth-bound**, which is what exposes link *contention* (at `block=1` it is
  latency-bound and the effect nearly vanishes). `kernel_iters` repeats the range for steady-state throughput.

## 1. Best placement (op="copy")
Wormhole B0, `shape=(1024,2048)` = 2048 tiles, 8 cores, `block=16` (see `report.md` for the stamp):

```
    placement          ns/op   vs column
    column          959160.9  (baseline)
    row             330068.6  -> 2.91x
    diagonal        327144.2  -> 2.93x
```
row/diagonal are ~2.2–2.8× faster than column, and the gap **grows with core count** — the contention
signature: a column line saturates its shared NoC links and stops scaling. **Row ≈ diagonal** on WH B0 n150
and on BH p150b too (BH p150b, same copy: column 383 253 → row 164 624 ns 2.33×, diagonal 165 034 ns 2.32×) — the
diagonal's edge over the row once expected on Blackhole's asymmetric grid does not show on BH p150b at 8 Tensix cores.
`split_work_to_cores(..., row_wise=True)` spreads across the DRAM-facing axis instead of the column trap.

## 2. Which NoC for reads vs writes (op="read"/"write")
DRAM is **column-localized** — on Wormhole DRAM lives only in NoC columns 0 and 5. The two NoCs route
[dimension-ordered on a torus](../../../../../METALIUM_GUIDE.md): **NoC0 = east→south**, **NoC1 = north→west**.

- **NoC0 (reads):** traffic fans east across columns first and only turns south at the reader's *own*
  column → spread the readers across columns and no link is shared.
- **NoC1 (writes):** the mirror — traffic climbs the *DRAM* column first (concentrating on cols 0/5), then
  runs along the consumer's row.

Measured 12-cell matrix (device ns/op; see `report.md` for the stamped block, `noc_placement_matrix.html`
for the per-cell NoC link-demand heatmaps):

```
    op     placement     noc0 ns    noc1 ns   faster
    read   row            ~202k      ~969k     NoC0 4.8x
    read   diagonal       ~204k      ~503k     NoC0 2.5x
    read   column         ~789k      ~514k     NoC1 1.5x   (column = the trap; never use it)
    write  row           ~1044k      ~244k     NoC1 4.3x
    write  diagonal       ~628k      ~246k     NoC1 2.5x
    write  column         ~626k      ~921k     NoC0 1.5x
```
- **Mirror symmetry:** `read·NoC0 ≈ write·NoC1` and `read·NoC1 ≈ write·NoC0` — each pair is the same
  physical links traversed in opposite directions (writes run a touch slower from store overhead).
- **The default pairing wins:** reads→NoC0 + writes→NoC1 gives the two fast cells for row/diagonal and
  keeps the two streams on separate physical networks. (`ReaderConfigDescriptor`/`WriterConfigDescriptor`
  default to exactly this.)
- The column exception (where the reverse NoC wins) is a property of the pathological column placement —
  not a reason to swap the default.
- **BH p150b** (DRAM in NoC columns 0 and 9; 8 Tensix cores, same shape, `block=16`, `kernel_iters=8`): the same
  winner in every row — read row NoC0 106k vs NoC1 376k ns (3.5×), read diagonal 107k vs 151k (1.4×), read column
  NoC1 153k vs NoC0 292k (1.9×); write row NoC1 105k vs NoC0 402k (3.8×), write diagonal 111k vs 196k (1.8×), write
  column NoC0 195k vs NoC1 371k (1.9×). Stamped block in [`report.md`](report.md).

## When it helps / when it does not
**Rule:** both levers pay once the line of Tensix cores is **bandwidth-bound** — on WH B0 n150 from ≳16 outstanding
2 KB reads (Tensix cores × `block`); at ≤ 8 they are within ~10%. Same verdicts on BH p150b, with smaller gains
(2.2–2.5× vs 2.9–3.3×). For a single Tensix core the NoC choice does not help on either board — WH B0 n150 even
loses 6% with the canonical pairing, BH p150b is flat. Identity copy (`op="copy"`), `shape=(1024, 2048)` = 2048
bf16 tiles (2 KB pages), interleaved DRAM (`TensorMemoryLayout::INTERLEAVED`); line length capped at 8 Tensix
cores so a diagonal fits. "Outstanding reads" = Tensix cores × `block` (2 KB pages in flight per barrier per
Tensix core).

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Placement helps:** column → row, 8 Tensix cores, `block=16` (128 outstanding reads) | 246 765 → 84 917 ns **2.91×** | 97 835 → 43 723 ns **2.24×** |
| **Placement does not help:** column → row, 2 Tensix cores, `block=1` (2 outstanding reads, latency-bound) | 1 106 889 → 1 077 065 ns **1.03×** | 838 190 → 829 478 ns **1.01×** |
| **NoC helps:** reads NoC1 → NoC0 (writes on the other NoC), row of 8 Tensix cores, `block=16` | 278 002 → 85 066 ns **3.27×** | 107 508 → 43 806 ns **2.45×** |
| **NoC does not help:** reads NoC1 → NoC0, 1 Tensix core at (0,0), `block=16` | 404 600 → 429 238 ns **0.94× (loses)** | 216 865 → 211 690 ns **1.02× — flat on BH, no loss** |
| Copy column → row, 8 Tensix cores, `block=16`, `kernel_iters=8` (device_perf) | 959 161 → 330 069 ns **2.91×** (diagonal 2.93×) | 383 253 → 164 624 ns **2.33×** (diagonal 2.32×) |
| Sweep: column → row, 4 Tensix cores × `block=1` (4 outstanding) | **1.04×** | not measured |
| Sweep: column → row, 8 Tensix cores × `block=1` (8 outstanding) | **1.09×** | not measured |
| Sweep: column → row, 8 Tensix cores × `block=2` (16 outstanding) | **1.40×** | not measured |
| Sweep: column → row, 2 Tensix cores × `block=16` (32 outstanding) | **1.31×** | not measured |
| Sweep: column → row, 4 Tensix cores × `block=16` (64 outstanding) | **2.20×** | not measured |
| **Boundary** | ≳16 outstanding 2 KB reads | between 2 (flat) and 128 (helps) outstanding reads; not swept |

**Why the boards differ:**
- **Smaller gains on BH p150b.** BH's NoC flit is 64 B (WH 32 B), so each link carries twice the bytes and a
  column line's shared links saturate less; the column baseline is ~2.5× faster on BH p150b (97.8 vs 246.8 µs)
  while the row lever is ~1.9× faster. Consistent with the link-width difference, not isolated.
- **Single Tensix core:** WH B0 n150 loses 6% with reads on NoC0 at (0,0); BH p150b does not (1.02×). The DRAM
  cores sit in different NoC columns (WH 0 and 5, BH 0 and 9), so the one-core route differs; not investigated.

- **Placement helps** when the line's outstanding reads contend: a column funnels every Tensix core's DRAM traffic
  through the same column links, a row spreads it across columns, so the row moves the same bytes 2.2–2.9× faster.
- **Placement stops paying** when the line is latency-bound (one page per barrier, few Tensix cores): no link is
  saturated, so each read costs one round trip wherever the Tensix core sits.
- **NoC0 for reads helps** on a spread line because NoC0 (east→south) fans the traffic out across columns before
  turning toward DRAM, while NoC1 (north→west) concentrates it on the DRAM columns.
- **The NoC choice does not help** without a line to disperse: one Tensix core has nothing to spread — keep the
  default for multi-core lines, but do not expect it to help a single-Tensix-core kernel.

Small payloads still benefit on WH B0 n150 if the line is bandwidth-bound: 64 tiles (8 tiles per Tensix core,
`block=16`, `kernel_iters=1`) measured 2.65×, 16 tiles (2 tiles per Tensix core) 1.51×. The NoC choice for reads is
likewise flat when latency-bound (`block=1`: 1.04× at 2 Tensix cores, 1.07× at 8 Tensix cores; `op="read"`, row).

*How measured:* `test_noc_placement_regimes`, `DEVICE KERNEL DURATION [ns]`, baseline/lever interleaved
(A,B,A,B…), median of 12 trials × 3 launches, `kernel_iters=2` unless stated.
- **WH B0 n150:** commit `adce58258c3` + test, 2026-10-12 (three runs agree within 0.01×); sweep rows from
  tt-probe (`kernel_iters=8`); device_perf row from the 2026-07-17 report ([`report.md`](report.md)).
- **BH p150b:** the same four cases (the fourth, then named `noc_loses_1core_block16`, is now
  `noc_flat_1core_block16`); device_perf row from `test_noc_placement_device_perf` (20 launches averaged).
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).

## Measure your own shapes/params
```bash
python -m ttnn.operations.examples.noc_placement --shape 2048,2048 --cores 8 --block 32
#   --shape H,W   tile-aligned bf16 tensor      --cores  line length (<= min grid dim)
#   --block N     pages in flight per barrier   --kernel-iters N  in-kernel repeat   --iters N  profiled launches
```

## Regenerate the report (`noc_placement_matrix.html`)
Unlike the other examples, the matrix visual is **reconstructed from code + tt-npe** (not hand-written).
`noc_report.py` drives: device-ns pass → NoC-trace capture (`profile_this.py --collect-noc-traces`) →
tt-npe simulation → aggregate → HTML. It needs [tt-npe](https://github.com/tenstorrent/tt-npe) built and
reachable via `$TT_NPE_HOME` (or at `<tt-metal>/../tt-npe`):
```bash
python -m ttnn.operations.examples.noc_placement --report   # rewrites noc_placement_matrix.html + report.md
```

## Run the committed sweep
```bash
# correctness (every copy placement/NoC is the identical identity copy)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_correctness
# device kernel duration for the 12-cell read/write × NoC × placement matrix
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_device_perf
# helps / does-not-help regimes (asserted speedups)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes
```

## Code
`noc_placement.py` (op), `kernels/{copy_reader,copy_writer,read_bench,write_bench}.cpp`, `noc_report.py`
(report pipeline). Numbers are single-box per arch (WH B0 n150, BH p150b) — illustrative of the mechanism, not a CI bound.
