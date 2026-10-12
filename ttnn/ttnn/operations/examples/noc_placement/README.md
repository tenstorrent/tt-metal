# noc_placement — placement × NoC × operation

Two kernel-level decisions govern how interleaved DRAM traffic contends on the Network-on-Chip,
and this example lets you switch and measure both:

1. **Where you place the line of cores** — spreading it across the DRAM-facing axis (a **row** or
   **diagonal**) beats stacking it on one axis (a **column**): same kernel, same work, ~2.2–2.8× less time.
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
signature: a column line saturates its shared NoC links and stops scaling. On this symmetric WH part
**row ≈ diagonal**; the diagonal's edge over row is a Blackhole (asymmetric-grid) story.
`split_work_to_cores(..., row_wise=True)` spreads across the DRAM-facing axis instead of the column trap.

## 2. Which NoC for reads vs writes (op="read"/"write")
DRAM is **column-localized** — on Wormhole DRAM lives only in NoC columns 0 and 5. The two NoCs route
[dimension-ordered on a torus](../../../../METALIUM_GUIDE.md): **NoC0 = east→south**, **NoC1 = north→west**.

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

## When it helps / when it does not
Board: **Wormhole B0 n150** (8×8 Tensix grid; line length is capped at 8 Tensix cores so a diagonal fits).
Numbers come from `test_noc_placement_regimes`: identity copy (`op="copy"`), `shape=(1024, 2048)` = 2048 bf16
tiles (2 KB pages), `kernel_iters=2`, baseline and lever measured interleaved (A,B,A,B…), median of
12 trials × 3 launches, spread = min–max over trials. "Outstanding reads" = Tensix cores × `block`
(2 KB pages in flight per barrier per core) — the traffic the line puts on the NoC at once.

| lever | condition | config | baseline ns | lever ns | speedup | trials | board |
|---|---|---|---|---|---|---|---|
| placement column → row | **helps:** bandwidth-bound line, 128 outstanding 2 KB reads | 8 cores, `block=16`, 256 tiles/core | 246 765 (246 316–247 525) | 84 917 (84 155–86 778) | **2.91×** | 12 | WH B0 n150 |
| placement column → row | **does not help:** latency-bound line, 2 outstanding reads | 2 cores, `block=1`, 1024 tiles/core | 1 106 889 (1 091 085–1 115 261) | 1 077 065 (1 076 902–1 077 855) | **1.03×** | 12 | WH B0 n150 |
| reads NoC1 → reads NoC0 (writes on the other NoC) | **helps:** spread row line, bandwidth-bound | row, 8 cores, `block=16` | 278 002 (276 867–279 155) | 85 066 (84 521–85 694) | **3.27×** | 12 | WH B0 n150 |
| reads NoC1 → reads NoC0 | **does not help (loses):** a single Tensix core at (0,0) | 1 core, `block=16`, 2048 tiles/core | 404 600 (404 361–404 747) | 429 238 (428 252–429 843) | **0.94×** | 12 | WH B0 n150 |

- **Placement helps** when the line's outstanding reads contend: a column funnels every core's DRAM traffic
  through the same column links, a row spreads it across columns, so the row moves the same bytes ~2.9× faster.
- **Placement stops paying** when the line is latency-bound (one page per barrier, few cores): no link is
  saturated, so each read costs one round trip wherever the core sits.
- **NoC0 for reads helps** on a spread line because NoC0 (east→south) fans the traffic out across columns before
  turning toward DRAM, while NoC1 (north→west) concentrates it on the DRAM columns.
- **The NoC choice does not help** without a line to disperse: one Tensix core at (0,0) has nothing to spread, and
  there the canonical reads-NoC0 / writes-NoC1 pairing is ~6 % *slower* than the reverse (stable across runs) —
  keep the default for multi-core lines, but do not expect it to help a single-core kernel.

Boundary sweep (probe, same copy and shape, `kernel_iters=8`, 12 trials × 3 launches, column → row):

| Tensix cores | `block` | outstanding 2 KB reads | speedup |
|---|---|---|---|
| 2 | 1 | 2 | 1.01× |
| 4 | 1 | 4 | 1.04× |
| 8 | 1 | 8 | 1.09× |
| 2 | 16 | 32 | 1.31× |
| 8 | 2 | 16 | 1.40× |
| 4 | 16 | 64 | 2.20× |
| 8 | 16 | 128 | 2.94× |

Rule of thumb on WH B0: placement is worth changing once the line has ≳16 outstanding 2 KB reads; at ≤8 it is
within ~10 %. Small payloads still benefit if the line is bandwidth-bound: 64 tiles (8 tiles/core, `block=16`,
`kernel_iters=1`) measured 2.65×, 16 tiles (2 tiles/core) 1.51×. The NoC choice for reads is likewise flat when
latency-bound (`block=1`: 1.04× at 2 cores, 1.07× at 8 cores; `op="read"`, row).

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
(report pipeline). Numbers are single-box, single-arch (WH B0) — illustrative of the mechanism, not a CI bound.
