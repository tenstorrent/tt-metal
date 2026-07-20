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
```

## Code
`noc_placement.py` (op), `kernels/{copy_reader,copy_writer,read_bench,write_bench}.cpp`, `noc_report.py`
(report pipeline). Numbers are single-box, single-arch (WH B0) — illustrative of the mechanism, not a CI bound.
