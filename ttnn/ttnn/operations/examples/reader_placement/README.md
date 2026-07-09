# reader_placement

**Trick:** where you place a *line* of reader cores on the grid decides how badly their
DRAM traffic contends on shared NoC links. Spreading the line across the DRAM-facing axis
(a **row** or a **diagonal**) beats stacking it on one axis (a **column**) — same kernel,
same work, ~2.8× less time. A column line is exactly what `split_work_to_cores(..., row_wise=False)`
(the default) hands you.

**Op:** `reader_placement(t, placement="column"|"row"|"diagonal", num_cores=N, block=B)` — a
pure interleaved DRAM→DRAM identity copy over `N` cores. Reader on NoC0, writer on NoC1. The
reader/writer kernels are byte-identical for all three placements; **only the physical cores
that run them move**, so any measured difference is attributable purely to the NoC routes.

## What it isolates
One kernel-level decision: **the core positions you assign work to**. Compute is nil (a copy);
the tensor is interleaved DRAM; `N` cores form a line placed as:

- `column` — left column `(0,0)..(0,N-1)` — the `split_work_to_cores` default (`row_wise=False`). Baseline / trap.
- `row` — top row `(0,0)..(N-1,0)`.
- `diagonal` — `(i,i)` — the mcast-friendly line (readers spread across both axes).

Reads are issued in **blocks** (`block` pages per NoC barrier) so many transactions are in
flight: this makes the copy **bandwidth-bound**, which is what exposes link *contention*. With
one-read-per-barrier it would be *latency*-bound and the effect nearly vanishes (~5–9%).

This is an identity copy, so placement moves **both** the read stream (NoC0, DRAM→core) and the
write stream (NoC1, core→DRAM) — it measures DRAM-*traffic* contention, not reads in isolation.
That is on purpose: it is exactly the situation `split_work_to_cores` puts you in (read+write of
an interleaved tensor across a grid).

## Measured (see `report.md` for the full stamp: box, arch, commit, date)

Wormhole B0, 8×8 grid, `shape=(1024,2048)` = 2048 tiles, `block=16`, 20 launches averaged
(in-process device profiler):

```
    cores  placement          ns/op   vs column
        4  column          131837.9  (baseline)
        4  row              60015.8  -> 2.20x
        4  diagonal         59367.1  -> 2.22x
        8  column          125303.4  (baseline)
        8  row              44131.9  -> 2.84x
        8  diagonal         44299.1  -> 2.83x
```

**Reading it:** row/diagonal are ~2.2–2.8× faster than column, and the gap *grows* with core
count — the contention signature. The tell: 4→8 cores, **column barely improves (132→125 µs)**
while **row scales (61→44 µs)**. A column line saturates its shared vertical NoC links, so
piling on more cores buys almost nothing; a spread-out line keeps scaling.

On this Wormhole box **row ≈ diagonal** (both spread across the DRAM-facing x-axis; that is what
matters here). The diagonal's *edge over row* is a Blackhole story — an asymmetric grid where a
row-mcast wants readers off the first column; not reproducible on this symmetric WH part.

## Measure your own shapes/params
```bash
python -m ttnn.operations.examples.reader_placement --shape 2048,2048 --cores 8 --block 32
#   --shape H,W   tile-aligned bf16 tensor      --cores  comma list of line lengths (<= min grid dim)
#   --block N     pages in flight per barrier    --iters  profiled launches per case (averaged)
```

## Run the committed sweep
```bash
# correctness (every placement is the identical identity copy)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_reader_placement.py::test_reader_placement_correctness
# device kernel duration, column vs row vs diagonal
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_reader_placement.py::test_reader_placement_device_perf
```
