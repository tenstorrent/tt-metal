# reader_placement — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` |
| arch | Wormhole B0 (8×8 compute grid, DRAM grid 12×1) |
| commit | `6c68e8f0916` |
| date | 2026-07-09 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 5 warmup + 20 timed launches per case, flush-bracketed, on-device duration averaged; whole matrix in one device session |

> Numbers are illustrative of the *effect*, not a CI bound — they are single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.reader_placement` to measure your own params.
> A different arch (e.g. Blackhole) should be appended as a new block, not overwritten.

## Config A — `shape=(1024,2048)` (2048 tiles, 4 MB), `block=16`

```
    cores  placement          ns/op   vs column
        4  column          131837.9  (baseline)
        4  row              60015.8  -> 2.20x
        4  diagonal         59367.1  -> 2.22x
        8  column          125303.4  (baseline)
        8  row              44131.9  -> 2.84x
        8  diagonal         44299.1  -> 2.83x
```

## Config B — `shape=(2048,2048)` (4096 tiles, 8 MB), `block=32`, 8 cores

```
    cores  placement          ns/op   vs column
        8  column          250451.2  (baseline)
        8  row              85601.0  -> 2.93x
        8  diagonal         84635.6  -> 2.96x
```

## Findings
- Direction matches the hypothesis: **row ≈ diagonal < column** (row/diagonal ~2.2–3.0× faster).
- The gap **grows with core count and block depth** — the NoC-contention signature. Column
  barely improves 4→8 cores (132→125 µs) while row scales (60→44 µs): a column line saturates
  its shared vertical NoC links.
- **row ≈ diagonal on this symmetric Wormhole part.** The diagonal's edge *over* row is a
  Blackhole (asymmetric grid) concern, not reproducible here.
- Effect is only visible when the copy is **bandwidth-bound** (`block` large enough to keep many
  transactions in flight). At `block=1` it is latency-bound and the delta collapses to ~5–9%.
- In-process profiling cross-validates the `run_device_perf` subprocess path to ~0.3% and runs
  the whole sweep ~10× faster (one device open + JIT, no per-case subprocess).
