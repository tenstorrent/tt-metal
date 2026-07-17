# split_reader — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-08-special-astancov-for-reservation-44672` |
| arch | Wormhole B0 (8×8 compute grid) |
| base commit | `89089ffd11e` |
| date | 2026-07-16 |
| metrics | `DEVICE KERNEL DURATION`, `DEVICE NCRISC KERNEL DURATION`, and `DEVICE BRISC KERNEL DURATION` in ns, read in-process with `ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data` |
| method | 5 warmup + 20 timed launches per variant and transaction size, flush-bracketed, on-device duration averaged; all points in one device session |

> Numbers are illustrative of the effect, not a CI bound. Re-run the CLI on the target architecture
> and workload. Add results from another architecture as a new block rather than replacing this one.

## Config A — 8 source cores, 8 tiles/source, 8 tiles/block, transaction-size sweep

The input is height-sharded in L1 across logical source row `(0,0)..(7,0)`. Consumer `(0,1)`
gathers all 64 tiles and BRISC copies them to DRAM in their original order. Split-off puts all
reads on NCRISC. Split-on gives the first four tiles of every block to NCRISC and the second four
to BRISC. Blocks affect only read scheduling and bounded buffering; the DRAM tensor equals the
input exactly and no compute kernel runs.

Depending on transaction size, the consumer issues 64–4,096 reads to transfer the same 128 KiB
from row L1 into its two gather CBs, then writes the same 128 KiB to DRAM in both modes.

```text
 transaction  NoC reads   off NCRISC   on NCRISC   on BRISC  off device  on device   speedup
       32 B       4096     146303.2     74854.8    84739.8    147427.2    84767.1     1.74x
       64 B       2048      82415.9     43319.5    50895.2     83488.8    50925.7     1.64x
      128 B       1024      50434.9     29264.6    34809.9     51507.7    34837.6     1.48x
      256 B        512      34802.6     21928.8    26396.8     35836.3    26434.4     1.36x
      512 B        256      27276.9     18553.7    22523.0     28317.8    22563.5     1.26x
     1024 B        128      24350.0     17314.9    21121.6     25385.4    21163.8     1.20x
     2048 B         64      24051.2     17008.3    20740.8     25085.1    20787.0     1.21x
```

All duration columns are ns/op. The per-RISC columns are profiler kernel lifetimes and include NoC
barriers and CB waits; they are not instruction-level utilization counters. `on BRISC` also
includes the unchanged DRAM writer work.

## Findings

- With split off, NCRISC time rises from **24.1 us** at 64 reads to **146.3 us** at 4,096 reads.
  This is the main signal: more commands add RISC-V issue work while transferred bytes stay fixed.
- At 4,096 reads the single NCRISC is the device bottleneck. Split mode moves half of the read
  workload to BRISC: NCRISC finishes in **74.9 us**, BRISC in **84.7 us**, and device time improves
  from **147.4 us** to **84.8 us** (**1.74×**).
- The gain falls to about **1.20×** at 1–2 KiB transactions, where RISC-V issue overhead is no
  longer the dominant fraction of runtime.
- Both modes pass exact equality against the bf16 input tensor.
- Source placement, consumer placement, CB capacity, input reads, DRAM writes, and output order
  are identical between modes; only the RISC-V assignment of second-half reads changes.
- BRISC always writes the output. Split-on additionally uses it to gather half of each block while
  NCRISC gathers the other half.

## Reproduce

```bash
python -m ttnn.operations.examples.split_reader \
  --cores 8 --tiles-per-core 8 --block-tiles 8 \
  --transaction-bytes 32 64 128 256 512 1024 2048 --iters 20
```

Performance remains reported evidence, not a test assertion.
