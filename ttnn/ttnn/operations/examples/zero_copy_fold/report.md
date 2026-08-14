# Kernel fold vs reader/compute/writer separation — compute_only (fold) vs reader_compute_writer (3 kernels)

Concept (op-agnostic): the reader/writer run on their own RISCs (NCRISC/BRISC) and arm/drain the CBs
concurrently with compute on TRISC; folding them into the compute kernel serializes that onto the
compute thread. The payload below is an incidental same-spec zero-copy sharded tilize (CBs aliased onto
the resident L1 shards → no DRAM/NoC), chosen only to isolate pure program structure — any op with a
reader/compute/writer split shows the same effect.

box=bgd-lab-16-special-dstoiljkovic-for-reservation-52921  arch=wormhole_b0  placement=HEIGHT-sharded resident-L1 (no DRAM/NoC)  N=5 (median)  kernel-iters=100 (steady-state)
Same tilize, same aliased CBs — only the program structure differs. ns = median DEVICE KERNEL DURATION per launch. Ratio = reader_compute_writer / compute_only (>1 => fold is faster).

| H×W | cores | tiles/core | reader_compute_writer ns | compute_only ns (×) |
|---|---:|---:|---:|---:|
| 64×64 | 2 | 2 | 222 | 299 (0.74×) |
| 128×64 | 4 | 2 | 222 | 297 (0.75×) |
| 64×128 | 2 | 4 | 363 | 466 (0.78×) |
| 128×128 | 2 | 8 | 652 | 785 (0.83×) |
| 256×128 | 8 | 4 | 363 | 466 (0.78×) |
| 512×128 | 4 | 16 | 1241 | 1380 (0.90×) |
| 1024×256 | 4 | 64 | 4694 | 5014 (0.94×) |

`reader_compute_writer` runs three kernels per core (a dataflow reader that arms the resident input CB, the compute tilize, a dataflow writer that drains the output CB) plus the reader->compute->writer circular-buffer handshake. `compute_only` folds the arm+drain into the single compute kernel. Both alias the CBs onto the resident L1 shards, so there is no NoC to hide the extra kernels' fixed dispatch+handshake cost — it shows up as latency on small shards and amortizes as tiles/core grows.
