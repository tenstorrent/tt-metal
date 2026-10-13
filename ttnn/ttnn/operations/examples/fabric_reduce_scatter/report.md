# fabric_reduce_scatter — device report

Illustrative, not a CI bound. Every number is from one box and one fabric configuration; re-run the sweep on yours.

| stamp | value |
|---|---|
| box | `qb2-120-p02t03` — Blackhole QuietBox 2 (2× p300c, 4 chips); every hop has 2 Ethernet links |
| fabric | `FABRIC_2D_TORUS_XY`, router payload 14,336 B (7 bf16 tiles per packet) unless stated |
| date | 2026-10-13 |
| code | `316bf3de8c4` + uncommitted changes |
| metric | `DEVICE KERNEL DURATION [ns]` (in-process device profiler), slowest chip, median of 5 calls, not traced |
| correctness | every cell: each chip's output within 0.25 (max abs) of the fp32 sum of the group's bf16 partials; the worst cell was 0.0625 |
| data | all 132 cells: [`report_data.csv`](report_data.csv) (matched by the repo's `*.csv` gitignore rule; `git add -f` it) |

**What is measured.** A 4-chip reduce-scatter. Each chip holds a full-size partial result split into 4 blocks, and
chip p ends with block p summed over all chips. The busiest link direction carries 1.5 blocks on a balanced ring, 2
on an unbalanced ring and 3 on a line.

**% of peak** is the busiest link direction's rate against 48.5 GB/s, a bare one-hop stream on a QuietBox.

**Baseline.** Every section varies **one** setting from this configuration:

| setting | baseline value |
|---|---|
| topology | 4-chip ring (`4x1` mesh, `topology=Ring`), balanced (`balance=True`) |
| links per hop | 1 |
| workers per link and direction | 1 (no mux) |
| final cores per link | 2 |
| arrival increment (`inc_every`) | every 8 chunks |
| circular buffer (`cb_bytes`) | 112 KiB → 4 chunks per batch |
| router payload | 14,336 B |

**Size ladder.** Bf16 TILE input per chip; rows must split into 4 tile-aligned blocks:

| tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| shape | 128×32 | 128×128 | 256×256 | 512×512 | 1024×1024 | 2048×2048 | 2048×4096 |
| input | 8 KiB | 32 KiB | 128 KiB | 512 KiB | 2 MiB | 8 MiB | 16 MiB |

Reproduce the baseline over the ladder; override one variable to vary one setting:

```bash
RS_FABRICS=2d_torus_xy RS_TOPOS=4x1_ring RS_BALANCE=1 RS_LINKS=1 RS_WORKERS=1 RS_FINALS=2 RS_INC_EVERY=8 RS_CB_KB=112 \
RS_PAYLOAD=14336 RS_TRIALS=5 RS_BASELINE=0 \
RS_SHAPE="128,32;128,128;256,256;512,512;1024,1024;2048,2048;2048,4096" \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_reduce_scatter.py -s
```

## Baseline: a fixed cost, then a steady rate

| tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| time, µs | 12.6 | 14.6 | 18.4 | 26.7 | 40.6 | 130.9 | 243.1 |
| busiest link, % of peak | 1 | 2 | 5 | 15 | 40 | 50 | 53 |

- **The fixed cost is about 12–16 µs per call.** Over ≥ 256 tiles the time fits roughly t = 16.4 µs + input bytes /
  74 GB/s. The fit is loose (up to 12% off): the curve bends between 1,024 and 4,096 tiles.
- **The fixed cost and the transfer are equal at about 600 tiles per chip** (~1.2 MiB):
  - below that the call is latency-bound and the link nearly idle (1–15% at 4–256 tiles);
  - at 16 MiB it reaches 53% of the link.

**Shape does not matter, only size.** Tall (H × 32) and wide (128 × W) inputs against square at equal tile counts:

| tiles | tall | wide | square |
|---|---|---|---|
| 64 | 2048×32: 19.9 µs (1.08×) | 128×512: 18.7 µs (1.02×) | 18.4 µs |
| 1,024 | 32768×32: 40.1 µs (0.99×) | 128×8192: 40.7 µs (1.00×) | 40.6 µs |
| 8,192 | 262144×32: 241.6 µs (0.99×) | 128×65536: 241.5 µs (0.99×) | 243.1 µs |

## One setting at a time

Each cell is time against the reference at the same size: **above 1 is slower**. The reference is the baseline,
except where a row says "vs 2 links". There it is the baseline with 2 links (1 worker, 14,336 B), so the row isolates
its own setting.

| setting | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 | µs at 8,192 |
|---|---|---|---|---|---|---|---|---|
| line instead of ring | 1.31 | 1.34 | 1.26 | 1.20 | 1.41 | 1.35 | 1.38 | 335.2 |
| ring, balance off | 0.96 | 1.03 | 1.01 | 1.02 | 1.19 | 1.25 | 1.29 | 312.9 |
| 2 links | 1.02 | 0.92 | 0.85 | 0.75 | 0.77 | 0.57 | 0.56 | 136.9 |
| 2 workers | 1.27 | 1.05 | 0.99 | 0.99 | 0.99 | 0.79 | 0.77 | 186.3 |
| 4 workers | 1.48 | 1.18 | 1.03 | 0.97 | 1.07 | 0.78 | 0.75 | 183.3 |
| 2 workers, vs 2 links | 1.32 | 1.31 | 1.26 | 1.17 | 1.17 | 0.95 | 0.92 | 125.9 |
| 4 workers, vs 2 links | 1.75 | 1.48 | 1.27 | 1.32 | 1.17 | 1.17 | 1.10 | 151.2 |
| 1 final core per link | 1.03 | 1.08 | 1.13 | 1.10 | 1.19 | 1.14 | 1.16 | 282.6 |
| 4 final cores per link | 0.99 | 1.01 | 0.99 | 0.95 | 0.96 | 0.89 | 0.90 | 219.3 |
| `inc_every` 2 | 0.97 | 0.99 | 0.93 | 0.86 | 0.98 | 0.99 | 1.02 | — |
| `inc_every` 4 | 0.95 | 1.03 | 0.96 | 0.90 | 0.96 | 0.97 | 0.98 | — |
| `inc_every` 16 | 1.07 | 1.03 | 0.98 | 1.01 | 1.17 | 1.00 | 1.01 | — |
| `inc_every` 32 | 0.97 | 1.02 | 1.01 | 1.00 | 1.27 | 1.01 | 1.02 | — |
| `cb_bytes` 56 KiB (2 per batch) | 0.91 | 0.95 | 0.97 | 0.98 | 1.10 | 1.07 | 1.09 | — |
| `cb_bytes` 224 KiB (8 per batch) | 1.06 | 1.14 | 1.13 | 1.10 | 1.07 | 0.98 | 0.97 | — |
| payload 4,352 B | 1.06 | 1.07 | 1.02 | 1.00 | 1.42 | 1.39 | 1.43 | 347.2 |
| payload 4,352 B, vs 2 links | 1.07 | 1.10 | 1.11 | 1.16 | 1.27 | 1.33 | 1.32 | 180.7 |

The baseline at 8,192 tiles is 243.1 µs (1 link) and 136.9 µs (2 links).

### Ring vs line
- **A ring is 1.2–1.4× faster at every size.**
- **Small sizes:** the ring has fewer serial hops, so a lower fixed cost (fit t0 16.4 vs 20.1 µs).
- **Large sizes:** the ring's busiest link carries 1.5 blocks instead of 3.

### Balancing the ring
On an even ring the block opposite each chip is split in two halves of its rows, one summed each way round, so the
busiest link carries 1.5 blocks instead of 2.
- **It pays from ~1,024 tiles:** 1.19× at 1,024, 1.25–1.29× from 4,096.
- **Below that it is neutral** (0.96–1.03).

### Links
- **Two links help from 16 tiles up:** 0.75–0.92 at 16–1,024 tiles, 0.56 at ≥ 4,096.
- **At 4 tiles they are neutral** (1.02).

### Workers per link (through a fabric mux)
A relay port reads two DRAM streams per chunk (its own partial and the arrived sum), adds them and sends the result.
With W workers per link and direction, each runs that pipeline on 1/W of the link's DRAM banks. They share the link
through a fabric mux on the core next to the Ethernet core.
- **At 1 link, 2 or 4 workers are 1.27–1.33× faster from 4,096 tiles** (0.75–0.79).
- **At 1 link they cost at small sizes:** 2 workers are 1.27× slower at 4 tiles, 4 workers 1.48×. The mux adds a
  hop and a connection handshake to every call (fit t0 20–22 µs vs 16.4).
- **Between 64 and 1,024 tiles they are neutral.**
- **At 2 links, 2 workers gain only 5–8% from 4,096 tiles** and are 1.17–1.32× slower below. 4 workers are slower at
  every size.

### Final cores per link
Each final core reads three DRAM streams per chunk (its own partial and the arrivals from both sides), adds them and
writes its share of block p.
- **1 final core is 1.08–1.19× slower** from 16 tiles up.
- **4 are 1.04–1.12× faster** from 256 tiles up.
- **The default 2 sits between.**

### Arrival increments (`inc_every`)
The sender attaches an increment of the receiver's arrival counter to every N-th chunk, and to the last chunk of
each stream. A relay or final core reads a chunk only once its counter covers it.
- **Every 2–4 chunks is 4–14% faster at 64–256 tiles**, and within 5% elsewhere.
- **Every 16–32 chunks costs 17–27% at 1,024 tiles**, and is within 7% elsewhere.
- Every 8 (the default) is within 16% of the best setting at every size.

### Circular buffer depth (`cb_bytes`)
Each port and final core's reader issues one batch of DRAM reads before waiting, and the next stage consumes one
batch at a time. The buffer holds two batches.
- **2 chunks per batch (56 KiB) is 2–9% faster up to 256 tiles** and 7–10% slower from 1,024.
- **8 per batch (224 KiB) is 6–14% slower up to 1,024 tiles** and 2–3% faster from 4,096.
- 4 per batch (112 KiB, the default) is within 10% of the best setting at every size.

### Router payload (fabric-wide, not a per-op setting)
The router's maximum packet payload is set once when the fabric is initialized, for every op in the process. Each op
sends packets up to that size.
- **4,352 B is 1.39–1.43× slower from 1,024 tiles** (1 link), 1.27–1.33× at 2 links.
- **Up to 256 tiles it is within 7% at 1 link**, and 7–16% slower at 2 links.
- At 4,352 B the busiest link reaches 36–37% of peak at 16 MiB.
