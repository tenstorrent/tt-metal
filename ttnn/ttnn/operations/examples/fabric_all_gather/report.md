# fabric_all_gather — device report

Illustrative, not a CI bound. Every number is from one box and one fabric configuration; re-run the sweep on yours.

| stamp | value |
|---|---|
| box | `qb2-120-p02t03` — Blackhole QuietBox 2 (2× p300c, 4 chips); every hop has 2 Ethernet links |
| fabric | `FABRIC_2D_TORUS_XY`, router payload 14,336 B (7 bf16 tiles per packet) unless stated |
| date | 2026-10-13 |
| code | `316bf3de8c4` + uncommitted changes |
| metric | `DEVICE KERNEL DURATION [ns]` (in-process device profiler), slowest chip, median of 5 calls, not traced |
| correctness | every cell bit-exact: each chip's output == the group's shards concatenated in chip order |
| data | all 90 cells: [`report_data.csv`](report_data.csv) (matched by the repo's `*.csv` gitignore rule; `git add -f` it) |

**What is measured.** A 4-chip all-gather: each chip contributes one shard and ends with all four. The busiest link
direction carries 1.5 shards on a balanced ring, 2 on an unbalanced ring and 3 on a line. Rates below:
- **receive GB/s per chip** = shard bytes × (G − 1) / time — what one chip takes in;
- **% of peak** = the busiest link direction's rate against 48.5 GB/s, a bare one-hop stream on a QuietBox.

**Baseline.** Every section varies **one** setting from this configuration:

| setting | baseline value |
|---|---|
| topology | 4-chip ring (`4x1` mesh, `topology=Ring`), balanced (`balance=True`) |
| links per hop | 1 |
| arrival increment (`inc_every`) | every 8 chunks |
| circular buffer (`cb_bytes`) | 112 KiB → 4 chunks per batch, 8 in the buffer |
| router payload | 14,336 B |

**Size ladder.** Bf16 TILE shards per chip, square-ish:

| tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| shape | 64×64 | 128×128 | 256×256 | 512×512 | 1024×1024 | 2048×2048 | 2048×4096 |
| shard | 8 KiB | 32 KiB | 128 KiB | 512 KiB | 2 MiB | 8 MiB | 16 MiB |

Reproduce one row (the baseline over the ladder; override one variable to vary one setting):

```bash
AG_FABRICS=2d_torus_xy AG_TOPOS=4x1_ring AG_VARIANTS=b AG_LINKS=1 AG_INC_EVERY=8 AG_CB_KB=112 AG_PAYLOAD=14336 \
AG_TRIALS=5 AG_SHAPE="64,64;128,128;256,256;512,512;1024,1024;2048,2048;2048,4096" \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s \
  -k "not reuse and not subdevice"
```

## Baseline: a fixed cost, then a steady rate

| tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| time, µs | 10.6 | 13.7 | 16.8 | 29.1 | 89.2 | 335.0 | 660.4 |
| receive GB/s per chip | 2.3 | 7.2 | 23.4 | 54.0 | 70.6 | 75.1 | 76.2 |
| busiest link, % of peak | 2 | 7 | 24 | 56 | 73 | 77 | 79 |

Over ≥ 256 tiles the time fits **t = 8.4 µs + bytes received / 77.2 GB/s** (largest error 1.2%). The 8.4 µs is a
fixed cost per call, and the two terms are equal at about 100 tiles per chip (~200 KiB):
- **below ~100 tiles** the call is latency-bound: the fixed cost dominates and the link is mostly idle (2–24% at
  4–64 tiles);
- **above ~1,000 tiles** it is bandwidth-bound, at 73–79% of the link.

**Shape does not matter, only size.** Tall (H × 32) and wide (32 × W) shards against square at equal tile counts:

| tiles | tall | wide |
|---|---|---|
| 64 | 2048×32: 17.8 µs (1.06×) | 32×2048: 17.0 µs (1.01×) |
| 1,024 | 32768×32: 89.2 µs (1.00×) | 32×32768: 89.7 µs (1.01×) |
| 8,192 | 262144×32: 663.8 µs (1.005×) | 32×262144: 663.3 µs (1.004×) |

Pages are interleaved round-robin over the DRAM banks whatever the shape, so the op sees the same chunks either way.
The 6% at 64 tiles (tall) is about 1 µs.

## One setting at a time

Each cell is time against the baseline at the same size: **above 1 is slower**.

| setting | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 | fit: t0 µs, BW GB/s/chip |
|---|---|---|---|---|---|---|---|---|
| line instead of ring | 1.29 | 1.25 | 1.39 | 1.55 | 1.78 | 1.83 | 1.82 | 9.8, 42.1 |
| ring, balance off | 1.02 | 0.93 | 1.10 | 1.20 | 1.26 | 1.28 | 1.27 | 9.4, 60.4 |
| 2 links | 0.92 | 0.80 | 0.80 | 0.63 | 0.58 | 0.53 | 0.53 | 8.2, 147.2 |
| `inc_every` 2 | 1.05 | 1.09 | 1.08 | 1.15 | 1.17 | 1.14 | 1.14 | 10.6, 67.9 |
| `inc_every` 4 | 0.95 | 1.05 | 1.00 | 1.03 | 1.02 | 1.01 | 1.01 | 8.8, 76.4 |
| `inc_every` 16 | 0.96 | 1.07 | 1.11 | 1.02 | 0.99 | 0.99 | 0.99 | 8.7, 78.1 |
| `inc_every` 32 | 1.02 | 1.07 | 1.11 | 1.02 | 0.99 | 0.98 | 0.98 | 8.7, 78.8 |
| `cb_bytes` 56 KiB (2 per batch) | 0.95 | 0.96 | 1.01 | 1.11 | 1.16 | 1.16 | 1.18 | 6.4, 65.1 |
| `cb_bytes` 224 KiB (8 per batch) | 1.12 | 1.12 | 1.19 | 1.06 | 1.04 | 1.02 | 1.02 | 9.6, 75.7 |
| payload 4,352 B | 1.15 | 1.11 | 1.27 | 1.77 | 1.99 | 2.04 | 2.06 | 8.9, 37.3 |
| 2 links + payload 4,352 B | 1.05 | 0.93 | 1.03 | 1.00 | 1.03 | 1.03 | 1.04 | 7.3, 74.0 |

The baseline fit for comparison is 8.4 µs and 77.2 GB/s per chip.

### Ring vs line
- **A ring wins at every size**: 1.25–1.39× at ≤ 64 tiles, 1.8× from 1,024 tiles up.
- **At small sizes the gain is latency:** a ring finishes in G/2 = 2 hops per direction, a line needs G − 1 = 3 in
  one direction, so the line's fixed cost is higher (9.8 µs vs 8.4).
- **At large sizes it is bandwidth:** the busiest link direction carries 1.5 shards on a balanced ring against 3 on
  a line.

### Balancing the ring
- **Balancing pays from ~64 tiles up**, 1.10× at 64 and 1.26–1.28× from 1,024. On an even ring it splits the
  shard opposite each receiver between the two directions: 1.5 shards on the busiest link direction instead of 2.
- **Below that it does nothing or slightly costs** (0.93–1.02 at 4–16 tiles): the split halves are a few chunks
  each, and the extra entry adds a step.
- It is **off by default** in this op. Turn it on for rings above ~64 tiles per chip.

### Links
- **Two links are 1.9× at large sizes** (0.53) and still help when tiny: 0.92 at 4 tiles, 0.80 at 16–64. Each link
  carries half the banks.
- The busiest link runs at 74% at 2 links against 79% at 1, so the second link is not quite free.

### Arrival increments (`inc_every`)
The sender attaches an increment of the receiver's arrival counter to every N-th chunk (and to each shard's last).
The receiving router stalls until each such write has landed.
- **Every 2 chunks is 8–17% slower at every size from 16 tiles.** The stalls dominate.
- **8 is the safe default.** 16 and 32 are 1–2% faster from 1,024 tiles up, but 7–11% slower at 16–64 tiles: a
  short shard has few chunks, so a coarse interval leaves its relay waiting for the end-of-shard increment.

### Circular buffer depth (`cb_bytes`)
The port's reader issues one batch of DRAM reads before waiting; the sender sends one batch before flushing. The
buffer holds two batches, so one is read while the other is sent.
- **2 chunks per batch (56 KiB) is 11–18% slower from 256 tiles up**, but 4–5% faster at ≤ 16 tiles, where a
  smaller batch fills sooner.
- **8 per batch (224 KiB) is 12–19% slower at ≤ 64 tiles** and 2–6% slower above. A short shard waits for a batch it
  cannot fill.
- **4 per batch (112 KiB, the default) is best or within 5% everywhere.**

### Router payload (fabric-wide, not a per-op setting)
The router's maximum packet payload is set once when the fabric is initialized, for every op in the process. Each op
sends packets up to that size.
- **The default 4,352 B is 2.0× slower than 14,336 B from 1,024 tiles up**, and 11–27% slower even at 4–64 tiles. At
  4,352 B the busiest link reaches only 38% of peak. The timings fit a fixed cost of about 235 ns per packet, which
  caps a link near 18.5 GB/s whatever the op or its worker count (inferred from the timings; the router was not
  profiled).
- **Two links at 4,352 B only match one link at 14,336 B** (1.00–1.05).

## Dual cycles on a 2D torus

`scheme="dual_cycles"` gathers over the whole torus on two edge-disjoint Hamiltonian cycles, each carrying half of
every shard (half the DRAM banks). Every chip then uses all four of its neighbours. Running it needs a torus with both
sides ≥ 3, i.e. a Galaxy; the numbers in this section are computed from `plan()` (`link_load`) at this code, not
timed on the QuietBox above.

**Busiest hop.** Shards carried by the busiest directed hop, per link; the call takes at least busiest × shard bytes
/ (links × link rate).

| torus | topology | chips per group | busiest hop | busiest hop, `balance=True` | links used per chip |
|---|---|---|---|---|---|
| 4 × 4 | ring along each column (`cluster_axis=0`) | 4 | 2 | 1.5 | 2 |
| 4 × 4 | ring along each row (`cluster_axis=1`) | 4 | 2 | 1.5 | 2 |
| 4 × 4 | snake ring over all chips | 16 | 8 | 7.5 | 2 |
| 4 × 4 | **dual cycles over all chips** | **16** | **4** | **3.75** | **4** |
| 4 × 8 | ring along each column (`cluster_axis=0`) | 4 | 2 | 1.5 | 2 |
| 4 × 8 | ring along each row (`cluster_axis=1`) | 8 | 4 | 3.5 | 2 |
| 4 × 8 | snake ring over all chips | 32 | 16 | 15.5 | 2 |
| 4 × 8 | **dual cycles over all chips** | **32** | **8** | **7.75** | **4** |

For a gather over the whole mesh, dual cycles halve the busiest hop of the snake: a 2× lower bandwidth bound. The
per-axis rings gather over smaller groups, so they are not the same collective.

**Reproduce on a Galaxy** (`4x4_*` on a 4 × 4 sub-torus):

```bash
AG_FABRICS=2d_torus_xy AG_TOPOS=4x8_snake_ring,4x8_dual_cycles AG_VARIANTS=base,b AG_LINKS=1 AG_TRIALS=5 \
AG_SHAPE="64,64;128,128;256,256;512,512;1024,1024;2048,2048;2048,4096" \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s \
  -k "not reuse and not subdevice"
```

### The stored cycles

`_HAMILTONIAN_CYCLES` in `program_descriptor_with_inline_kernels.py`, which `plan()` uses for these two shapes
(any other shape is searched). Each cycle lists `(row, col)` mesh coordinates in ring order; consecutive entries are
torus neighbours and the last closes back to the first. The two cycles of a shape share no link and together use
every link of the torus.

```python
_HAMILTONIAN_CYCLES = {
    (4, 4): (
        [
            (0, 0), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 0),
            (2, 0), (2, 3), (3, 3), (3, 2), (2, 2), (2, 1), (3, 1), (3, 0),
        ],
        [
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (3, 3), (0, 3),
            (1, 3), (2, 3), (2, 2), (1, 2), (0, 2), (3, 2), (3, 1), (0, 1),
        ],
    ),
    (4, 8): (
        [
            (0, 0), (3, 0), (2, 0), (1, 0), (1, 1), (0, 1), (3, 1), (2, 1),
            (2, 2), (1, 2), (0, 2), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3),
            (0, 4), (3, 4), (2, 4), (1, 4), (1, 5), (0, 5), (3, 5), (2, 5),
            (2, 6), (2, 7), (3, 7), (3, 6), (0, 6), (1, 6), (1, 7), (0, 7),
        ],
        [
            (0, 0), (1, 0), (1, 7), (2, 7), (2, 0), (2, 1), (1, 1), (1, 2),
            (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (0, 7), (3, 7), (3, 0),
            (3, 1), (3, 2), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6),
            (2, 6), (3, 6), (3, 5), (3, 4), (3, 3), (0, 3), (0, 2), (0, 1),
        ],
    ),
}
```

Each chip's position on each cycle (position p sends to p + 1 and p − 1, wrapping):

**4 × 4, cycle A**

| row \ col | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0 | 3 | 2 | 1 |
| **1** | 7 | 4 | 5 | 6 |
| **2** | 8 | 13 | 12 | 9 |
| **3** | 15 | 14 | 11 | 10 |

**4 × 4, cycle B**

| row \ col | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0 | 15 | 12 | 7 |
| **1** | 1 | 2 | 11 | 8 |
| **2** | 4 | 3 | 10 | 9 |
| **3** | 5 | 14 | 13 | 6 |

**4 × 8, cycle A**

| row \ col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **0** | 0 | 5 | 10 | 15 | 16 | 21 | 28 | 31 |
| **1** | 3 | 4 | 9 | 14 | 19 | 20 | 29 | 30 |
| **2** | 2 | 7 | 8 | 13 | 18 | 23 | 24 | 25 |
| **3** | 1 | 6 | 11 | 12 | 17 | 22 | 27 | 26 |

**4 × 8, cycle B**

| row \ col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **0** | 0 | 31 | 30 | 29 | 10 | 11 | 12 | 13 |
| **1** | 1 | 6 | 7 | 8 | 9 | 22 | 23 | 2 |
| **2** | 4 | 5 | 18 | 19 | 20 | 21 | 24 | 3 |
| **3** | 15 | 16 | 17 | 28 | 27 | 26 | 25 | 14 |
