# 8-chip rings on a Blackhole Galaxy — send modes, device time

Answers [`../../HANDOFF_8RING.md`](../../HANDOFF_8RING.md). Overview of all ring sizes: [`../README.md`](../README.md).

| | |
|---|---|
| Machine | `bh-glx-120-d06u14`, Blackhole Galaxy, 32 chips |
| Mesh / axis | `FAG_MESH=8,4 FAG_AXIS=0`: four 8-chip column rings, run at the same time; device time = slowest chip of the mesh |
| Fabric | `FABRIC_1D_RING`, `RELAXED_INIT`; 2 links per neighbour pair, wrap-around links included |
| Router payload | 14,336 B (`FAG_PAYLOAD` default) |
| Dtype / layout | bf16, TILE, DRAM interleaved; per-chip shard = tiles × 2 KiB |
| Method | the handoff's: `[filler, all-gather] x 8` traced, replayed 5×, slowest chip's program time per call, median |
| Date / code | 2026-10-22, `cf4a7a0b625` + the `mcast_split` / `mcast_halves` / sink-core change committed with these results |
| Correctness | every case bit-exact on all 32 chips; nothing refused. `ms@auto`, `mh`, `mh@auto` also passed 6 alternating-input calls into one reused output, each checked |

## Variants

Busiest hop = shards on the most loaded link direction, per link.

| variant | what each chip sends | busiest hop |
|---|---|---|
| `b` | relay (one hop at a time), far shard split between the directions | 3.5 |
| `m` | own shard: one 4-chip multicast toward p+1, one 3-chip toward p−1 | 4 |
| `mb` | as `m`, the 4-chip leg alternating direction by chip parity | 4 |
| `ms` | `mcast_split`: half A over 4 chips + half B over 3 toward p+1; B over 4 + A over 3 toward p−1 | 3.5 |
| `mh` | `mcast_halves`: half A over all 7 other chips toward p+1, half B over all 7 toward p−1 | 3.5 |
| `…@auto` | per-chip port placement next to each link's Ethernet core (multicast variants keep their arrival counters on sink cores); without it, chip 0's placement on every chip (`uniform`) | — |

## Results (device µs; `*` = fastest at that size)

Square shards; the tall / wide ones (2048×32, 32×2048, 32768×32, 32×32768) match square within a few % (full rows in
[`fag_modes_8ring_tables.txt`](fag_modes_8ring_tables.txt)).

**1 link**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 14.3 | 12.3 | 11.5 | 12.4 | 20.3 | 13.8 | 10.6 | **10.1*** | 11.4 | 12.8 |
| 16   | 17.9 | 16.5 | 15.1 | 16.3 | 22.4 | 17.4 | **11.9*** | 12.8 | 14.9 | 14.1 |
| 64   | 30.6 | 38.3 | 35.3 | 32.7 | 40.2 | 30.8 | 29.8 | 33.4 | 25.5 | **19.8*** |
| 256  | 81.6 | 102.7 | 100.9 | 88.3 | 94.0 | 81.7 | 94.1 | 113.9 | 83.7 | **81.4*** |
| 1024 | 312.9 | 365.1 | 363.9 | 319.7 | 323.9 | **312.5*** | 386.1 | 372.9 | 314.0 | 312.7 |
| 4096 | 1228.4 | 1408.7 | 1410.6 | 1239.2 | 1238.1 | 1228.0 | 1537.0 | 1476.1 | 1231.9 | **1222.9*** |
| 8192 | 2445.2 | 2795.1 | 2800.8 | 2460.9 | 2457.1 | 2444.9 | 2935.3 | 3012.5 | 2454.3 | **2438.9*** |

**2 links**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 12.2 | 10.6 | 10.5 | 11.2 | 17.6 | 12.1 | 9.3 | **8.8*** | 10.4 | 12.2 |
| 16   | 14.5 | 13.1 | 12.4 | 14.0 | 20.0 | 14.4 | **10.4*** | 11.4 | 14.3 | 13.1 |
| 64   | 23.9 | 24.2 | 22.3 | 22.1 | 29.3 | 23.2 | 16.9 | 17.3 | 18.6 | **16.2*** |
| 256  | 51.5 | 61.5 | 57.7 | 50.8 | 59.3 | 43.5 | 50.0 | 55.1 | 42.6 | **26.7*** |
| 1024 | 196.0 | 193.7 | 190.2 | 168.8 | 174.2 | 158.4 | 198.2 | 191.0 | **158.2*** | 158.2 |
| 4096 | 765.4 | 714.3 | 712.8 | 635.0 | 640.3 | **615.1*** | 762.6 | 746.3 | 620.6 | 626.2 |
| 8192 | 1521.4 | 1415.0 | 1410.0 | 1253.7 | 1263.4 | **1224.2*** | 1523.8 | 1448.0 | 1237.4 | 1235.4 |

**Winners per link count**

| tiles/chip | 1 link | 2 links |
|---|---|---|
| ≤ 16 | `m@auto` / `mb@auto` | `m@auto` / `mb@auto` |
| 64 | `mh@auto` (36% faster than `b@auto`) | `mh@auto` ≈ `m@auto` |
| 256 | `mh@auto` ≈ `b@auto` | `mh@auto` (26.7 vs 43.5 µs for `b@auto`) |
| ≥ 1024 | `b@auto` ≈ `mh@auto` ≈ `ms@auto` (within 1%) | `b@auto` ≈ `ms@auto` ≈ `mh@auto` (within 2%) |

## The handoff's question: does alternating (`mb`) still win on 8 chips?

**No.** On the 4-chip QB2 ring `mb` beat `m` by 10–20%. Here they are within a few % at small sizes and equal from
1,024 tiles, and both lose to the relay `b` from 64 tiles up at 1 link (1409 vs 1228 µs at 4,096 tiles).

**Mechanism (per-port profile, [`fag_zones_8ring_tables.txt`](fag_zones_8ring_tables.txt)).** When each multicast
leg finished sending, 1 link, 4,096 tiles, over all 32 chips (µs):

| | 4-chip leg: median / max | 3-chip leg: median / max |
|---|---|---|
| `m`  | 1290 / 1403 | 1045 / 1373 |
| `mb` | 754 / 1403 | 1387 / 1409 |

- Both legs hit the same ~1,400 µs ceiling on some chip, and most of each leg is spent waiting for a fabric slot
  (median slot wait 1,101 µs on `m`'s 4-chip leg): the links are full. Which leg is slow follows the link it rides,
  not its hop count; in `mb` the 3-chip leg is usually the slow one.
- Time follows the busiest hop's bytes: 1228 / 1409 = 0.872 against 3.5 / 4 = 0.875. Relay and multicast move
  ~24 GB/s per link alike; the relay simply puts 3.5 shards on its busiest hop where a whole-shard multicast must put 4.
- So splitting the hop *range* (4 = 2 + 2, 3 = 1 + 2) cannot help: the same shard still crosses the same links.
  Splitting the *shard* does: `ms` and `mh` bring the busiest hop to 3.5 with no relays, and `ms` matches `b` within 1%
  at size.

**Placement.** Multicast needed `uniform` placement in the handoff (one arrival-counter address per multicast packet).
With sink cores it now runs `@auto` too. That is worth 20–30% to multicast up to 64 tiles. At 1 link from 256 tiles,
`m@auto` / `mb@auto` are 5–9% *slower* than uniform (1537 vs 1409 µs at 4,096); not profiled, cause unknown.
`ms@auto` and `mh@auto` do not show it.

## Bandwidth (busiest hop, per link direction)

At ≥ 1,024 tiles every variant without a placement handicap runs at 23–24 GB/s per link direction, at 1 and 2 links
(2 links scale ~2×: per-chip receive 48 GB/s at 1 link, ~96 GB/s at 2). Uniform `b` at 2 links reaches only 19.3 GB/s
(its port cores far from their Ethernet cores). `mh@auto` at 256 tiles / 2 links computes to 34.4 GB/s, above that
ceiling, and was re-checked bit-exact over alternating calls; it is not profiled. The QB2 4-ring of the handoff ran
`mb` at ~44 GB/s per link (756 µs, 2 shards of 16 MiB): this Galaxy's links, in this setup, run at about half that.

## Files

| file | what |
|---|---|
| `fag_modes_8ring_sweep1.log` | `FAG_MODES` lines: `b,m,mb,b@auto` (the handoff's sweep) |
| `fag_modes_8ring_sweep2.log` | `b,m,mb,ms` and their `@auto` forms, same session |
| `fag_modes_8ring_sweep3.log` | `mh,mh@auto` plus a `b@auto` rerun (within 2.9% of sweep 2) |
| `fag_modes_8ring_tables.txt` | `fag_modes_analyze.py fag_modes_8ring_sweep3.log fag_modes_8ring_sweep2.log` (sweep 2 numbers, `mh` from sweep 3) |
| `fag_zones_8ring_tables.txt` | `fag_zones_analyze.py` per-port tables, `b,m,mb`, 16–4,096 tiles, 1 and 2 links (the profiler CSV, 5.3 GB, is not kept) |

Not run: `base`, `d`, `db` (optional in the handoff; on the 4-ring `d` / `db` never won), and per-port profiles of
`ms`, `mh` and the `@auto` multicast variants.

## Reproduce

```bash
FAG_MESH=8,4 FAG_AXIS=0 MODES_FABRICS=1d_ring \
MODES_VARIANTS=b,m,mb,ms,mh,b@auto,m@auto,mb@auto,ms@auto,mh@auto \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s
FAG_MESH=8,4 FAG_AXIS=0 scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_zones.py -s
```

The full ladder with ten variants runs longer than 10 minutes; these results were taken in three chunks of
`MODES_SHAPES` (small shapes, 1,024-tile shapes, 2048² + 2048×4096). On this node the worktree's `python_env` had no
interpreter, so the runs used `/opt/venv/bin/python` with the main checkout's `python_env` site-packages on
`PYTHONPATH`, and `build/tools` was linked to `build_Release/tools` for tracy.
