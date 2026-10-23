# 8-chip rings on a Blackhole Galaxy, FABRIC_2D_TORUS_XY — send modes, device time

The [1D-ring 8-ring sweep](../8ring_galaxy_bh/README.md) repeated on a 2D torus fabric. Overview of all ring sizes and
fabrics: [`../README.md`](../README.md).

| | |
|---|---|
| Machine | `bh-glx-110-a09u02`, Blackhole Galaxy, 32 chips (the 1D results: `bh-glx-120-d06u14`) |
| Mesh / axis | `FAG_MESH=8,4 FAG_AXIS=0`: four 8-chip column rings (N/S), run at the same time; device time = slowest chip |
| Fabric | `FABRIC_2D_TORUS_XY`, `RELAXED_INIT`; 2 links per neighbour pair, wrap-around links included |
| Router payload | 14,336 B |
| Dtype / layout, method | as the 1D sweep: bf16 TILE, traced `[filler, all-gather] x 8`, slowest chip, median |
| Date | 2026-10-23, the commit that adds these results (2D routes along the connection's direction) |
| Correctness | all 220 cases (11 shapes × 2 link counts × 10 variants) bit-exact on all 32 chips; nothing refused |

Variants and busiest hops as in the [1D README](../8ring_galaxy_bh/README.md#variants) (`b` 3.5 shards, `m` / `mb`
4, `ms` / `mh` 3.5). On the 2D fabric every packet is routed by hop count along its connection's direction (N or S
here); a multicast runs the ring's spine with no E/W branches.

## Results (device µs; `*` = fastest at that size; in brackets the 1D-ring time)

**1 link**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 16.1 (14.3) | 14.0 (12.3) | 12.9 (11.5) | 13.3 (12.4) | 23.9 (20.3) | 15.8 (13.8) | **12.3*** (10.6) | 12.4 (10.1) | 14.5 (11.4) | 16.2 (12.8) |
| 16   | 19.6 (17.9) | 20.9 (16.5) | 18.7 (15.1) | 20.1 (16.3) | 28.0 (22.4) | 19.3 (17.4) | **16.8*** (11.9) | 17.4 (12.8) | 20.1 (14.9) | 18.5 (14.1) |
| 64   | 32.5 (30.6) | 48.6 (38.3) | 44.3 (35.3) | 42.0 (32.7) | 52.9 (40.2) | **31.8*** (30.8) | 43.4 (29.8) | 36.8 (33.4) | 38.3 (25.5) | 54.1 (19.8) |
| 256  | 85.2 (81.6) | 124.6 (102.7) | 111.5 (100.9) | 109.7 (88.3) | 120.1 (94.0) | **83.6*** (81.7) | 130.3 (94.1) | 101.8 (113.9) | 115.7 (83.7) | 123.4 (81.4) |
| 1024 | 319.4 (312.9) | 473.6 (365.1) | 390.6 (363.9) | 410.1 (319.7) | 439.6 (323.9) | **319.4*** (312.5) | 484.2 (386.1) | 391.9 (372.9) | 419.2 (314.0) | 460.9 (312.7) |
| 4096 | 1246.4 (1228.4) | 1853.8 (1408.7) | 1486.5 (1410.6) | 1630.2 (1239.2) | 1781.7 (1238.1) | **1245.3*** (1228.0) | 1879.4 (1537.0) | 1502.2 (1476.1) | 1629.2 (1231.9) | 1814.0 (1222.9) |
| 8192 | 2477.5 (2445.2) | 3672.9 (2795.1) | 2943.6 (2800.8) | 3254.1 (2460.9) | 3573.2 (2457.1) | **2477.0*** (2444.9) | 3689.2 (2935.3) | 2960.0 (3012.5) | 3269.2 (2454.3) | 3634.8 (2438.9) |

**2 links**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 13.0 (12.2) | 11.5 (10.6) | 11.2 (10.5) | 11.8 (11.2) | 20.5 (17.6) | 13.0 (12.1) | **8.7*** (9.3) | 9.5 (8.8) | 11.3 (10.4) | 16.1 (12.2) |
| 16   | 15.3 (14.5) | 15.1 (13.1) | 14.2 (12.4) | 15.1 (14.0) | 23.6 (20.0) | 15.2 (14.4) | **12.8*** (10.4) | 13.1 (11.4) | 15.5 (14.3) | 17.0 (13.1) |
| 64   | 24.5 (23.9) | 31.1 (24.2) | 27.7 (22.3) | 27.3 (22.1) | 36.2 (29.3) | **24.2*** (23.2) | 25.3 (16.9) | 24.7 (17.3) | 27.3 (18.6) | 33.0 (16.2) |
| 256  | 49.3 (51.5) | 75.2 (61.5) | 70.2 (57.7) | 72.3 (50.8) | 81.7 (59.3) | **45.4*** (43.5) | 80.2 (50.0) | 104.9 (55.1) | 71.5 (42.6) | 175.9 (26.7) |
| 1024 | 187.1 (196.0) | 271.7 (193.7) | 234.1 (190.2) | 245.7 (168.8) | 264.2 (174.2) | **163.5*** (158.4) | 260.0 (198.2) | 340.6 (191.0) | 250.6 (158.2) | 303.7 (158.2) |
| 4096 | 733.9 (765.4) | 1023.9 (714.3) | 890.8 (712.8) | 922.7 (635.0) | 956.7 (640.3) | **627.0*** (615.1) | 967.0 (762.6) | 897.1 (746.3) | 881.0 (620.6) | 972.1 (626.2) |
| 8192 | 1457.3 (1521.4) | 2031.0 (1415.0) | 1772.9 (1410.0) | 1817.8 (1253.7) | 1889.1 (1263.4) | **1244.0*** (1224.2) | 1930.8 (1523.8) | 1813.6 (1448.0) | 1691.1 (1237.4) | 1942.5 (1235.4) |

Square shards; the tall / wide ones match square within a few % (up to 11% at 64 tiles, `mh@auto` at 2 links; all
rows and the GB/s tables in [`fag_modes_8ring_2d_tables.txt`](fag_modes_8ring_2d_tables.txt)).

**Winners per link count**

| tiles/chip | 1 link | 2 links |
|---|---|---|
| ≤ 16 | `m@auto` | `m@auto` |
| 64 | `b@auto` | `b@auto` ≈ `m@auto` ≈ `mb@auto` |
| ≥ 256 | `b@auto` (16–18% ahead of the best multicast) | `b@auto` (26–35% ahead of the best multicast) |

## Reading

- **Relay runs as on the 1D ring.** `b@auto` is within 1–4% of its 1D time at every size ≥ 256 tiles and runs its
  busiest hop at 23.6–23.7 GB/s per link direction at size (2 links: 23.4–23.6). Uniform `b` at 2 links is even 4%
  *faster* than on 1D (1457 vs 1521 µs, 20.1 vs 19.3 GB/s).
- **Multicast over 3 or more chips loses a quarter of the link rate.** At 8,192 tiles, 1 link: `m` 3673 µs (+31% on
  1D), `ms` 3254 (+32%), `mh` 3573 (+45%): their busiest hop moves 16.4–18.3 GB/s per link direction, against ~24 on
  1D. At 2 links the gap widens (+44%, +45%, +50%). On the 1D ring the split multicasts `ms` / `mh` matched the relay
  at size; on 2D the relay wins at every size from 64 tiles (1 link) / 256 tiles (2 links).
- **`mb` is the exception**: 2944 µs at 1 link (+5% on 1D, 22.8 GB/s), the best multicast at size, though its legs
  are the same 4- and 3-chip multicasts as `m`'s. See the per-port profile below.
- **Small sizes**: whole-shard `m@auto` is fastest at ≤ 16 tiles, as on 1D, and at 2 links `m@auto` at 4 tiles is
  faster than on 1D (8.7 vs 9.3 µs). `mh@auto` no longer wins the middle: 54.1 µs at 64 tiles, 1 link, against 19.8
  on 1D. `mh@auto` at 256 tiles / 2 links (175.9 µs, 2.2× `mh`) and `mb@auto` at 1,024 tiles / 2 links (340.6, 1.5×
  `mb`, and barely faster than at 1 link) are outliers; not re-run.

## Per-port profile ([`fag_zones_8ring_2d_tables.txt`](fag_zones_8ring_2d_tables.txt))

When each multicast leg finished sending, 1 link, 4,096 tiles, over all 32 chips (µs), against the 1D ring's:

| | 4-chip leg: median / max | 3-chip leg: median / max | 4-chip leg: median wait for a fabric slot |
|---|---|---|---|
| `m`, 2D  | 1780 / 1830 | 1278 / 1815 | 1553 |
| `m`, 1D  | 1290 / 1403 | 1045 / 1373 | 1101 |
| `mb`, 2D | 1439 / 1482 | 736 / 1474 | 1217 |
| `mb`, 1D | 754 / 1403 | 1387 / 1409 | 564 |

- On 2D, as on 1D, most of every leg is spent waiting for a fabric slot: the links are full. What changes is how
  much they carry while full. `m`'s 4-chip legs all run toward p+1, so every link in that direction carries four
  multicasts in transit, each written and forwarded at every hop; it ends at 1830 µs (18 GB/s). In `mb` the long leg
  alternates direction, both directions carry a mix of 4- and 3-chip multicasts, and the slowest leg ends at 1482 µs.
- On 1D the slow leg in `mb` is the 3-chip one; on 2D it is the 4-chip one.
- Why a link filled with long multicasts runs slower on the 2D router than on the 1D one is not resolved here: the
  per-hop cost of write-and-forward in the 2D router has not been measured on its own. `ms` and `mh` are not
  profiled.

## Bandwidth (busiest hop, per link direction)

At ≥ 1,024 tiles: `b` / `b@auto` 23.0–23.7 GB/s at 1 link, `b@auto` 22.4–23.6 at 2 links (the same as on 1D); `mb` /
`mb@auto` 21.4–22.8 at 1 link; `m`, `ms`, `mh` and their `@auto` forms 15.9–18.3 at 1 link and 11.5–17.4 at 2 links.

## Files

| file | what |
|---|---|
| `fag_modes_8ring_2d.log` | `FAG_MODES` lines, all ten variants, one session |
| `fag_modes_8ring_2d_tables.txt` | `fag_modes_analyze.py fag_modes_8ring_2d.log` |
| `fag_zones_8ring_2d_tables.txt` | `fag_zones_analyze.py` per-port tables, `b,m,mb`, 16–4,096 tiles, 1 and 2 links (the profiler CSV, 2.4 GB, is not kept) |

## Reproduce

```bash
FAG_MESH=8,4 FAG_AXIS=0 MODES_FABRICS=2d_torus_xy \
MODES_VARIANTS=b,m,mb,ms,mh,b@auto,m@auto,mb@auto,ms@auto,mh@auto \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s
FAG_MESH=8,4 FAG_AXIS=0 ZONES_FABRIC=2d_torus_xy scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_zones.py -s
```

Taken as one tt-sbatch job on the same Galaxy as the 4-ring results (plain `pytest --timeout=0`: 10 min for the
220 cases, 3 min for the zones). Under `python -m tracy -r` the zones run exits 1 after the test passes: tracy's ops
report fails to parse the example's inline kernel source in the device CSV; `fag_zones_analyze.py` reads that CSV
directly and is not affected.
