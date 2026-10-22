# 4-chip rings on a Blackhole Galaxy — send modes, device time

The [8-ring sweep](../8ring_galaxy_bh/README.md) repeated on the Galaxy's 4-chip rows. Overview of all ring sizes:
[`../README.md`](../README.md).

| | |
|---|---|
| Machine | `bh-glx-120-d06u14`, Blackhole Galaxy, 32 chips |
| Mesh / axis | `FAG_MESH=8,4 FAG_AXIS=1`: eight 4-chip row rings, run at the same time; device time = slowest chip of the mesh |
| Fabric | `FABRIC_1D_RING`, `RELAXED_INIT`; 2 links per neighbour pair |
| Router payload | 14,336 B |
| Dtype / layout, method | as the 8-ring: bf16 TILE, traced `[filler, all-gather] x 8`, slowest chip, median |
| Date | 2026-10-22, same code as the 8-ring results |
| Correctness | all 220 cases (11 shapes × 2 link counts × 10 variants) bit-exact on all 32 chips; nothing refused |

On 4 chips the variants are (G = 4): `b` relay with the far shard split (busiest hop 1.5 shards); `m` one 2-chip
multicast + one unicast (2); `mb` the same, alternating by parity (2); `ms` half A over 2 chips + half B unicast toward
p+1, the reverse toward p−1 (1.5); `mh` half A over all 3 other chips each way (1.5). `@auto` as in the 8-ring README.

## Results (device µs; `*` = fastest at that size)

**1 link**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 8.4 | 7.2 | 6.9 | 7.3 | 9.5 | 8.4 | **5.7*** | 5.8 | 6.4 | 6.5 |
| 16   | 10.2 | 9.0 | 8.5 | 8.5 | 9.9 | 9.6 | 7.0 | 7.0 | 7.7 | **6.9*** |
| 64   | 16.1 | 19.4 | 18.0 | 16.0 | 17.1 | 15.7 | 15.9 | 16.4 | 12.6 | **8.0*** |
| 256  | 38.1 | 50.3 | 48.3 | 39.0 | 40.5 | 37.9 | 46.7 | 46.2 | 35.6 | **29.4*** |
| 1024 | 136.8 | 180.8 | 180.1 | 137.9 | 139.3 | 136.6 | 180.4 | 178.2 | 134.2 | **127.3*** |
| 4096 | 528.6 | 701.4 | 703.4 | 530.0 | 531.6 | 528.4 | 813.5 | 701.9 | 526.5 | **520.4*** |
| 8192 | 1050.9 | 1392.7 | 1398.8 | 1052.3 | 1053.6 | 1050.5 | 1592.9 | 1397.2 | 1048.5 | **1041.6*** |

**2 links**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 7.6 | 6.6 | 6.5 | 6.7 | 8.5 | 7.3 | 5.7 | **5.4*** | 6.1 | 6.2 |
| 16   | 8.7 | 7.7 | 7.4 | 7.8 | 9.1 | 8.3 | **5.9*** | 6.2 | 7.2 | 6.5 |
| 64   | 12.6 | 13.7 | 12.6 | 11.4 | 12.7 | 12.2 | 8.8 | 8.9 | 8.9 | **7.0*** |
| 256  | 27.0 | 31.3 | 29.0 | 26.1 | 26.7 | 22.2 | 23.5 | 23.4 | 18.8 | **13.1*** |
| 1024 | 97.5 | 102.1 | 94.4 | 87.2 | 85.8 | 71.4 | 90.8 | 89.5 | 68.2 | **65.6*** |
| 4096 | 379.9 | 384.3 | 359.2 | 334.6 | 306.5 | 266.4 | 362.3 | 350.8 | 265.2 | **257.2*** |
| 8192 | 758.2 | 761.1 | 718.5 | 665.3 | 601.1 | 527.9 | 717.8 | 698.5 | 527.6 | **517.0*** |

Tall / wide shards are within 4% of square at equal tiles, except `mb@auto` at 2 links (one shape 17% faster);
all rows and the GB/s tables are in [`fag_modes_4ring_tables.txt`](fag_modes_4ring_tables.txt).

## Reading

- **`mh@auto` is fastest from 16 tiles (1 link) / 64 tiles (2 links) to the largest size**: about 2× `b@auto` at
  64 tiles (8.0 vs 15.7 µs, 1 link), 13.1 vs 22.2 µs at 256 tiles / 2 links, and 1–5% ahead at size. Whole-shard
  multicast (`m@auto`, `mb@auto`) wins only at 4 tiles (and 16 tiles at 2 links).
- **Busiest-hop bytes decide the large sizes**: `m` / `mb` (2 shards) are ~33% slower than `b` (1.5) at 1 link, and
  every 1.5-shard variant without a placement handicap runs at 23.8–24.3 GB/s per link direction.
- **Placement**: uniform `b` at 2 links reaches only 16.6 GB/s (`b@auto` 23.8); `m@auto` at 1 link is again slower
  than uniform `m` at size (1593 vs 1393 µs at 8,192 tiles).
- **Against the 8-ring**: the best 8-ring time at 8,192 tiles / 1 link is 2.34× the 4-ring's (2438.9 vs 1041.6 µs)
  against a busiest-hop ratio of 3.5 / 1.5 = 2.33.
- **Against the QB2 4-ring of the handoff** (same ring size): `b` 1051 vs 866 µs and `mb` 1399 vs 756 µs at 8,192
  tiles / 1 link. The ~24 GB/s link ceiling is this box's, not the ring length's.

## Files

| file | what |
|---|---|
| `fag_modes_4ring.log` | `FAG_MODES` lines, all ten variants |
| `fag_modes_4ring_tables.txt` | `fag_modes_analyze.py fag_modes_4ring.log` |

## Reproduce

```bash
FAG_MESH=8,4 FAG_AXIS=1 MODES_FABRICS=1d_ring \
MODES_VARIANTS=b,m,mb,ms,mh,b@auto,m@auto,mb@auto,ms@auto,mh@auto \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s
```

(Taken in three `MODES_SHAPES` chunks, as the 8-ring.)
