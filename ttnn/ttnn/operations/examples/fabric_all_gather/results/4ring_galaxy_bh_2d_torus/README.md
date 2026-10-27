# 4-chip rings on a Blackhole Galaxy, FABRIC_2D_TORUS_XY — send modes, device time

The [1D-ring 4-ring sweep](../4ring_galaxy_bh/README.md) repeated on a 2D torus fabric. Overview of all ring sizes and
fabrics: [`../README.md`](../README.md).

| | |
|---|---|
| Machine | `bh-glx-110-a09u02`, Blackhole Galaxy, 32 chips (the 1D results: `bh-glx-120-d06u14`) |
| Mesh / axis | `FAG_MESH=8,4 FAG_AXIS=1`: eight 4-chip row rings (E/W), run at the same time; device time = slowest chip |
| Fabric | `FABRIC_2D_TORUS_XY`, `RELAXED_INIT`; 2 links per neighbour pair, wrap-around links included |
| Router payload | 14,336 B |
| Dtype / layout, method | as the 1D sweep: bf16 TILE, traced `[filler, all-gather] x 8`, slowest chip, median |
| Date | 2026-10-23, the commit that adds these results (2D routes along the connection's direction) |
| Correctness | all 220 cases (11 shapes × 2 link counts × 10 variants) bit-exact on all 32 chips; nothing refused |

Variants as in the 1D README (G = 4): `b` relay with the far shard split (busiest hop 1.5 shards); `m` one 2-chip
multicast + one unicast (2); `mb` the same, alternating by parity (2); `ms` half A over 2 chips + half B unicast toward
p+1, the reverse toward p−1 (1.5); `mh` half A over all 3 other chips each way (1.5). On the 2D fabric every packet is
routed by hop count along its connection's direction (E or W here), multicast and unicast alike.

## Results (device µs; `*` = fastest at that size; in brackets the 1D-ring time)

**1 link**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 9.0 (8.4) | 7.6 (7.2) | 7.8 (6.9) | 7.9 (7.3) | 10.4 (9.5) | 8.8 (8.4) | 6.6 (5.7) | **6.2*** (5.8) | 6.6 (6.4) | 7.0 (6.5) |
| 16   | 10.7 (10.2) | 10.1 (9.0) | 9.3 (8.5) | 9.1 (8.5) | 10.9 (9.9) | 10.7 (9.6) | 8.3 (7.0) | 7.6 (7.0) | 8.4 (7.7) | **7.4*** (6.9) |
| 64   | 16.5 (16.1) | 20.9 (19.4) | 18.3 (18.0) | 16.6 (16.0) | 19.9 (17.1) | 16.6 (15.7) | 19.0 (15.9) | 16.1 (16.4) | 15.7 (12.6) | **11.7*** (8.0) |
| 256  | 38.8 (38.1) | 51.3 (50.3) | 49.5 (48.3) | 40.2 (39.0) | 44.8 (40.5) | 38.8 (37.9) | 49.1 (46.7) | 46.0 (46.2) | **38.6*** (35.6) | 52.6 (29.4) |
| 1024 | 139.4 (136.8) | 183.4 (180.8) | 182.5 (180.1) | 140.8 (137.9) | 168.9 (139.3) | **139.2*** (136.6) | 182.0 (180.4) | 179.8 (178.2) | 139.8 (134.2) | 204.9 (127.3) |
| 4096 | 538.7 (528.6) | 709.5 (701.4) | 711.5 (703.4) | 539.6 (530.0) | 692.6 (531.6) | **536.1*** (528.4) | 707.8 (813.5) | 709.0 (701.9) | 538.8 (526.5) | 930.8 (520.4) |
| 8192 | 1069.7 (1050.9) | 1407.7 (1392.7) | 1416.0 (1398.8) | 1070.8 (1052.3) | 1392.3 (1053.6) | **1064.0*** (1050.5) | 1406.7 (1592.9) | 1413.1 (1397.2) | 1068.9 (1048.5) | 1427.4 (1041.6) |

**2 links**

| tiles | b | m | mb | ms | mh | b@auto | m@auto | mb@auto | ms@auto | mh@auto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4    | 8.2 (7.6) | 7.1 (6.6) | 6.9 (6.5) | 7.3 (6.7) | 9.0 (8.5) | 7.8 (7.3) | 6.1 (5.7) | **5.8*** (5.4) | 6.4 (6.1) | 6.7 (6.2) |
| 16   | 8.9 (8.7) | 8.1 (7.7) | 8.0 (7.4) | 8.3 (7.8) | 9.7 (9.1) | 8.6 (8.3) | 6.8 (5.9) | **6.6*** (6.2) | 7.4 (7.2) | 7.2 (6.5) |
| 64   | 13.2 (12.6) | 14.5 (13.7) | 13.1 (12.6) | 12.4 (11.4) | 14.5 (12.7) | 12.6 (12.2) | 10.7 (8.8) | 10.1 (8.9) | 10.7 (8.9) | **9.1*** (7.0) |
| 256  | 27.9 (27.0) | 34.7 (31.3) | 32.3 (29.0) | 32.1 (26.1) | 40.6 (26.7) | 22.3 (22.2) | 26.9 (23.5) | 26.0 (23.4) | 24.0 (18.8) | **21.7*** (13.1) |
| 1024 | 99.6 (97.5) | 119.6 (102.1) | 111.5 (94.4) | 110.2 (87.2) | 129.8 (85.8) | **73.2*** (71.4) | 94.1 (90.8) | 90.0 (89.5) | 80.7 (68.2) | 94.5 (65.6) |
| 4096 | 385.1 (379.9) | 456.8 (384.3) | 418.5 (359.2) | 434.2 (334.6) | 495.2 (306.5) | **271.2*** (266.4) | 361.0 (362.3) | 355.5 (350.8) | 307.6 (265.2) | 367.3 (257.2) |
| 8192 | 763.2 (758.2) | 907.1 (761.1) | 825.2 (718.5) | 869.0 (665.3) | 984.9 (601.1) | **536.2*** (527.9) | 716.7 (717.8) | 708.3 (698.5) | 613.2 (527.6) | 730.0 (517.0) |

Square shards; the tall / wide ones match square within 8% (all rows and the GB/s tables in
[`fag_modes_4ring_2d_tables.txt`](fag_modes_4ring_2d_tables.txt)).

## Reading

- **Relay is unchanged by the fabric.** `b` / `b@auto` are within 1–2% of the 1D ring at size (1064.0 vs 1050.5 µs,
  8,192 tiles, 1 link; 536.2 vs 527.9 at 2 links) and run their busiest hop at 23.5–23.7 GB/s per link direction.
- **Short multicasts are unchanged too, the 3-chip one is not.** `m`, `mb` and `ms` (multicast over at most 2 chips)
  are within 1–2% of 1D at 1 link and 8,192 tiles. `mh`, which multicasts over all 3 other chips each way, levels off
  at ~18 GB/s per link direction (1392 vs 1054 µs; `mh@auto` 1427 vs 1042), where on 1D it was the fastest variant.
- **At 2 links every multicast falls behind.** `ms@auto` is 16% and `mh@auto` 41% slower than on 1D at 8,192 tiles;
  `b@auto` keeps scaling (536 µs, 2.0× its 1-link time) and is the fastest from 1,024 tiles.
- **`mh@auto` is still the best choice at 64 tiles** (11.7 vs 16.6 µs for `b@auto`, 1 link; 9.1 vs 12.6 at 2 links),
  and whole-shard multicast `@auto` at ≤ 16 tiles, as on 1D, though `mh@auto` at 64 tiles is 46% slower than its 1D
  time (8.0 µs). At ≤ 16 tiles every variant is 2–19% slower than on 1D, relay included; this ran on a different
  Galaxy, so the fabric and the box are not separated there.
- **Uniform placement at 2 links** caps `b` at 16.5 GB/s again (1D: 16.6); `b@auto` reaches 23.5.

**Winners per link count**

| tiles/chip | 1 link | 2 links |
|---|---|---|
| 4 | `mb@auto` | `mb@auto` |
| 16 | `mh@auto` | `mb@auto` |
| 64 | `mh@auto` (30% ahead of `b@auto`) | `mh@auto` (28% ahead) |
| 256 | `ms@auto` ≈ `b@auto` ≈ `b` | `mh@auto` ≈ `b@auto` |
| ≥ 1,024 | `b@auto` ≈ `ms@auto` ≈ `b` (within 1%) | `b@auto` (10–13% ahead of `ms@auto`) |

## Files

| file | what |
|---|---|
| `fag_modes_4ring_2d.log` | `FAG_MODES` lines, all ten variants |
| `fag_modes_4ring_2d_tables.txt` | `fag_modes_analyze.py fag_modes_4ring_2d.log` |

## Reproduce

```bash
FAG_MESH=8,4 FAG_AXIS=1 MODES_FABRICS=2d_torus_xy \
MODES_VARIANTS=b,m,mb,ms,mh,b@auto,m@auto,mb@auto,ms@auto,mh@auto \
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s
```

Taken as one tt-sbatch job (plain `pytest --timeout=0`, 8 min for all 220 cases).
