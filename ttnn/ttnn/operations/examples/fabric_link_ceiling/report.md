# fabric_link_ceiling — device report

Illustrative, not a CI bound. Metric: `DEVICE KERNEL DURATION [ns]` (in-process device profiler), the
slowest chip of each launch, median of 3 launches. GB/s = bytes streamed per link per direction ÷ kernel ns.
Correctness (every receiving chip's landing ring == its peer's source ring) passed for every row.

## Blackhole — `bh-qb-11-special-mstaletovic-for-reservation-93463` · 4× p150a (2×2 mesh, axis-0 pairs) · 2026-09-29 · `13f181e3779`

Setup: 64 MiB per link per direction per launch; one sender core per link at logical (link, 0); source and
landing rings of 8 packets in L1; sender RISCV_0 (NoC1), receiver RISCV_1 (NoC0).

```
fabric=FABRIC_1D  router max payload=4352 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      2421789              27.71
header_ring       uni      1      2156060              31.13
flush_per_packet  bi       1      2431674              27.60
header_ring       bi       1      1995596              33.63
flush_per_packet  uni      2      2421520              27.71
header_ring       uni      2      2156764              31.12
flush_per_packet  bi       2      2566320              26.15
header_ring       bi       2      1997559              33.59

fabric=FABRIC_1D  router max payload=8704 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      1607142              41.76
header_ring       uni      1      1570961              42.72
flush_per_packet  bi       1      1711439              39.21
header_ring       bi       1      1702161              39.43
flush_per_packet  uni      2      1690624              39.69
header_ring       uni      2      1679938              39.95
flush_per_packet  bi       2      1861296              36.05
header_ring       bi       2      2082284              32.23

fabric=FABRIC_1D  router max payload=14336 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      1382881              48.53
header_ring       uni      1      1382624              48.54
flush_per_packet  bi       1      1390693              48.25
header_ring       bi       1      1390381              48.27
flush_per_packet  uni      2      1668025              40.23
header_ring       uni      2      1649036              40.69
flush_per_packet  bi       2      1786581              37.56
header_ring       bi       2      2198116              30.53

fabric=FABRIC_1D  router max payload=15232 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      2037563              32.93
header_ring       uni      1      1997433              33.59
flush_per_packet  bi       1      2211690              30.34
header_ring       bi       1      2181251              30.76
flush_per_packet  uni      2      2219359              30.23
header_ring       uni      2      2195279              30.56
flush_per_packet  bi       2      2465627              27.21
header_ring       bi       2      2507496              26.76

fabric=FABRIC_2D  router max payload=4352 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      2707227              24.79
header_ring       uni      1      2710469              24.76
flush_per_packet  bi       1      2713441              24.73
header_ring       bi       1      2715666              24.71
flush_per_packet  uni      2      2708787              24.77
header_ring       uni      2      2710355              24.76
flush_per_packet  bi       2      2712479              24.74
header_ring       bi       2      2713833              24.73

fabric=FABRIC_2D  router max payload=8704 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      1598284              41.99
header_ring       uni      1      1540325              43.57
flush_per_packet  bi       1      1642348              40.86
header_ring       bi       1      1590727              42.19
flush_per_packet  uni      2      1654217              40.57
header_ring       uni      2      1670405              40.17
flush_per_packet  bi       2      1903090              35.26
header_ring       bi       2      2030613              33.05

fabric=FABRIC_2D  router max payload=14336 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      1766091              38.00
header_ring       uni      1      1744786              38.46
flush_per_packet  bi       1      1918910              34.97
header_ring       bi       1      1892085              35.47
flush_per_packet  uni      2      1936988              34.64
header_ring       uni      2      1859089              36.10
flush_per_packet  bi       2      2128177              31.53
header_ring       bi       2      2183477              30.73

fabric=FABRIC_2D  router max payload=15232 B
variant           dir  links    kernel ns  GB/s per link-dir
flush_per_packet  uni      1      1599980              41.94
header_ring       uni      1      1603267              41.85
flush_per_packet  bi       1      1766624              37.98
header_ring       bi       1      1760287              38.12
flush_per_packet  uni      2      1647414              40.73
header_ring       uni      2      1650785              40.65
flush_per_packet  bi       2      1961933              34.20
header_ring       bi       2      2173286              30.87
```

### Placement × sender NoC (2 links, header_ring, FABRIC_1D, 14,336 B)

Logical cores → NoC coords on this box. The two links' Ethernet cores are at NoC (3,1) and (4,1) (read from a
NoC trace). `under_eth` puts each link's core in its Ethernet core's column.

```
placement  NoC coords      send  dir     kernel ns  GB/s per link-dir
adjacent   (1,2) (2,2)     NoC1  uni       1651618              40.63
adjacent   (1,2) (2,2)     NoC1  bi        2198841              30.52
rows       (1,2) (1,3)     NoC1  uni       1651465              40.63
rows       (1,2) (1,3)     NoC1  bi        1695630              39.58
under_eth  (3,2) (4,2)     NoC1  uni       1383376              48.51
under_eth  (3,2) (4,2)     NoC1  bi        1389985              48.28
adjacent   (1,2) (2,2)     NoC0  uni       1651721              40.63
adjacent   (1,2) (2,2)     NoC0  bi        HANG (deterministic; sender stuck in noc_async_write to its router)
under_eth  (3,2) (4,2)     NoC0  uni       1382639              48.54
```
