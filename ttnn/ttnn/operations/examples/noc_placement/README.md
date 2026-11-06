# noc_placement — placement × NoC × operation

**Measured on:** WH B0 n150 (`bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`, 2026-07-17; regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19; write split and full-grid rows: `bh-50-special-adamjanovic-for-reservation-143315`, 13×10 = 130 Tensix cores, 2026-10-26; `swap_columns`: same board, 2026-10-30)

Two kernel-level decisions govern how interleaved DRAM traffic contends on the Network-on-Chip,
and this example lets you switch and measure both:

1. **Where you place the line of cores** — spreading it across the DRAM-facing axis (a **row** or
   **diagonal**) beats stacking it on one axis (a **column**): same kernel, same work, ~2.2–2.9× less time on both WH B0 n150 and BH p150b.
   A column line is exactly what `split_work_to_cores(..., row_wise=False)` (the default) hands you.
2. **Which NoC a stream uses** — **reads want NoC0, writes want NoC1** (the tt-metal default) for the
   spread placements you actually use. It is not that NoC0 is intrinsically faster: NoC0's routing
   *disperses* column-localized DRAM traffic while NoC1's *concentrates* it (mirror-image for writes).
3. **Splitting one stream across both NoCs** (`write_split=k`: every k-th output page written on the read
   NoC) — mostly **does not pay**. On the full worker grid at the DRAM ceiling it is flat (1/2: 1.02×,
   1/4: 1.04×, BH p150b); it lifts a full-grid copy only while the DRAM write stream sits *below* the
   ceiling (1.08–1.12×); wherever the NoC, not DRAM, is the bound (a few Tensix cores, L1 destinations)
   it **loses** 3–38%, because the usual pairing already gives each NoC one stream.
4. **Swapping the pairing on the DRAM-adjacent columns only** (`swap_columns=N`: the Tensix cores in the N worker
   columns nearest DRAM read on NoC1 / write on NoC0, the rest keep reads NoC0 / writes NoC1) — the measured
   exception to rule 2. With 3 columns swapped it gains **1.05–1.15×** while the usual pairing is below the DRAM ceiling
   (short streams, a partial grid, deep issue, DRAM→L1); it is **flat** (0.99–1.01×) on a long full-grid stream at the
   ceiling and when latency-bound. Swap 2–4 columns; at 6 the gain shrinks (176 MB: 0.97×), all of them loses 0.62–0.86× (BH p150b).

**Op:** `noc_placement(t, *, op, noc, placement, num_cores=N, block=B, kernel_iters=K, memory_config=None, write_split=0, noc_mode=None, swap_columns=0, swap_anchor="first")` — interleaved DRAM
data movement over a line of `N` cores. The kernels are byte-identical across placements; only the
physical cores and the chosen NoC(s) move, so any measured difference is pure NoC routing.

## The switch space (cases are switchable)
```python
from ttnn.operations.examples.noc_placement import noc_placement, OPS, NOCS, PLACEMENTS
# OPS=("read","write","copy")  NOCS=("noc0","noc1")  PLACEMENTS=("column","row","diagonal")

noc_placement(t, op="read",  noc="noc0", placement="row")       # DRAM->L1 read bench on NoC0
noc_placement(t, op="write", noc="noc1", placement="diagonal")  # L1->DRAM write bench on NoC1
noc_placement(t, op="copy",  noc="noc0", placement="column")    # identity copy: reads NoC0, writes NoC1
noc_placement(t, op="copy",  noc="noc0", placement="grid", write_split=4)  # whole grid; every 4th page's write on NoC0
noc_placement(t, op="copy",  noc="noc0", placement="grid", noc_mode="dynamic")  # the control: dynamic mode, no split
noc_placement(t, op="copy",  noc="noc0", placement="grid", swap_columns=3)  # 3 DRAM-nearest columns: reads NoC1 / writes NoC0
```
- `op="read"` / `op="write"` isolate a **single stream** (one kernel, a fixed L1 scratch, no CB handshake
  and no partner kernel) so the NoC-for-reads vs NoC-for-writes question is answered without cross-contention.
- `op="copy"` is the realistic identity DRAM→DRAM copy: reads on `noc`, writes on the **other** NoC
  (so the two streams never share links). `noc="noc0"` = the canonical reader-NoC0 / writer-NoC1 pairing.
- Reads are issued in **blocks** (`block` pages per NoC barrier) so many transactions are in flight — this
  makes the movement **bandwidth-bound**, which is what exposes link *contention* (at `block=1` it is
  latency-bound and the effect nearly vanishes). `kernel_iters` repeats the range for steady-state throughput.
- `placement="grid"` = the first `num_cores` Tensix cores of the worker grid, row-major (default: all of them);
  `memory_config=ttnn.L1_MEMORY_CONFIG` puts the copy's output in Tensix-core L1 (an L1 input works too).
- `write_split=k` (copy only): every k-th output page is written on the **read** NoC instead of the write NoC —
  1/k of the write stream moves (k=1: all writes share the read NoC). Reads never change. One RISC-V issuing on
  both NoCs needs `NOC_MODE::DM_DYNAMIC_NOC` (both copy kernels switch; a Tensix core's DM kernels must share one
  mode), and the per-block barrier covers both NoCs. `noc_mode="dynamic"` with `write_split=0` is the control that
  isolates what the mode alone costs. At `write_split=0` the split code compiles out (`if constexpr`).
- `swap_columns=N` (copy only): the Tensix cores in the N logical worker columns nearest DRAM get the reversed
  pairing (reads on the other NoC, writes on `noc`); `"all"` = every column. The columns come from the device, not
  from constants: `dram_adjacent_columns(device, anchor)` takes the logical worker core the device names as closest to
  each DRAM bank (`ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment`) and ranks columns by distance from
  the column next to bank 0 (`swap_anchor="first"`) or from the nearest such column (`"every"`). The two core groups
  run the same kernel sources with a different NoC config, in `NOC_MODE::DM_DEDICATED_NOC`.

## 1. Best placement (op="copy")
Wormhole B0, `shape=(1024,2048)` = 2048 tiles, 8 cores, `block=16` (see `report.md` for the stamp):

```
    placement          ns/op   vs column
    column          959160.9  (baseline)
    row             330068.6  -> 2.91x
    diagonal        327144.2  -> 2.93x
```
row/diagonal are ~2.2–2.8× faster than column, and the gap **grows with core count** — the contention
signature: a column line saturates its shared NoC links and stops scaling. **Row ≈ diagonal** on WH B0 n150
and on BH p150b too (BH p150b, same copy: column 383 253 → row 164 624 ns 2.33×, diagonal 165 034 ns 2.32×) — the
diagonal's edge over the row once expected on Blackhole's asymmetric grid does not show on BH p150b at 8 Tensix cores.
`split_work_to_cores(..., row_wise=True)` spreads across the DRAM-facing axis instead of the column trap.

## 2. Which NoC for reads vs writes (op="read"/"write")
DRAM is **column-localized** — on Wormhole DRAM lives only in NoC columns 0 and 5. The two NoCs route
[dimension-ordered on a torus](../../../../../METALIUM_GUIDE.md): **NoC0 = east→south**, **NoC1 = north→west**.

- **NoC0 (reads):** traffic fans east across columns first and only turns south at the reader's *own*
  column → spread the readers across columns and no link is shared.
- **NoC1 (writes):** the mirror — traffic climbs the *DRAM* column first (concentrating on cols 0/5), then
  runs along the consumer's row.

Measured 12-cell matrix (device ns/op; see `report.md` for the stamped block, `noc_placement_matrix.html`
for the per-cell NoC link-demand heatmaps):

```
    op     placement     noc0 ns    noc1 ns   faster
    read   row            ~202k      ~969k     NoC0 4.8x
    read   diagonal       ~204k      ~503k     NoC0 2.5x
    read   column         ~789k      ~514k     NoC1 1.5x   (column = the trap; never use it)
    write  row           ~1044k      ~244k     NoC1 4.3x
    write  diagonal       ~628k      ~246k     NoC1 2.5x
    write  column         ~626k      ~921k     NoC0 1.5x
```
- **Mirror symmetry:** `read·NoC0 ≈ write·NoC1` and `read·NoC1 ≈ write·NoC0` — each pair is the same
  physical links traversed in opposite directions (writes run a touch slower from store overhead).
- **The default pairing wins:** reads→NoC0 + writes→NoC1 gives the two fast cells for row/diagonal and
  keeps the two streams on separate physical networks. (`ReaderConfigDescriptor`/`WriterConfigDescriptor`
  default to exactly this.)
- The column exception (where the reverse NoC wins) is a property of the pathological column placement —
  not a reason to swap the default.
- **BH p150b** (DRAM in NoC columns 0 and 9; 8 Tensix cores, same shape, `block=16`, `kernel_iters=8`): the same
  winner in every row — read row NoC0 106k vs NoC1 376k ns (3.5×), read diagonal 107k vs 151k (1.4×), read column
  NoC1 153k vs NoC0 292k (1.9×); write row NoC1 105k vs NoC0 402k (3.8×), write diagonal 111k vs 196k (1.8×), write
  column NoC0 195k vs NoC1 371k (1.9×). Stamped block in [`report.md`](report.md).

## 3. Split one stream's writes across both NoCs (`write_split`)
Identity copy, reads NoC0 / writes NoC1 as the baseline (`NOC_MODE` dedicated). `dynamic` = same pairing in
`NOC_MODE::DM_DYNAMIC_NOC`, no split (the control); `1/4`, `1/2`, `all` = that fraction of the output pages written
on NoC0 instead of NoC1 (dynamic mode); `swapped` = reads NoC1 / writes NoC0 (dedicated). BH p150b
(`bh-50-special-adamjanovic-for-reservation-143315`, 13×10 = 130 Tensix cores, CHIP_FREQ 1350 MHz), `block=16`
unless stated, median of 5 rotated trials × 3 launches, `test_noc_placement_write_split_perf`. GB/s = (read +
write bytes) / device kernel duration. Speedup = baseline / variant (> 1 = faster).

| Copy (2 KB bf16 tile pages) | Tensix cores | baseline ns (GB/s) | dynamic | 1/4 | 1/2 | all | swapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRAM→DRAM 32 MB, 126 tiles/core | 130 | 172 776 (388) | 1.00× | **1.04×** (403) | **1.02×** (395) | 0.77× | 0.83× |
| DRAM→DRAM 64 MB, 252 tiles/core | 130 | 373 606 (359) | 1.00× | **1.09×** (393) | **1.09×** (390) | 0.84× | 0.92× |
| DRAM→DRAM 128 MB, 504 tiles/core | 130 | 755 836 (355) | 1.00× | 1.04× (370) | **1.08×** (385) | 0.85× | 0.94× |
| DRAM→DRAM 32 MB, 256 tiles/core | 64 | 186 289 (360) | 1.01× | 1.05× (376) | 1.04× (374) | 0.78× | 0.82× |
| L1→DRAM 16 MB (only the writes touch DRAM) | 130 | 60 321 (556) | 1.00× | **1.12×** (625) | 1.01× | 0.57× | 0.56× |
| DRAM→L1 16 MB (writes land in L1) | 130 | 42 786 (784) | 1.00× | 0.93× | 0.88× | 0.76× | 0.62× |
| L1→L1 16 MB (no DRAM) | 130 | 36 337 (923) | 1.02× | 0.78× | 0.75× | 0.67× | 0.86× |
| DRAM→DRAM 4 MB, row, `kernel_iters=2` | 8 | 54 025 (311) | 1.00× | 0.97× | 0.78× | 0.47× | 0.50× |
| DRAM→DRAM 4 MB, row, `kernel_iters=2` | 2 | 113 457 (148) | 0.91× | 0.83× | 0.81× | 0.70× | 0.77× |
| DRAM→DRAM 4 MB, `kernel_iters=2` | 1 | 221 646 (76) | 0.90× | 0.82× | 0.80× | 0.78× | 0.94× |
| same, `block=64` | 1 | 137 031 (122) | 0.73× | 0.64× | 0.62× | 0.69× | 1.00× |

Grid 32 MB at other block sizes: `block=8` 1/4 0.99×, 1/2 0.97×; `block=32` 1/4 1.03×, 1/2 1.01× (exploratory runs).

- **At the DRAM ceiling it is flat.** Every full-grid DRAM→DRAM variant tops out at ~390–405 GB/s (the full-grid
  DRAM→L1 copy reads DRAM at 392 GB/s, the same ceiling), so moving writes to the other NoC cannot add DRAM
  bandwidth: 32 MB, already at 388 GB/s, gains 1–4%; 1/2 ranges 0.97–1.02× over `block` 8–32.
- **Below the ceiling it lifts the copy to the ceiling, not past it.** The 64 / 128 MB footprints and the 64-core
  grid run at 355–360 GB/s unsplit; 1/4 or 1/2 brings them to 370–393 GB/s (1.04–1.09×). The write-only DRAM copy
  (L1→DRAM) writes DRAM at only 278 GB/s on NoC1; a quarter on NoC0 lifts it to 312 GB/s (1.12×), a half
  overshoots (1.01×). Consistent with the NoC1 write path into the DRAM endpoints, not DRAM, limiting those cases;
  not isolated.
- **Where the NoC is the bound it loses.** The usual pairing already spreads the copy as one stream per NoC; moving
  writes onto NoC0 piles them on the read stream (L1 destinations 0.75–0.93×, an 8-core row 0.78× at 1/2).
- **On 1–2 Tensix cores `NOC_MODE::DM_DYNAMIC_NOC` itself costs 9–27%** (`dynamic` column), before any split: the
  copy is RISC-V issue/latency-bound there, and dynamic mode's per-transaction bookkeeping is on that path. On
  ≥ 8 Tensix cores the mode is free (0.99–1.02×).
- **Swapping the pairing** (reads NoC1, writes NoC0) costs 0.83× on the full grid at 32 MB (0.92–0.94× at
  64–128 MB), 0.50× on an 8-core row, 0.56–0.62× when one side is L1, and ~0.95–1.00× on one Tensix core.

## 4. Swap the pairing on the DRAM-adjacent columns only (`swap_columns`)
Identity copy over the whole worker grid unless stated, interleaved DRAM in and out, `kernel_iters=1`. Baseline =
every Tensix core reads on NoC0 / writes on NoC1. On this BH p150b (13×10 = 130 Tensix cores) the device names
logical columns 0 (banks 0–3) and 7 (banks 4–7) as nearest DRAM, so `f3` = columns 0, 1, 2 (the three westmost) and
`e4` = columns 0, 1, 6, 7. Median of 5 rotated trials × 3 launches (`test_noc_placement_swap_columns_perf`); size =
read + write bytes; speedup = baseline / variant.

| Copy | pages per barrier | baseline ns (GB/s) | f1 | f2 | **f3** | f4 | f6 | e4 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 MB, 15 tiles/core | 16 | 24 623 (341) | 1.05× | 1.08× | **1.15×** | 1.17× | 1.14× | 1.17× | 0.71× |
| 34 MB, 63 tiles/core | 1 | 91 439 (367) | 1.01× | 1.01× | **0.99×** | 1.03× | 1.00× | 1.03× | 0.82× |
| 34 MB | 16 | 83 774 (400) | 1.01× | 1.02× | **1.03×** | 1.03× | 0.99× | 1.04× | 0.76× |
| 34 MB | 64 | 83 755 (401) | 1.02× | 1.04× | **1.07×** | 1.05× | 1.00× | 1.08× | 0.74× |
| 67 MB, 126 tiles/core | 16 | 173 869 (386) | 1.02× | 1.04× | **1.05×** | 1.06× | 1.03× | 1.07× | 0.83× |
| 176 MB, 330 tiles/core | 16 | 432 265 (408) | 1.01× | 1.01× | **1.00×** | 1.01× | 0.97× | 1.02× | 0.82× |
| 176 MB | 64 | 439 761 (401) | 1.03× | 1.02× | **1.01×** | 1.02× | 0.98× | 1.04× | 0.80× |
| 65 Tensix cores (5 rows), 67 MB | 16 | 187 819 (357) | 1.04× | 1.07× | **1.11×** | 1.14× | 1.09× | 1.13× | 0.86× |
| DRAM→L1 34 MB | 16 | 42 648 (787) | 1.03× | 1.06× | **1.09×** | 1.10× | 0.98× | 1.12× | 0.62× |

All cells (incl. `block` 2, 4, 8, 32, 128 and `e2`/`e3`/`e6`) are in [`report.md`](report.md).

- **It helps where the usual pairing sits below the DRAM ceiling and the copy is bandwidth-bound:** short full-grid
  streams (8 MB: 1.15×), a partial grid (65 Tensix cores: 1.11×), deep issue (34 MB at 64 pages per barrier: 1.07×),
  and the NoC-bound DRAM→L1 copy (1.09–1.12×). It lifts those copies to ~390–430 GB/s, not past the ceiling.
- **It does not help** on a long full-grid stream already at the ceiling (176 MB, ~405 GB/s: 1.00–1.04×) or when
  latency-bound (1 page per barrier: 1.00×). Pages per barrier matter: the 34 MB copy goes 1.00× → 1.03× → 1.07× at
  1 → 16 → 64.
- **How many columns:** 3–4 from one DRAM column, or 2 on each side of each DRAM-adjacent column (`e4`), which is
  best or within 0.01× of best in every case. At 6 columns (46% of the grid) the gain shrinks or turns into a
  loss (8 MB 1.14×, 34 MB 0.99×, 176 MB 0.97×); swapping every
  column is the full reversed pairing and loses (0.62–0.86×). The rule "reads NoC0 / writes NoC1" still holds for
  most of the grid.
- **Field evidence** on the same board type (3 westmost columns, another kernel): 33 MB 83.5 → 75.3 µs (1.11×),
  168 MB 390.1 → 375.0 µs (1.04×). This reconstruction matches the 34 MB baseline (83.8 µs) and gets 1.07× at 64
  pages per barrier (1.03× at 16); at 176 MB its baseline is slower (432 µs) and the gain smaller (1.00–1.04×). The
  difference is not explained.
- **Mechanism: not isolated.** Consistent with the cores next to DRAM finishing last under the usual pairing: the gain
  is largest when completion is uneven (short streams, partial grid) and vanishes when DRAM bandwidth holds every core
  back. Per-column times were not measured.

## When it helps / when it does not
**Rule:** both levers pay once the line of Tensix cores is **bandwidth-bound** — on WH B0 n150 from ≳16 outstanding
2 KB reads (Tensix cores × `block`); at ≤ 8 they are within ~10%. Same verdicts on BH p150b, with smaller gains
(2.2–2.5× vs 2.9–3.3×). For a single Tensix core the NoC choice does not help on either board — WH B0 n150 even
loses 6% with the canonical pairing, BH p150b is flat. Identity copy (`op="copy"`), `shape=(1024, 2048)` = 2048
bf16 tiles (2 KB pages), interleaved DRAM (`TensorMemoryLayout::INTERLEAVED`); line length capped at 8 Tensix
cores so a diagonal fits. "Outstanding reads" = Tensix cores × `block` (2 KB pages in flight per barrier per
Tensix core).

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Placement helps:** column → row, 8 Tensix cores, `block=16` (128 outstanding reads) | 246 765 → 84 917 ns **2.91×** | 97 835 → 43 723 ns **2.24×** |
| **Placement does not help:** column → row, 2 Tensix cores, `block=1` (2 outstanding reads, latency-bound) | 1 106 889 → 1 077 065 ns **1.03×** | 838 190 → 829 478 ns **1.01×** |
| **NoC helps:** reads NoC1 → NoC0 (writes on the other NoC), row of 8 Tensix cores, `block=16` | 278 002 → 85 066 ns **3.27×** | 107 508 → 43 806 ns **2.45×** |
| **NoC does not help:** reads NoC1 → NoC0, 1 Tensix core at (0,0), `block=16` | 404 600 → 429 238 ns **0.94× (loses)** | 216 865 → 211 690 ns **1.02× — flat on BH, no loss** |
| Copy column → row, 8 Tensix cores, `block=16`, `kernel_iters=8` (device_perf) | 959 161 → 330 069 ns **2.91×** (diagonal 2.93×) | 383 253 → 164 624 ns **2.33×** (diagonal 2.32×) |
| Sweep: column → row, 4 Tensix cores × `block=1` (4 outstanding) | **1.04×** | not measured |
| Sweep: column → row, 8 Tensix cores × `block=1` (8 outstanding) | **1.09×** | not measured |
| Sweep: column → row, 8 Tensix cores × `block=2` (16 outstanding) | **1.40×** | not measured |
| Sweep: column → row, 2 Tensix cores × `block=16` (32 outstanding) | **1.31×** | not measured |
| Sweep: column → row, 4 Tensix cores × `block=16` (64 outstanding) | **2.20×** | not measured |
| **Boundary** | ≳16 outstanding 2 KB reads | between 2 (flat) and 128 (helps) outstanding reads; not swept |
| **NoC helps (full grid):** reads NoC1 → NoC0, 130 Tensix cores, 32 MB DRAM→DRAM, `kernel_iters=1` | not measured | 208 978 → 173 195 ns **1.21×** (13×10 board) |
| **Write split helps:** 1/4 of writes on NoC0, 130 Tensix cores, 16 MB L1→DRAM (DRAM write stream below the ceiling) | not measured | 60 141 → 53 772 ns **1.12×** (13×10 board) |
| **Write split does not help:** 1/2 of writes on NoC0, 130 Tensix cores, 32 MB DRAM→DRAM (at the DRAM ceiling, 388 GB/s) | not measured | 173 706 → 170 239 ns **1.02×** (13×10 board) |
| **Write split loses:** 1/2 on NoC0, 130 Tensix cores, 16 MB DRAM→L1 (NoC-bound) | not measured | 42 777 → 48 763 ns **0.88×** (13×10 board) |
| **Write split loses:** 1/2 on NoC0, row of 8 Tensix cores, 4 MB DRAM→DRAM, `kernel_iters=2` | not measured | 54 146 → 68 922 ns **0.79×** (13×10 board) |
| **Column swap helps:** `swap_columns=3`, 130 Tensix cores, 8 MB DRAM→DRAM read+write, `block=16` | not measured | 24 675 → 21 474 ns **1.15×** (13×10 board) |
| **Column swap helps:** `swap_columns=3`, 65 Tensix cores, 67 MB, `block=16` | not measured | 186 766 → 169 068 ns **1.10×** (13×10 board) |
| **Column swap helps:** `swap_columns=3`, 130 Tensix cores, 34 MB, `block=64` | not measured | 84 134 → 78 665 ns **1.07×** (13×10 board) |
| **Column swap does not help:** `swap_columns=3`, 130 Tensix cores, 176 MB, `block=16` (at the DRAM ceiling) | not measured | 432 281 → 429 853 ns **1.01×** (13×10 board) |
| **Column swap does not help:** `swap_columns=3`, 130 Tensix cores, 34 MB, `block=1` (latency-bound) | not measured | 91 253 → 91 147 ns **1.00×** (13×10 board) |

**Why the boards differ:**
- **Smaller gains on BH p150b.** BH's NoC flit is 64 B (WH 32 B), so each link carries twice the bytes and a
  column line's shared links saturate less; the column baseline is ~2.5× faster on BH p150b (97.8 vs 246.8 µs)
  while the row lever is ~1.9× faster. Consistent with the link-width difference, not isolated.
- **Single Tensix core:** WH B0 n150 loses 6% with reads on NoC0 at (0,0); BH p150b does not (1.02× on the 12×10
  board, 1.06× on the 13×10 board — so `noc_flat_1core_block16` asserts ≤ 1.08×). The DRAM cores sit in different
  NoC columns (WH 0 and 5, BH 0 and 9), so the one-core route differs; not investigated.
- **Second BH p150b board (13×10 = 130 Tensix cores):** the line regimes keep their verdicts with different sizes —
  placement 1.80× (12×10 board 2.24×), NoC 1.97× (2.45×), latency-bound placement 1.00×.

- **Placement helps** when the line's outstanding reads contend: a column funnels every Tensix core's DRAM traffic
  through the same column links, a row spreads it across columns, so the row moves the same bytes 2.2–2.9× faster.
- **Placement stops paying** when the line is latency-bound (one page per barrier, few Tensix cores): no link is
  saturated, so each read costs one round trip wherever the Tensix core sits.
- **NoC0 for reads helps** on a spread line because NoC0 (east→south) fans the traffic out across columns before
  turning toward DRAM, while NoC1 (north→west) concentrates it on the DRAM columns.
- **The NoC choice does not help** without a line to disperse: one Tensix core has nothing to spread — keep the
  default for multi-core lines, but do not expect it to help a single-Tensix-core kernel.
- **The column swap helps** while the full-grid copy is below the DRAM ceiling and bandwidth-bound (short stream,
  partial grid, deep issue, DRAM→L1) with 2–4 DRAM-adjacent columns swapped; **it does not help** at the ceiling or
  when latency-bound, and swapping most of the grid loses.
- **The write split helps** only while the write stream into DRAM is below the DRAM ceiling on a full grid (a
  quarter of the pages on the read NoC); **it does not help** at the ceiling (flat) and **loses** wherever the NoC is
  the bound, because the usual pairing already gives each NoC one stream.

Small payloads still benefit on WH B0 n150 if the line is bandwidth-bound: 64 tiles (8 tiles per Tensix core,
`block=16`, `kernel_iters=1`) measured 2.65×, 16 tiles (2 tiles per Tensix core) 1.51×. The NoC choice for reads is
likewise flat when latency-bound (`block=1`: 1.04× at 2 Tensix cores, 1.07× at 8 Tensix cores; `op="read"`, row).

*How measured:* `test_noc_placement_regimes`, `DEVICE KERNEL DURATION [ns]`, baseline/lever interleaved
(A,B,A,B…), median of 12 trials × 3 launches, `kernel_iters=2` unless stated (the full-grid rows: `kernel_iters=1`).
- **WH B0 n150:** commit `adce58258c3` + test, 2026-10-12 (three runs agree within 0.01×); sweep rows from
  tt-probe (`kernel_iters=8`); device_perf row from the 2026-07-17 report ([`report.md`](report.md)).
- **BH p150b:** the same four cases (the fourth, then named `noc_loses_1core_block16`, is now
  `noc_flat_1core_block16`); device_perf row from `test_noc_placement_device_perf` (20 launches averaged).
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.
- **BH p150b, 13×10 board:** full-grid and write-split rows, `bh-50-special-adamjanovic-for-reservation-143315`,
  2026-10-26, commit `5eac7659e6c` + this example's write-split change, CHIP_FREQ 1350 MHz.
- **BH p150b, 13×10 board, column swap rows:** same box, 2026-10-30, commit `30351ce33f5` + this example's
  `swap_columns` change, CHIP_FREQ 1350 MHz.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).

## Measure your own shapes/params
```bash
python -m ttnn.operations.examples.noc_placement --shape 2048,2048 --cores 8 --block 32
#   --shape H,W   tile-aligned bf16 tensor      --cores  line length (<= min grid dim)
#   --block N     pages in flight per barrier   --kernel-iters N  in-kernel repeat   --iters N  profiled launches

# write split vs the usual pairing (+ the dynamic-mode control and the swapped pairing) on one copy
python -m ttnn.operations.examples.noc_placement --write-split --placement grid --shape 8192,4096 --in-mem dram --out-mem dram
#   --placement grid|row|column|diagonal  --cores N (default: whole grid)  --in-mem/--out-mem dram|l1  --trials N

# swap the pairing on the N DRAM-nearest columns (N = 1,2,3,4,6 from bank 0's column, 2,3,4,6 from every DRAM column, all)
python -m ttnn.operations.examples.noc_placement --swap-columns --shape 2048,4096 --block 64
#   same options as --write-split
```

## Regenerate the report (`noc_placement_matrix.html`)
Unlike the other examples, the matrix visual is **reconstructed from code + tt-npe** (not hand-written).
`noc_report.py` drives: device-ns pass → NoC-trace capture (`profile_this.py --collect-noc-traces`) →
tt-npe simulation → aggregate → HTML. It needs [tt-npe](https://github.com/tenstorrent/tt-npe) built and
reachable via `$TT_NPE_HOME` (or at `<tt-metal>/../tt-npe`):
```bash
python -m ttnn.operations.examples.noc_placement --report   # rewrites noc_placement_matrix.html + report.md
```

## Run the committed sweep
```bash
# correctness (every copy placement/NoC is the identical identity copy)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_correctness
# device kernel duration for the 12-cell read/write × NoC × placement matrix
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_device_perf
# helps / does-not-help regimes (asserted speedups)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes
# write split: bitwise identity copy on every split / pairing, then the measured sweep (section 3's table)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_correctness
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_perf
# column swap: bitwise identity copy for N in {1,3,6,all} x both anchors, then the measured sweep (section 4's table)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_correctness
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_perf
```

## Code
`noc_placement.py` (op; `_ordered_cores()` placements incl. `grid`, `_dm_config()` NoC + `NOC_MODE`,
`dram_adjacent_columns()` / `_swapped_columns()` and the two kernel groups in `_copy()` for `swap_columns`),
`kernels/{copy_reader,copy_writer,read_bench,write_bench}.cpp` (`copy_writer.cpp` holds the `write_split` page
routing and the two-NoC barrier), `noc_report.py` (report pipeline). Numbers are single-box per arch (WH B0 n150, BH p150b) — illustrative of the mechanism, not a CI bound.
