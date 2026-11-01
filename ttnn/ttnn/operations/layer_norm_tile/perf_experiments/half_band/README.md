# Refinement 4: half-tile (16-row) band work split (R3)

`bench.sh <label> [LN_KNOBS] [-k expr]` runs the perf + guard rows once under `--profile`; `rep.sh <label> <n>
[LN_KNOBS] [-k expr]` repeats a selection and prints medians (`logs/rep_<label>.txt`). The `-k` expression goes
through `PYTEST_ADDOPTS` (run_safe_pytest.sh re-splits a quoted `-k`). `core_zones.py <device csv> <run id>
<rows>` prints the zone timeline of the latest-ending and a median core with that many compute rows.

Device kernel us, BH p100a, medians (R1 = `BAND_ROWS=32`):

| variant | focus1 | focus2 | focus3 | notes |
|---|---|---|---|---|
| R1 | 36.5 | 48.3 | 34.3 | |
| R3 even-first (3-band cores at the grid bottom) | 35.4 | 47.3 | 40.0 | |
| R3 heavy-first | 35.6 | 49.2 | 33.2 | |
| R3 even-first, heavy readers in the bottom half on NoC0 | 35.4 | 46.9 | 41.0 | parked HEAVY_READER_NOC_BY_Y |
| R3 without the Perf 1 NoC plan | 50.8 | 52.7 | 40.2 | |
| R3 + streamed stats / per-window normalize (EF / HF) | 38.2 / 39.9 | 48.4 / 51.8 | 38.2 / 37.4 | parked HALF_BAND_STREAM |
| shipped gate | 35.7 | 47.25 (pooled n=16; R1 47.9) | 33.5 | |

Under-filled shapes, one band per core vs R1 (3 runs): 32x32 19.8 -> 18.6, 3x500 21.9 -> 18.6, 2x512 14.7 -> 13.3;
R3 that unbalances a balanced R1 split: 8x384 36.1 -> 38.4, 64x17 21.7 -> 22.7 (kept on R1).

focus2 timeline (R3): every core's first compute row lands at ~23-25 us (DRAM saturated by the whole grid); the
3-band cores' lone band then reads 23.6 -> 42 us (24 KiB at ~1.3 GB/s) while the light cores' output writes take
the DRAM, and its full-tile compute (~7 us) + write end the op at ~50 us. ~18.9 MB in ~47 us is ~400 GB/s
against ~448 GB/s peak.
