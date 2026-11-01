# Refinement 3: streamed row statistics + per-window normalize

`bench.sh <label> [LN_KNOBS]` runs the 12 perf + guard rows once under `--profile` (summary in `logs/<label>.txt`,
device log copied to `/tmp/ln3/<label>.device.csv`). `rep.sh <label> <n> [LN_KNOBS]` repeats the four focus rows.
`phases.py`, `row1.py`, `dram_profile.py` read the device log: per-core phase percentiles, per-zone durations
of the 2-row cores' second row, and aggregate DRAM read/write GB/s per microsecond.

Variants (BH p100a, device kernel us; base = 1a03660c1f4, 5-run medians unless noted):

| variant | focus1 | focus2 | focus3 | notes |
|---|---|---|---|---|
| base | 36.5 | 47.6 | 34.0 | |
| A streamed one-pass, all cores | 35.8-36.2 | 48.8-50.9 | 34.4 | unshifted: wrong at mean 100 with a residual |
| B streamed mean, two-pass | 36.3-37.9 | 49.0-49.8 | 34.4 | never a win |
| C resident one-pass | 36.7 | 49.0 | 33.6 | |
| D resident two-pass (raw window) | 36.7 | 47.7 | 34.3 | = base; the gated fallback |
| E A + per-window normalize, all cores | 40.5-41.9 | 49.2-52.3 | 34-35 | heavy cores' row-1 residual read 5.6 -> 16 us |
| F E gated to balanced splits | 35.9 | 48.9 | 34.2 | |
| H F + STREAMED_BLOCKS=1 | 36.3 | 51.8 | 34.6 | |
| I per-window normalize on heavy cores | 36.2-37.1 | 49-51 | 34.5 | |
| K shifted one-pass (correct), all unbalanced L1 | 37.9 | 48.3 | 34.6 | DEST-reuse pass per tile |
| final (split-class gate) | 36.45 (8 runs) | 47.15 | 34.15 | balanced rows -1.3..-2.9 |
