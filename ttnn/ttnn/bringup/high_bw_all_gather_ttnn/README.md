# ttnn.bringup.high_bw_all_gather (high_bw_all_gather_ttnn)

## What it does
Fork of `ttnn.experimental.high_bw_all_gather`. Adds a row RANGE: with `gathered_dim_start`, each rank moves its
local range `[gathered_dim_start, gathered_dim_start + gathered_dim_size / ranks)` along `dim` into the same range
of its fixed worst-case output slot (the stock op moves only the prefix `[0, gathered_dim_size / ranks)`). Bytes
outside the range stay unchanged, so several ranges in a row compose a prefix: a long history gathered in pieces,
each short enough to hide beside one compute phase. Everything else is the stock op.

## Consider it instead of
`ttnn.experimental.high_bw_all_gather`, when one gather of a persistent buffer is to be split into pieces (rows
[a, b) of every rank) without a slice / slice_write copy of each piece.

## Options
- `gathered_dim_start` (default None = the prefix, the stock program: no define, no extra arg): the first local
  element of the range along `dim`; needs `gathered_dim_size` (the range's total length over the ranks), an
  interleaved input and no `gathered_prefix_tensor`. Runtime (hash-excluded; its presence adds the kernel define
  `HBAG_FIRST_PAGE` and one common arg): one program serves every start.

## Measured
- deepseek_v41_flash_d_p PB.6L, 8x4 Galaxy, one 1M history item (32 owners x 16,256 rows of 320 B records, 158.8 MiB
  a chip) on the gather column (10 cores, x = 11): the whole prefix 2.75 ms, half 1.51, a quarter 0.87 (stock
  prefixes; a range costs what a prefix of its length does): tests/packed/test_long_hide_probe.py (bench).

## Limits
- Interleaved input only (an ND-sharded input converts its whole shard first: refused with a start).
- Not with the trace-safe `gathered_prefix_tensor` (refused).
- Tested on the 8x4 Galaxy (FABRIC_2D torus, cluster_axis None and 0), ROW_MAJOR uint8 (320 B) and bf16 (1024 B) rows.
- Source tests not carried / baseline not recorded: owner rule for the B tasks (2026-11-19: no fork regression or unit
  suites on the 8x4; run only the tests you add); `tests/source.yaml` lists none yet.

## How to call
```
ttnn.bringup.high_bw_all_gather(x, 2, out, cluster_axis=None, gathered_dim_size=owners * (b - a),
                                gathered_dim_start=a, ready_semaphore=s0, data_valid_semaphore=s1)
```
Tests: `tests/unit/test_range_gather.py` (on the resident runner through
`models/demos/deepseek_v41_flash_d_p/tests/packed/test_range_gather_fork.py`).

## Used by
deepseek_v41_flash_d_p (PB.6L: tt/history_gather.prefetch_gather_range, PACKED_LONG_HIDE=on)

## History
CHANGELOG.md
