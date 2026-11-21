# ablation_r19 (Perf 19)

- `apply_hooks.py <op dir>`: adds the `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul` stubs (payload stubbed,
  CB / trid / semaphore sync kept) to an op copy, for the round-start tree (C_out fold, ring startup split, kh reuse).
- `ab.sh <label> <tokens> <idx,...>`: one `--profile` run of the FLUX compare harness on the op in place; window
  medians appended to `/tmp/perf19/ab_results.txt` (this round's copy: `ab_results.txt`).
- `guard_pan.sh`, `guard_tile.sh`, `guard_hash.sh`: Panoptic compare timings, single-device TILE timings, Panoptic
  output sha256 (keep each call under the 10-minute tool limit: 5-6 Panoptic cases per run).
- `zone_hash_check.py <op dir> [...]`: lists profiler zone locations that share a 16-bit hash (a `--profile` run that
  compiles both throws).
- `BRIEF_COMMON.md`: the brief every optimizer of this round received (focus table, breakdown, code map).
