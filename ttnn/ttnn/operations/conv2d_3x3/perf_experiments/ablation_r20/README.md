# ablation_r20 (Perf 20)

- `apply_hooks.py <op dir>`: adds the `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul` stubs (payload stubbed,
  CB / trid / semaphore sync kept) to an op copy, for the round-start tree (routed store, band ring, whole-stream fill
  split; the tilize stub covers both `direct_tilize` and the ring walk's `fifo_tilize`).
- `ab.sh <label> <tokens> <idx,...>`: one `--profile` run of the FLUX compare harness on the op in place; window medians
  appended to `/tmp/perf20/ab_results.txt` (this round's copy: `ab_results.txt`, labels f / fw / fws / fwst / fwstb /
  all = the cumulative peel, s_store = store alone, base_a / base_b = the round start on all 17).
- `plan.py [op dir]`: host-only (ttnn stubbed) blocking of every focus case.
- `guard_pan.sh`, `guard_tile.sh`, `guard_hash.sh`: Panoptic compare timings, single-device TILE timings, Panoptic
  output sha256; results `pan_{base,final}.txt`, `tile_{base,final}.txt` (base = round start swapped into place).
  Keep each call under the 10-minute tool limit: 5-6 Panoptic cases per run.
- `zone_hash_check.py <op dir> [...]`: profiler zone-hash collisions among the op copies' zones and every location the
  last profiled session compiled (`generated/profiler/.logs/*zone_src_locations.log`, firmware included).
- `BRIEF_COMMON.md`: the brief every optimizer of this round received.
