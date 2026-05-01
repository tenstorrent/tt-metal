# Ring Joint SDPA — Uniform `q_chunks_per_core` (CT)

## What changed

Per-core Q-chunk loop count is now a **compile-time arg** (`q_chunks_per_core`)
and is **uniform across all cores**. Previously cores got `base` or `base+1`
chunks (or `base`/`base+2` in zigzag), with the K-mcast injector loop padded
to `max_q_per_core` via an RT arg.

After the change:

- `q_chunks_per_core = ceil(total_q_chunks / num_cores)` (or
  `ceil(pairs/num_cores) * 2` for zigzag), computed once in the program factory
  and passed as a CT arg to all three kernels.
- Each core's `global_q_start = i * q_chunks_per_core`. Real chunk count is
  capped to `min(q_chunks_per_core, total - global_q_start)`; trailing cores
  may have phantom iterations.
- Reader's outer Q loop count is the CT `q_chunks_per_core` for **all** paths.
- Phantom iters reuse the existing K-mcast padded-iter mechanism (reserve `2×`
  tiles so write ptr stays put, do the mcast handshake, skip `cb_push_back` /
  Q / V). Non-K-mcast phantom iters `continue` at the top of the loop (no
  cross-core sync needed).
- Compute & writer keep RT-bounded loops over real chunks (CB sync via reader).
- Dropped RT args: `max_q_per_core` from the K-mcast args block.

## Files touched

- `ring_joint_sdpa_program_factory.cpp` — uniform distribution, new CT arg in
  reader/writer/compute arg lists, K-mcast `next_core_q_chunks =
  q_chunks_per_core`.
- `kernels/dataflow/ring_joint_reader.cpp` — CT `q_chunks_per_core` (slot 26),
  loop count is CT, drop RT `max_q_per_core`, non-K-mcast phantom skip.
- `kernels/dataflow/ring_joint_writer.cpp` — CT `q_chunks_per_core` (slot 29),
  TensorAccessorArgs offset bumped.
- `kernels/compute/ring_joint_sdpa.cpp` — CT `q_chunks_per_core` (slot 40,
  informational; compute loops over real chunks via CB sync).
- `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` — added
  `RING_JOINT_QK_FILTER` env-var to scope the perf-table sweep to a specific
  `q{N}-k{N}` substring.

## How to test

Activate the python env and use `run_safe_pytest.sh` (manages device locks
and resets after the run; never run `tt-smi -r` directly):

```bash
source python_env/bin/activate
```

### Accuracy — mla_100k (Blackhole, all 3 chunk sizes)

```bash
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy \
  -k mla_100k
```

Last result on this branch: 3/3 PASS, PCC ≈ 0.9996, RMSE ≈ 0.0066.

### Perf table — narrow to a single (q, k) pair

```bash
RING_JOINT_QK_FILTER=q160-k320 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table \
  -k mla_100k
```

Drop `RING_JOINT_QK_FILTER` to sweep all chunk sizes for the model.

Last result on this branch: q=160, k=320 → 5.177 ms, math util 58.4%,
87/100 compute cores, 280 iters/core.
