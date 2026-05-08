# Ring Joint SDPA: Compute-side Q/K/V wait zones

Measures how long the compute kernel blocks on each of the three input CBs
(`cb_q_in`, `cb_kt_in`, `cb_v_in`) during ring joint SDPA. If `WAIT_*` zones
are short, the reader is feeding compute fast enough; if they are long,
data movement is on the critical path.

Zones are gated to a single ring iteration so the timeline stays readable
on rings with many iterations.

## Knobs

`compute_streaming.hpp`:

- `PROFILE_RING_ITER_N` — which ring iteration to instrument (0..ring_size-1).
  Defaults to 0. Edit and rerun to retarget; the kernel JIT cache rebuilds
  automatically because the source hash changes.

`ring_joint_sdpa_program_factory.cpp`:

- Compute kernel `opt_level` is set to `O2` (default for compute is `O3`).
  Required: at `O3`, the extra zone code pushes the program past the TENSIX
  kernel-config buffer (~70 KB) and you get `Program size (...) too large
  for kernel config buffer` at `EnqueueMeshWorkload`.

## Cache gotcha

TT-Metal's JIT cache key does **not** include the per-kernel `opt_level`
(only the build-system default). If you change `opt_level` in
`ComputeConfig` alone, stale O3 binaries are still served from
`~/.cache/tt-metal-cache/<id>/kernels/ring_joint_sdpa/` and you'll hit the
program-size error. Delete that subdir to force a rebuild:

```bash
rm -rf ~/.cache/tt-metal-cache/*/kernels/ring_joint_sdpa
```

(Changing `PROFILE_RING_ITER_N` does invalidate the cache because it lives
in the source.)

## Run

Requires a 4-device Blackhole ring topology.

```bash
source python_env/bin/activate
SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check[mla_100k-q160-k320-ring4]"
```

Tracy file lands at:

```
generated/profiler/ttnn_ring_joint_sdpa_perf_check/.logs/tracy_profile_log_host.tracy
```

Open it in Tracy and filter compute-thread zones for `WAIT_Q`, `WAIT_K`,
`WAIT_V`. They appear only on the iteration matching `PROFILE_RING_ITER_N`.

## Sweep all ring iterations

To compare waits across all 4 ring iterations, repeat with
`PROFILE_RING_ITER_N` set to 0, 1, 2, 3 in turn, copying the tracy file
out between runs (the next run overwrites it). The four `iter*.tracy`
files in `ring_sdpa_wait_zones/` were produced this way.
