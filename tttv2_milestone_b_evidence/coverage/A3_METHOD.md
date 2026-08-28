
## Method, and what a "run" costs here

One serial device queue (`cov_queue.sh`), one pytest process at a time, never
piped. Each item is **one node id in its own fresh process** — not a choice of
style but a hard requirement of this stack, per finding D-C3: the device weight
cache is fingerprinted with `MeshDevice.id()`, so the second test in a pytest
process misses on all 965 Llama weights and pays 26 minutes and 138 GB to
re-stage them.

Between items the harness reaps any process still holding `/dev/tenstorrent`
(`cov_ensure_mesh_free.sh`, comm restricted to `python`/`python3`/`pytest` and
gated on the fd actually being open) and runs `tt-smi -glx_reset` after any
non-clean exit (`cov_after_device_run.sh`, 900 s cap — 600 s was measured too
tight for a wedged ARC controller).

Measured wall clock at this tree, warm disk cache:

| | |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer build, warm | ~5.5 min |
| Qwen 64-layer build, warm | ~2 min |
| Llama teacher-forced 512/511, whole process | ~18 min |
| Qwen teacher-forced 512, whole process | ~16 min |
| a `tt-smi -glx_reset` after a failure | ~2–4 min |

Two consequences that shaped the night's ordering:

1. a Qwen case costs roughly a third of the Llama case with the same shape, so
   Qwen coverage was run first — it buys more distinct claims per hour;
2. an item that fails costs its own wall clock *plus* a reset, so a block of
   expected failures is more expensive than a block of expected passes.
