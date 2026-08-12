# rms_norm perf experiments — moved

The Perf-1 tournament's isolated micro-benchmarks live at

    tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/<idea_slug>/

not here. Reason: every bench is a torch-referenced harness, and
`scripts/validate_no_global_torch_imports.py` (a pre-commit hook) forbids a global
`import torch` anywhere under `ttnn/ttnn/`. The benches belong in the tests tree;
this file is the pointer from the op that owns them.

One dir per floated idea, each self-contained (kernels + host + measured results in
its docstrings):

| dir | idea | verdict |
|---|---|---|
| `colvalid_payload`    | I2 column-0-valid gather / mcast payload            | WIN — graduated |
| `cskip_finalize`      | I3 even-parity SFPU scope for the finalize          | WIN — graduated |
| `scaler_offpath`      | I4 reduce constants off the reader critical path    | WIN — graduated |
| `worksplit_retune`    | I7 combine-tree-level admissibility in the work split | WIN — graduated |
| `allgather_combine`   | I1 mcast all-gather combine                         | REGRESSION (5.3x) |
| `apply_fusion`        | I5 fuse the two apply passes, delete cb_normed      | REGRESSION (1.13x) |
| `sumsq_reduce_merge`  | I6 merge sumsq + reduce_stat in DEST                | REGRESSION (0.94x) |

See `changelog.md`'s `## Perf 1` entry for the measured numbers and the round-2 leads
the three nulls produced.
