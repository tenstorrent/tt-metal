
## The environment, in four lines

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data     # reaches BOTH checkpoints; inherited value is empty
ls /sys/class/tenstorrent | wc -l                 # 32. NOT /dev/tenstorrent - those nodes persist after a board falls off
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=240 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py    # 12 s, opens a real cluster
```

A `SKIPPED` in a run you meant to count is a failed run, not a green one:
`hf_config_or_skip` skips silently under a wrong `HF_HOME`.

## The cost model you need to schedule against

| Thing | Wall clock |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer build, warm device weight cache | **~5.5 min** |
| Llama 80-layer build, *cold* weight set | **~26 min, and 138 GB of disk** |
| Llama teacher-forced 512/511 decode, after the build | ~18 min |

Every test in the step-7 files builds its own model and several build two, so a
17-node-id file is a three-hour run. That single fact is why attempt 2 ran
subsets and says, per row, how many fresh processes each claim got. Plan a night
around builds, not around tests.

## Two things not to spend a run on

* **`Prefetcher2DConfig.release_global_cb_on_prefill`** — `mb-llama` attempt 3
  implemented it and refuted it on hardware. Dropping the last Python reference
  to a `global_circular_buffer` does not return its L1; there is no `deallocate`
  on the type.
* **A `memory_config`-only fix for D-C1** — both the prefill and the decode page
  table are DRAM-*interleaved* on device (measured, `a2_01b`), so
  `is_sharded()` is false for both and cannot separate them by itself.
