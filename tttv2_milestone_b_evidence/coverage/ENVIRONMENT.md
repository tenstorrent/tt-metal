# `mb-coverage` — environment (attempt 2)

Recorded 2026-08-27/28, unattended, by the Milestone B step-7 coverage job,
**attempt 2**. Attempt 1's version of this file is preserved verbatim beside it
as `ENVIRONMENT_attempt1.md`; where the two disagree, this one is later and was
measured on a live mesh.

## The correction that matters

Attempt 1 recorded, as its headline, *"the mesh never came back … eleven boards
off the PCIe bus … `ttnn` cannot open a cluster at all"*. **That is not true of
this machine at 2026-08-27 23:21 UTC.** Established before any planning, by the
cheapest real check there is:

```sh
$ ls /sys/class/tenstorrent | wc -l
32
$ python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=240 \
    models/common/tests/models/galaxy/test_partition_wh_galaxy.py
5 passed in 12.32s          # opens a real 8x4 cluster; "multidevice with 32 devices is created"
```

Log: `logs2/a2_00_mesh_health.log`. Every device number in this attempt's
`REPORT.md` was measured after that check.

The mesh was repaired between attempt 1 (03:31 UTC) and `mb-qwen` attempt 2
(which ran 17:53–22:51 UTC and qualified both models on silicon). Attempt 1 was
not wrong about what it saw; it was superseded.

## Tree

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit at start | `b1e824537a4699003413dfa863db9fa3bb6253ad` (`Re-qualify the step-5 gate and the Q/K norm at this attempt's final commit`) |
| Milestone A base for the boundary greps | `bc6ad03bfc2` |
| Milestone A reference branch | `gongyu/tttv2_wh_glx_2d_modules` — read only, never written |

Attempt 1's two commits are both ancestors of this commit
(`git merge-base --is-ancestor 1cd451cd965 HEAD` → true), so every step-7 test
file it wrote is present in the tree this attempt measured.

## Host and mesh

| | |
| --- | --- |
| Kernel | `6.8.0-83-generic` |
| RAM | 566 GiB total, ~273 GiB available at start |
| `ls /sys/class/tenstorrent \| wc -l` | **32** (the health check; `/dev/tenstorrent` is not one) |
| Mesh | WH Galaxy, `(8, 4)`, exclusive to this job |
| Python env | pre-built `python_env/`, **not** recreated; no rebuild of tt-metal |

## The one environment variable that silently ruins a night

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data      # reaches BOTH checkpoints
```

Inherited value in this job's shell: **empty**. Under an empty or wrong value
`hf_config_or_skip` turns every real-checkpoint test into a `SKIPPED` and the run
looks green having measured nothing. Every script in this directory exports it.
Confirmed present under it: `models--meta-llama--Llama-3.3-70B-Instruct` and
`models--Qwen--Qwen3-32B`.

## Reference token files

`models/tt_transformers/tests/reference_outputs/{Llama-3.3-70B-Instruct,Qwen3-32B}.refpt`

Both hold **1024** tokens (`reference_tokens` shaped `(1, 1024)`, `top5_tokens`
`(1024, 5)` and `(1023, 5)`). This is a hard limit on two things measured here:
the 512/511 accuracy gate fits exactly, and `_distinct_rows(length, 32)` — which
needs `length + 32` tokens — cannot produce 32 distinct rows at length 1024 or
2048 from this file. See `REPORT.md` §A2 area 2.

## Run harness

`cov_run3.sh` → `cov_device_run.sh` → `cov_after_device_run.sh`, driven by
`cov_seq2.sh` over a pipe-delimited manifest. Copied from
`../qwen/`, with one deliberate change, recorded because it silently destroys
multi-test runs:

> the qwen `device_run.sh` arms its teardown-grace timer on the **first per-test
> verdict** in the log. In a one-node-id run that is the last thing before
> teardown; in a whole-file run it fires 90 s after test 1 passes and reaps a
> healthy process mid-test-2. `cov_device_run.sh` arms on the pytest **session
> summary** instead, and adds an idle-mtime trigger (no log write for
> `MB_IDLE`=900 s with a verdict already in hand) for the hang case attempt 3 of
> `mb-llama` described.

Never pipes pytest. Signals only a PID whose `comm` is python/python3/pytest.
Resets the mesh (`tt-smi -glx_reset`, 900 s cap) after any non-clean run.

## Costs measured here, for whoever schedules the next night

| Thing | Wall clock |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer model build from the warm device weight cache | **~5.5 min** |
| Qwen 64-layer model build from the warm device weight cache | ~4 min |

Every test in these files builds its own model, and several build two. That
single number is what makes a 17-node-id file a three-hour run and is the reason
this attempt ran subsets rather than whole files where it had to choose.
