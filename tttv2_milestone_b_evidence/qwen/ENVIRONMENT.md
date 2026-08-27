# `mb-qwen` attempt 2 — environment

Written 2026-08-27 during the run. Every fact here was read off this machine on
the day, not carried forward from a previous attempt's document.

## Mesh

```text
ls /sys/class/tenstorrent | wc -l     32      <- the authoritative count
ls /dev/tenstorrent      | wc -l      32
tt-smi -ls                            32 Wormhole boards, tt-galaxy-*
host                                  wh-glx6u-06-special-ctr-apbernal-for-reservation-118042
kernel                                Linux 6.8.0-83-generic #83~22.04.1-Ubuntu
```

**The mesh is healthy.** `mb-qwen` attempt 1 reported eleven boards off the PCIe
bus and spent its night on host work; that was true when it was written and is
not true now. `mb-llama` attempt 3 ran a full night of device work on this mesh
and met its finish condition, and this attempt opened a `(8, 4)` cluster on its
first try:

```text
models/common/tests/models/galaxy/test_partition_wh_galaxy.py   5 passed in 12.93s
    tttv2_milestone_b_evidence/qwen/logs2/a2_01_partition.log
```

Read the count from `sysfs`, never from `/dev/tenstorrent`: the device nodes
persist after a board falls off the bus, which is what misled two earlier jobs.

## Checkpoints

```text
HF_HOME=/localdev/ctr-apbernal/hf_data
  hub/models--Qwen--Qwen3-32B                    17/17 shards, revision 9216db5781bf…
  hub/models--meta-llama--Llama-3.3-70B-Instruct symlink farm into /proj_sw
```

**The `HF_HOME` this job inherited from its environment was wrong.** The driver
runs with

```text
HF_HOME=/localdev/ctr-apbernal/hf_data/hub/          <- inherited, WRONG
```

which makes the Hugging Face cache `/localdev/ctr-apbernal/hf_data/hub/hub`, a
directory holding only `models--mistralai--Mistral-7B-Instruct-v0.3`. Under it
`AutoConfig.from_pretrained("Qwen/Qwen3-32B")` raises and `hf_config_or_skip`
**skips**, so a run would look green having measured nothing. Every script under
this directory exports the correct value explicitly:

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data      # note: no trailing /hub
```

Verified by resolution, offline:

```text
HF_HOME=/localdev/ctr-apbernal/hf_data   Qwen/Qwen3-32B -> qwen3, hidden 5120,
                                         64 heads, 8 kv heads, head_dim 128,
                                         64 layers, vocab 151936, inter 25600,
                                         attention_bias False
```

`/proj_sw/user_dev/hf_data` — the path every Llama harness script hardcodes —
reaches Llama only. The Llama `ENVIRONMENT.md` sentence "either path reaches the
same shards" is true for Llama and false for Qwen.

## Python

```text
python_env/bin/activate       (pre-built; not rebuilt or recreated by this job)
Python                        3.10.21
torch                         2.11.0+cpu
transformers                  5.12.1
build                         build_Release, reused
```

## Repository

```text
/proj_sw/user_dev/ctr-apbernal/tt-metal
branch   apbernal/tttv2_wh_glx_2d_modules_milestone_b
base     690737450a8  (mb-llama attempt 3's final commit)
```

## Geometry, read off the mesh rather than assumed

`logs2/a2_01_geometry.log`, `test_..._geometry_is_decoupled_8x4_qwen3_32b`:

```text
dim=5120  n_heads=64  head_dim=128  attention_dim=8192   (1.60 x dim)
local_dim=1280  local_attention_dim=1024  local_qkv_size=1280  local_hidden_dim=3200
wo is [8192, 5120]; per mesh row [1024, 1280]
wo DRAM shard  (local_attention_dim) : 12 cores, shape [1024, 128]
wo DRAM shard  if dim were used      : 12 cores, shape [1280, 128]
padded vocabulary 153600 (19200/device)
```

`local_qkv_size == local_dim == 1280` for this model. A confusion between the
fused-QKV width and the residual width is therefore **shape-invisible** here;
`local_attention_dim` (1024) is the only one of the three that differs, and it is
the one `wo`'s placement must be built from.

## Harness

Copied from `tttv2_milestone_b_evidence/llama/` with `HF_HOME` corrected and the
log directory renamed:

```text
run3_sequence.sh <manifest>   serial: one pytest on the mesh at a time
run3.sh <name> <node> [args]  one cycle: run, reap, tt-smi -glx_reset
device_run.sh                 never pipes pytest; reaps only the PID it started
after_device_run.sh           reset after any non-clean run
ensure_mesh_free.sh           only signals a python holding /dev/tenstorrent
host_gate.sh                  the Llama host selection (host only)
```

`MB_DEADLINE` bounds the whole cycle; `MB_PYTEST_TIMEOUT` bounds pytest.

### One harness limitation this attempt measured

`device_run.sh` arms its teardown grace timer on the first `PASSED|FAILED|ERROR`
in the log. A decode-mode `TT_FATAL` in `run a2_03_qknorm` produced **no such
line at all**: pytest holds the `-v` verdict until the test's teardown phase
completes, and teardown is exactly what hangs. The log's last write was at
15:36 and the process still held all 64 `/dev/tenstorrent` fds seven minutes
later. Only the wrapper's full `MB_DEADLINE` would have reaped it. Budget for
that, or watch the log's mtime rather than its contents.

## Addendum — the exact invocations behind every published number

Every device run was `run3_sequence.sh <manifest>`; the manifests are
`seq_a2_*.txt` in this directory and each line is
`<wrapper-deadline> <pytest-deadline> <logname> <node-id>`. The wrapper expands to

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
python -u -m pytest -v -rA --color=no -p no:cacheprovider \
  --timeout=<pytest-deadline> <node-id> -o faulthandler_timeout=600 >> <log> 2>&1
```

with `tt-smi -glx_reset` after any non-clean run. `TTTV2_GALAXY_CCL_TRACE` was
1 for `seq_a2_01` … `seq_a2_08` (which is where the `[ccl]` shard-exactness lines
come from) and 0 from `seq_a2_09` on, because it adds three device synchronizes
per token and the accuracy gate is 511 tokens.

| gate | node id |
| --- | --- |
| step-5 block | `models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py::test_qwen3_32b_galaxy_one_layer_prefill_and_decode_8x4_qwen3_32b_b32_s128` |
| prefill 2048 | `…::test_qwen3_32b_galaxy_one_layer_prefill_2048_8x4_qwen3_32b_b1_s2048` |
| Q/K norm alone | `…::test_qwen3_32b_galaxy_qk_norm_head_local_8x4_qwen3_32b_decode_and_prefill` |
| geometry | `…::test_qwen3_32b_galaxy_geometry_is_decoupled_8x4_qwen3_32b` |
| decode bisection | `…::test_qwen3_32b_galaxy_decode_bisection_8x4_qwen3_32b_b32_s128` |
| full model | `models/common/tests/models/qwen3_32b_galaxy/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_full_model_prefill_and_first_decode_token` |
| accuracy gate | `…::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1` |
| demo batch 1 | `models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch1` |
| demo batch 32 | `…::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination` |

## Costs, measured on this machine

| cost | measured |
| --- | --- |
| mesh open + `test_partition_wh_galaxy.py` | 13 s test, ~55 s cycle |
| one-layer reference load (3 of 17 shards) | seconds |
| one step-5 block cycle (test + reap + reset) | **~3 min**, test itself 108-115 s |
| prefill 2048 cycle | ~3.5 min, test 178 s |
| decode bisection cycle | ~2.5 min |
| staging 64 layers to device, **first time in a tree** | one **10 min** run (108 GB written under `model_cache/Qwen`) |
| the same staging, every process after | cache hit; full-model test 128 s |
| accuracy gate, 511 eager decode steps, CCL trace **off** | **~13 min** |
| Llama accuracy gate, same conditions | ~21.5 min |
| demo batch 1 / batch 32 | 143 s / 169 s |

Qwen is 32B against Llama's 70B and every cycle is roughly half. The whole Qwen
gate set - block x3, prefill 2048 x3, bisection x3, full model x3, accuracy x3,
both demos x3 - is about **two hours** once the device weight cache is warm.

## Disk

```text
model_cache/Qwen        108 G
model_cache/meta-llama  139 G
/proj_sw free after both  196 G
```
