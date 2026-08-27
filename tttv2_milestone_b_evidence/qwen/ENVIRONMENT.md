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
