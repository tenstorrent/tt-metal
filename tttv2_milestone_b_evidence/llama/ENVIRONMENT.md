# `mb-llama` — environment

Everything `mb-coverage` needs to make a paired comparison against this job's
numbers, and everything `mb-qwen` needs to reproduce its setup.

## Tree

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit at job start | `b350e51554470414d5a8b08f5ea9775c986145a4` |
| Base this job built on | `52def65194c3938ed6e5cb6f52661ec3a3a15547` (last code-bearing commit before this job) |
| Milestone A reference tip | `bc6ad03bfc21d6a26f88169cc87ea2d8176f0fbf` (read-only) |

Commits added by this job are listed in `REPORT.md` §"Commits".

## Host

```text
Linux 6.8.0-83-generic
Python 3.10.21   (python_env/, pre-existing; not rebuilt)
torch 2.11.0+cpu
transformers 5.12.1
pytest 9.0.3, pytest-timeout 2.4.0
566 GB RAM total, ~309 GB available at job start
```

`build_Release/` was reused, not rebuilt:

```text
CMAKE_BUILD_TYPE   = Release
CMAKE_CXX_COMPILER = /usr/bin/clang++-20
CMAKE_CXX_FLAGS    = (empty)
ENABLE_DISTRIBUTED = ON
ENABLE_ASAN/DEBUG/DEBUG_LOG/COVERAGE/MEMORY_DEBUG/ALLOC_DEBUG = OFF
ENABLE_CCACHE      = TRUE
```

Environment variables that matter:

```sh
TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
PYTHONPATH=/proj_sw/user_dev/ctr-apbernal/tt-metal
TT_DEVICE_LOCK_PATH=/tmp/tt_device.lock
TT_DEVICE_LOCK_TIMEOUT=3600
VIRTUAL_ENV=/proj_sw/user_dev/ctr-apbernal/tt-metal/python_env
```

## Mesh

```text
32 boards, Wormhole, tt-galaxy, board series 010003510...
/dev/tenstorrent : 32 entries
KMD version      : 2.4.1
IOMMU            : enabled
mesh shape       : (8, 4), all 32 devices
device_params    : dispatch_core_axis = COL, fabric_config = FABRIC_1D_RING
compute_with_storage_grid_size : x=7, y=10   (measured on device, not assumed)
AI clock at sync : ~0.9855 GHz
```

Full `tt-smi -ls` output: `logs/05_tt_smi_ls_baseline.log`.

### The decode sub-device partition, as measured

This is the single most load-bearing fact of the whole job, and it is not
derivable from a mocked mesh. Measured by
`models/common/tests/models/galaxy/test_partition_wh_galaxy.py`
(`logs/39_partition_probe_run4.log`):

```text
compute grid            x=0..6, y=0..9          (70 cores)
worker_cores()          {[1-0 - 3-9], [5-0 - 6-9]}   50 cores
prefetch senders        x=0 and x=4, 12 cores
in NO sub-device        {[0-1 - 0-3], [0-6 - 0-8], [4-3 - 4-3], [4-8 - 4-8]}   8 cores
```

Two consequences that cost this job most of its device time:

1. The worker envelope is **not contiguous** — the `x=4` sender column splits it
   — so its *bounding box* (`x=1..6`) is not a safe stand-in for it. Several ttnn
   ops use the bounding box.
2. The sender ∪ worker union does **not** cover the compute grid, so a program
   built over the full grid touches cores owned by no sub-device.

## Checkpoint

```text
meta-llama/Llama-3.3-70B-Instruct
snapshot 6f6073b423013f6a7d4d9f39144961bfbfbc386b
30 safetensors shards, 141.1 GB of tensors
```

**It is not in the default HF cache.** `~/.cache/huggingface/hub` holds
config-only entries and no Llama at all. The real checkpoint is in the shared
cache, so every run in this job exported:

```sh
export HF_HOME=/proj_sw/user_dev/hf_data
```

Without it, `from_pretrained` tries to download into `/home/ctr-apbernal`, which
has a 9.4 GB quota.

Config, verified against `LLAMA33_70B_CHECKPOINT_CONTRACT`:

```text
hidden_size 8192   num_attention_heads 64   num_key_value_heads 8   head_dim 128
intermediate_size 28672   vocab_size 128256   num_hidden_layers 80
rms_norm_eps 1e-05   rope_theta 500000.0
rope_scaling: llama3, factor 8.0, low 1.0, high 4.0, original_max_position 8192
attention_bias false   tie_word_embeddings false   torch_dtype bfloat16
```

Derived: `padded_vocab_size 128256`, `rope_table_len 8192`,
`local_dim 2048`, `local_qkv_size 1280`, `local_hidden_dim 3584`,
`local_attention_dim 1024`, `users_per_column 8`.

### Loading only what is needed

`models/common/tests/models/galaxy/galaxy_checkpoint.py` reads only the
safetensors shards that hold the requested layers plus the embedding, final norm
and LM head. For layer 0 that is 3 shards of 30, ~12 GB, ~12 GB peak RSS, about
two minutes cold and well under one warm — against 141 GB of I/O and allocation
for `from_pretrained`-then-truncate.

Its tensors were verified **bitwise equal** to the shards
(`q_proj`, `o_proj`, `down_proj`, `input_layernorm`, `embed_tokens`, `norm`,
`lm_head`), and the rotary module is built from the checkpoint's own config, so
Llama 3 scaling is the real one. It is the real checkpoint's layer 0, not a
synthetic stand-in.

Storage throughput measured on the shared cache: **~460 MB/s**.

## Exact invocations

Every device run went through `device_run.sh`, one node id per process, never
piped:

```sh
export HF_HOME=/proj_sw/user_dev/hf_data
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=900 <NODE_ID>
```

Host regression gate — `host_gate.sh`:

```sh
python -m pytest -q -rA --color=no -p no:cacheprovider \
  --ignore-glob="*_wh_galaxy*.py" \
  models/common/tests/modules/attention/test_attention_1d_arch_config.py \
  models/common/tests/modules/attention/test_attention_2d.py \
  models/common/tests/modules/embedding/test_embedding_2d.py \
  models/common/tests/modules/lm_head/test_lm_head_2d.py \
  models/common/tests/modules/mlp/test_mlp_1d_arch_config.py \
  models/common/tests/modules/mlp/test_mlp_2d.py \
  models/common/tests/modules/prefetcher/test_prefetcher_2d.py \
  models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py \
  models/common/tests/modules/rope/test_rope_2d.py \
  models/common/tests/modules/sampling/test_sampling_1d_release.py \
  models/common/tests/modules/sampling/test_sampling_2d.py \
  models/common/tests/modules/test_tensor_utils.py \
  models/common/tests/models/galaxy \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py
```

Baseline at job start: **395 passed**. At the end of this job: **398 passed**
(three tests added). No test was deleted, `xfail`ed, skipped or relaxed.

### Three things about pytest here that are easy to lose a run to

1. **`pytest.ini` sets `timeout = 300` globally.** Any device test that loads a
   checkpoint blows through it and dies looking exactly like a hang. Every device
   run in this job passes `--timeout=900` explicitly.
2. **`pytest.ini` `addopts` already contains `-vvs`**, so output is uncaptured,
   but Python still block-buffers stdout to a file. Device runs use `python -u`;
   without it a `TT_FATAL` traceback is lost when the process is killed.
3. **Two of the brief's own gate paths take the mesh** — job 0 recorded this.
   `models/common/tests/models/galaxy` collects
   `test_column_user_selector_wh_galaxy.py`, and `test_plans.py` opens the UMD
   driver for all 32 chips when run as a whole file. So the host gate must never
   run while a device session is live, and it uses `--ignore-glob`.

## Scripts in this directory

| Script | What it is for |
| --- | --- |
| `host_gate.sh` | the host regression selection above |
| `device_run.sh` | one device run; reaps **its own child by PID** on timeout |
| `ensure_mesh_free.sh` | reap processes that hold `/dev/tenstorrent` — and only those |
| `after_device_run.sh` | reap, then `tt-smi -glx_reset` after any non-clean run |
| `cycle.sh` | `device_run` + `after_device_run`; always run it in the background |

Three harness bugs were hit and fixed while building these, all worth inheriting:

* a reaper matching `pgrep -f pytest` killed the **next** run's pytest a second
  after it started (empty log, exit 137). Reap your own child by PID.
* the same reaper killed a concurrent **host-only** gate. Only reap a process
  that has `/dev/tenstorrent` open.
* `tt-smi -glx_reset` fails with `[Errno 19] No such device` if a holder is still
  alive. Kill the holder first, then reset.

## Infrastructure instability worth reporting

Recurring, roughly every other device run in the second half of the session:

```text
RuntimeError: Timed out waiting for ETH heartbeat on device
ASIC ID: 87032054158471220, ETH core e9-0 (NOC0) to advance. Stuck at 0xabcd....
  tt::umd::TopologyDiscovery::eth_heartbeat_running
  tt::umd::TopologyDiscovery::discover_remote_devices
```

Always the **same ASIC**, `87032054158471220`, on ETH core `e9-0` or `e8-0`, and
always at mesh open during topology discovery. A `tt-smi -glx_reset` clears it;
it then returns after the next aborted run. Once it hung topology discovery
outright rather than erroring (`logs/70_decode_step_run21.log`).

It correlates with a preceding `TT_FATAL` inside a multi-sub-device program,
which leaves the mesh un-drainable, so the working theory is a dirty fabric that
the reset does not always fully restore on that one board. Mitigation adopted:
reset after *every* non-clean run, and health-check with the 13-second partition
probe before spending a run that loads a checkpoint.

Logs: `48_`, `52_`, `56_`, `61_glx_reset_eth*.log`, and the ETH failures in
`47_`, `51_`, `55_`, `68_`, `70_`.

---

# Addendum — attempt 2 (2026-08-27)

Attempt 1 ended `BLOCKED (infra)`. Attempt 2 ran on the same tree, on a mesh
that had been recovered in the meantime.

## Tree

| | |
| --- | --- |
| Commit at attempt-2 start | `6a3e78a7227cbb22e8fa789adae2b91e3aeb0bdf` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` (unchanged) |

Note that `6a3e78a` is *after* `mb-qwen`, `mb-coverage` and `mb-signoff` ran, so
this attempt inherits their test files
(`test_full_model_wh_galaxy.py`, `test_step7_coverage_wh_galaxy.py`,
`models/common/tests/models/galaxy/test_step7_*.py`) and their commits. None of
those files had ever executed either.

## Mesh, re-measured at attempt-2 start

```text
ls /dev/tenstorrent | wc -l   -> 32
tt-smi -ls                    -> exit 0, 32 Wormhole tt-galaxy boards,
                                 including board 7 at 0000:08:0x
```

Attempt 1 recorded `tt-smi -ls` aborting inside `tt_umd` and listing zero
boards, because `/dev/tenstorrent/7` was unreadable. It now enumerates. The
recovery was an out-of-band power cycle, not anything either attempt did.

The partition is unchanged from attempt 1 — re-measured on device, not carried
over (`logs2/a2_00_partition.log`):

```text
compute_with_storage_grid_size: x=7 y=10
worker_cores:      {[1-0 - 3-9], [5-0 - 6-9]}  (50 cores)
prefetch senders:  x=0 and x=4                 (12 cores)
in no sub-device:  {[0-1 - 0-3], [0-6 - 0-8], [4-3], [4-8]}  (8 cores)

decode-qkv  allowed_worker_cores {[1-0 - 3-0]}  blocks_x=3 blocks_y=1
decode-wo   allowed_worker_cores {[1-0 - 3-0]}  blocks_x=3 blocks_y=1
prefill-128-qkv  {[1-0 - 3-3]}  blocks_x=3 blocks_y=4
prefill-2048-wo  {[1-0 - 3-3]}  blocks_x=3 blocks_y=4
```

## Host

Unchanged from attempt 1, except:

```text
566 GB RAM total, ~402 GB available at attempt-2 start
64 cores
mesh_device.dram_grid_size().x = 12   (the LM head weight's DRAM shard count)
```

## Checkpoint resolution — read this before running anything

`HF_HOME` was set in the inherited shell to
`/localdev/ctr-apbernal/hf_data/hub/`, which is **wrong as an `HF_HOME`**:
`transformers` looks for `$HF_HOME/hub`, so that path resolves to
`/localdev/ctr-apbernal/hf_data/hub/hub`, which does not exist, and every
checkpoint test *skips*. This is the same failure mode commit `0c1ccd8557c`
recorded.

Both of these work, and the first is what the harness scripts export:

```sh
export HF_HOME=/proj_sw/user_dev/hf_data          # 368 GB, holds the real blobs
export HF_HOME=/localdev/ctr-apbernal/hf_data     # symlink farm into the above
```

`/localdev/ctr-apbernal/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct`
is 4 KB of symlinks whose blobs live under `/proj_sw/user_dev/hf_data`; either
`HF_HOME` reaches the same 30 safetensors shards.

## Exact invocations

Device runs all go through the attempt-1 harness, unchanged:

```sh
export HF_HOME=/proj_sw/user_dev/hf_data
MB_DEADLINE=<seconds> bash tttv2_milestone_b_evidence/llama/cycle.sh \
    tttv2_milestone_b_evidence/llama/logs2/<name>.log <FILE-OR-NODEID>
```

which is `device_run.sh` (one un-piped `python -u -m pytest -v -rA --color=no
-p no:cacheprovider --timeout=900`, reaped by PID) followed by
`after_device_run.sh` (reap any device holder, then `tt-smi -glx_reset` after
any non-clean exit).

Host regression gate:

```sh
bash tttv2_milestone_b_evidence/llama/host_gate.sh <log>
```

**`host_gate.sh` takes the mesh.** It holds 64 `/dev/tenstorrent` file
descriptors — `models/common/tests/models/galaxy/test_recipes.py` and
`models/common/tests/modules/lm_head/test_lm_head_2d.py` both open a device — so
it must never run concurrently with a device cycle. Attempt 1's report said this
about `test_plans.py`; it is true of more of the "host" gate than that.

Also: do not pass `models/common/tests/modules/lm_head` as a *directory*. That
collects `test_lm_head_1d.py`, a real 8-device suite that walks a dozen
checkpoints and runs for well over ten minutes. `host_gate.sh` names the
`test_lm_head_2d.py` file for exactly this reason.

---

# Addendum — attempt 3, 2026-08-27

Re-verified rather than inherited, as the house rules require.

| Item | Value | Where it was read |
| --- | --- | --- |
| Commit at start | `45efb7c10e8349d1093f6c1a17dbc6d8ac2b65e6` | `git rev-parse HEAD`, echoed into every `logs3/*.log` header |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` | driver log |
| Boards | 32 Wormhole, no `ARC startup error` | `tttv2_milestone_b_runs/20260827T112601Z/mb-llama_tt_smi_before.log` (driver probe, 11:26 UTC) |
| `/dev/tenstorrent` | 32 | every `logs3/*.log` header |
| Firmware bundle | 18.12.1 | `logs3/a3_02_step2_gate.log`, UMD `topology_discovery.cpp:575` |
| KMD | 2.4.1 | same log, `cluster.cpp:150` |
| IOMMU | enabled | same log, `cluster.cpp:147` |
| Host RAM | 566 GB total | `free -g` |
| Python | 3.10.21, pytest 9.0.3 | pytest header in every log |
| Build | reused `build_Release/` and `python_env/`; nothing rebuilt | brief's fixed parameters |
| Mesh state at start | reset clean before the first run | `logs3/a3_00_reset.log`, `Re-initialized 32 boards after reset` |

## Attempt 3's invocations

Every device run went through `run3.sh`, which is `cycle.sh` with the CCL trace
exported, a settable pytest-level timeout and its reset logs in `logs3/`:

```sh
MB_DEADLINE=<wrapper deadline> MB_PYTEST_TIMEOUT=<pytest deadline> \
  ./tttv2_milestone_b_evidence/llama/run3.sh <logname> <node-id> \
      -o faulthandler_timeout=600
```

Two changes to attempt 2's scripts, both backward compatible:

* `device_run.sh` takes `MB_PYTEST_TIMEOUT` (default 900). A prefill 2048 or an
  80-layer load legitimately needs longer, and a pytest timeout that fires early
  costs a whole run.
* `after_device_run.sh` takes `MB_RESET_DIR` so attempt 3's reset logs do not
  land in attempt 1's `logs/`.

`-o faulthandler_timeout=600` is new and is diagnostic only — nothing committed
depends on it. Attempt 2 needed `gdb -p` to find out where a hang was, because a
device-side CCL hang leaves no Python traceback. pytest's own faulthandler plugin
dumps every thread's Python stack from a watchdog thread after 600 s and then
lets the run continue, which names the hanging Python line for free.

## Attempt 3 — the exact invocations behind every number

Every device run went through `run3.sh`, or `run3_sequence.sh` for a serial batch.
`HF_HOME=/proj_sw/user_dev/hf_data` and `TTTV2_GALAXY_CCL_TRACE=1` are exported by
`run3.sh` itself.

```sh
# the step-2 gate: prefill 128 + decode at batch 32, logits and both caches
MB_DEADLINE=1200 MB_PYTEST_TIMEOUT=1080 \
  ./tttv2_milestone_b_evidence/llama/run3.sh a3_32_step2_gate_run1 \
  'models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_and_decode' \
  -o faulthandler_timeout=600
# ... repeated as a3_33_step2_gate_run2 and a3_34_step2_gate_run3

# single-row prefill at the full 2048 recipe
MB_DEADLINE=1800 MB_PYTEST_TIMEOUT=1680 \
  ./tttv2_milestone_b_evidence/llama/run3.sh a3_35_prefill2048_run1 \
  'models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_2048' \
  -o faulthandler_timeout=600
# ... repeated as a3_36 and a3_37

# the sub-module bisection (diagnostic; reports each boundary, asserts on logits)
MB_DEADLINE=900 MB_PYTEST_TIMEOUT=780 \
  ./tttv2_milestone_b_evidence/llama/run3.sh a3_31_bisect_fusedqk \
  'models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py::test_llama33_70b_galaxy_decode_bisection' \
  -o faulthandler_timeout=600

# the one-layer runner/paged-KV smoke
LLAMA33_70B_GALAXY_DEMO_LAYERS=1 LLAMA33_70B_GALAXY_DEMO_TOKENS=8 \
MB_DEADLINE=900 MB_PYTEST_TIMEOUT=780 \
  ./tttv2_milestone_b_evidence/llama/run3.sh a3_42_demo_1layer_smoke \
  'models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch1' \
  -o faulthandler_timeout=600

# step 3, as a serial manifest (see logs3/a3_seq_step3_driver.out)
./tttv2_milestone_b_evidence/llama/run3_sequence.sh /tmp/mb_seq_step3.txt
#   3600 3400 a3_43_full_prefill_first_token  ...::test_llama33_70b_galaxy_full_model_prefill_and_first_decode_token
#   5400 5200 a3_44_accuracy_gate_run1        ...::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
#   3600 3400 a3_45_demo_batch1_80layer       demo.py::test_llama33_70b_galaxy_direct_demo_batch1
#   5400 5200 a3_46_batch32_isolation         ...::test_llama33_70b_galaxy_batch32_slots_are_isolated
```

Host gates:

```sh
python -m pytest -q -rN --color=no -p no:cacheprovider \
  models/common/tests/modules/sampling/test_sampling_2d.py \
  models/common/tests/modules/lm_head/test_lm_head_2d.py \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py     # 77 passed
```

### Two measured costs, for whoever plans the next night

| Cost | Measured |
| --- | --- |
| `AutoModelForCausalLM.from_pretrained` on the 141 GB 80-layer checkpoint | **~60 s** (723 tensors, safetensors) |
| Staging 80 layers of weights to device, **first time in a tree** | ~2.3 s per tensor of `LazyWeight` cache generation, ~400 tensors, **~15 min** |
| The same staging, **second process onward** | `[cache hit]` at ~0.05 s per tensor, well under a minute |
| One step-2 gate cycle (test + reap + `glx_reset`) | ~7 min, of which the test is ~2.5 min |
| One prefill-2048 cycle | ~9 min, of which the test is 2.7-4.1 min |

Attempt 2 expected the checkpoint load to dominate a step-3 night. It does not -
the **first** device staging does, and it is a one-off. Order the runs so the
cheapest 80-layer test pays for the cache.
