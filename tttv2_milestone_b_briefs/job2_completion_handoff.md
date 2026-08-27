# Job 2 (`mb-qwen`) → `mb-coverage`: completion handoff

Written 2026-08-27 by `mb-qwen`, unattended.
Full account: `tttv2_milestone_b_evidence/qwen/REPORT.md`.
Environment and mesh facts: `tttv2_milestone_b_evidence/qwen/ENVIRONMENT.md`.
Commit produced: `768c5ca2771`.

## Read this paragraph first

**The mesh is still down, and it got worse.** `mb-llama` reported one dead board
(7). This job found **eleven** boards off the PCIe bus:

```text
missing: 0 1 2 3 4 5 6 7 10 11 14        (11 of 32)
present: 8 9 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

Both permitted recovery attempts were spent and both failed. `tt-smi -glx_reset`
cannot run because it must open `/dev/tenstorrent/7` first; `tt-smi -r` reset the
21 visible devices and then failed re-initialising with `Read 0xffffffff over
PCIe ID 17`. **Neither can recover a board that is not on the bus.** This needs
an IPMI power cycle of the tray or a host reboot — outside what an unattended
job may do.

**Check the mesh before you plan anything, and do not use the check the house
rules give you:**

```sh
ls /dev/tenstorrent | wc -l        # returned 32 the whole time. USELESS - stale nodes.
ls /sys/class/tenstorrent | wc -l  # returned 21. THIS is the real count.
tt-smi -ls                         # aborts in tt_umd topology discovery
```

`ls /dev/tenstorrent | wc -l` is the first line of the house-rules run procedure
and **it lied to both this job and the last one**. Use the `sysfs` count.

If it is still broken: say so and stop. Do not burn a night on retries — that is
now two jobs' worth of evidence that the reset paths cannot fix this.

**There is a second, independent blocker for Qwen specifically.**
**Qwen3-32B's weights are not on this machine.**

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-32B    12K, config.json ONLY
/proj_sw/user_dev/hf_data/hub/                      no Qwen3-32B at all
HF_HOME                                             unset in this job's env
```

`mb-llama`'s handoff warned about exactly this and it was right. Even with a
healthy mesh you cannot run a Qwen full model or the accuracy gate until
someone fetches ~65 GB into `/proj_sw/user_dev/hf_data`. **Llama-3.3-70B *is*
there.** Plan around that asymmetry: if you get one working night, Llama is the
model you can actually load.

## Where both models stand

Be precise about this, because it determines what `mb-coverage` can even attempt.

| | Llama-3.3-70B | Qwen3-32B |
| --- | --- | --- |
| Weights on this host | **yes** (`/proj_sw/user_dev/hf_data`) | **no** (config only) |
| Host adaptor qualified | yes (job 1, 9 tests, 3 processes) | **yes** (this job, 13 tests, 3 processes) |
| Constructs + tears down on mesh | yes, once, 109 s (job 1) | **never attempted** |
| Decode graph executes | most of one layer (job 1) | **never attempted** |
| Any PCC on silicon | **none** | **none** |
| Any accuracy number | **none** | **none** |
| Demo output | **none** | **none** |

**Neither model has a single numerical result from silicon.** Plan step 7 sits
on top of a block that has never been qualified for either model. If you are
told "both models work", that is not what the evidence says.

## Working entry points and the exact commands

### Host — these work today, mesh or no mesh

```sh
# Qwen adaptor + placement contracts (45 tests, ~20 s, no device)
python -u -m pytest -q -p no:cacheprovider \
  models/common/tests/models/qwen3_32b_galaxy/test_model_host.py \
  models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py

# Llama equivalent
python -u -m pytest -q -p no:cacheprovider \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py \
  models/common/tests/models/llama33_70b_galaxy/test_hf_conversion_host.py
```

**The regression gate in your brief is not host-only.** `models/common/tests/modules`
collects device suites; with the mesh down they *error* rather than skip (289 of
them). Use:

```sh
python -m pytest -q --ignore-glob="*_wh_galaxy*.py" \
  --ignore=models/common/tests/modules/moe/test_generalized_moe_gate.py \
  --ignore=models/common/tests/modules/moe/test_tt_moe_decode.py \
  --ignore=models/common/tests/modules/moe/test_tt_moe_gate.py \
  models/common/tests/modules \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py \
  models/common/tests/models/qwen3_32b_galaxy/test_model_host.py \
  models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py
# 410 passed, 0 failed
```

### Device — first thing to run when the mesh returns

```sh
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=900 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py
```

13 s, no checkpoint, no weights. Job 1 wrote it; it prints the real partition
and is the cheapest mesh health check available. **Neither job has been able to
run it since it was written.** It is unproven against a healthy mesh.

**There is no command that produces a passing Qwen decode or prefill.** Your
brief's template asks this handoff for one. It does not exist, and inventing one
would be worse than saying so.

## What you inherit in the tree

One new commit on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`:

```text
768c5ca2771  Qualify the Qwen Galaxy weight conversion and the 64-head geometry on host
```

### Shared code: NOT touched

**This job changed nothing under `models/common/models/galaxy/`.** Only the Qwen
package and its tests. So Llama's job-1 evidence is not invalidated by anything
here, and Llama's device gates did not need re-running (they could not have been
— no mesh).

### Two UNVERIFIED changes you are inheriting

Both are ports of fixes job 1 qualified on silicon *for Llama*; the Qwen package
had carried the defects unchanged. **Neither has been seen to run on a mesh.**
Treat them the way job 1 told this job to treat its `in0_block_w` change: as
hypotheses with a good pedigree, not as fixes.

1. **`qwen3_32b_galaxy/model.py::_relocate`** — was the three-argument
   `to_memory_config(t, memcfg, dtype)`, which reaches `ttnn::prim::copy` and
   the full compute grid. Now the worker-confined
   `sharded_to_interleaved` / `interleaved_to_sharded` pair. This was job 1's
   "single highest-value thing to do before your first device run".
2. **The embedding's decode output** — was `ttnn.L1_MEMORY_CONFIG` (interleaved,
   so `ttnn.embedding` spread over the whole grid). Now `decode.residual_memcfg`.

Pinned by three new tests in `test_model_host.py`. The embedding one was
confirmed to fail against the unfixed code.

### A host-testing capability worth knowing about

`test_model_host.py` never called `build_qwen3_32b_galaxy_transformer_2d_config`,
so module-to-module placement wiring had **zero** host coverage. It turns out
the whole transformer config builds against the `MagicMock(spec=ttnn.MeshDevice)`
that file already uses, and `resolve_galaxy_decode_placements` returns **real**
`MemoryConfig` objects on it. See `_transformer_config()` in that file.

This is cheap and it caught a real defect. A mocked mesh still cannot tell you
the partition (job 1's warning stands), but it *can* tell you whether module A's
output placement is the same object as module B's expected input placement —
which is where two of this milestone's nine defects lived.

## Qwen-specific facts, settled so you do not have to

1. **Risk 4 — fused QKV bias: RESOLVED.** `Qwen/Qwen3-32B`'s real `config.json`
   declares `"attention_bias": false`. No contract change is needed. A test now
   asserts it and checks every field of `QWEN3_32B_CHECKPOINT_CONTRACT`; it runs
   rather than skips, because the config is cached even though the weights are
   not.
2. **The residual dtype agrees with the shared all-reduce buffer.** Qwen's
   `decode_residual_dtype` is `ttnn.bfloat16`, matching `plans.py`'s default. Job
   1's D-B8 warning does **not** apply to Qwen. Nothing to pass through.
3. **The ring widths are as your brief describes, and here is why.**
   `RING_ALIGNMENT = 32 * 24 = 768`, ring shard = 160.
   Qwen `local_hidden_dim` 3200 **is** a multiple of 160 → scatters the logical
   width → **resource key 800**, placement 960. Llama 3584 is **not** → scatters
   the padded width → key 960, placement 960. Arithmetic, not a defect. Covered
   by `test_decode_ring_widths_differ_from_llama_by_exact_divisibility`. Still
   device-unverified — if a Qwen decode all-gather cannot find its resource,
   this is still the first pair to inspect.
4. **The 64-head geometry is qualified on host, not on silicon.** Attention
   rebuilt from the converted tensors alone reproduces unmodified HF
   `Qwen3Attention` at PCC ≥ 0.9999 on a fixture with the real decoupled ratio.
   The `wo` pairing `(local_attention_dim 1024, local_dim 1280)` is correct at
   `model.py:483` and pinned.
5. **The trap: `local_qkv_size == local_dim == 1280` for Qwen3-32B.** A confusion
   between the fused-QKV width and the residual width is **shape-invisible** on
   this model. `local_attention_dim` (1024) is the one that differs. If you are
   reviewing Qwen placement code, this is the pair that shape checks cannot save
   you from.
6. **Qwen's decode residual placement is 10 cores, not Llama's 16.**
   `local_dim` 1280 / 128 = 10 (grid `[2-0 - 3-4]`, shard `[32, 128]`), against
   Llama's 2048 / 128 = 16. Job 1's handoff warned that every placement number in
   its `ENVIRONMENT.md` was Llama's. Re-derived; do not carry Llama's over.

## Still open, inherited and not addressed here

Nothing in this list was touched by this job, because none of it is reachable
without a mesh.

* **D-B9** — the attention decode matmul's circular buffers clash with L1 on
  `x=1..3` by ~20 kB. Open. Job 1's `in0_block_w` `gcd(k,4)` change is in the
  tree and **device-unverified**. The structural answer is still to move the
  attention decode matmuls to the 24-core ring/`gather_in0` form the MLP already
  uses; `attention_qkv_collective_input_memcfg` is already shaped for those 24
  cores. Budget a whole night if you attempt it.
* **L1** (global-CB ownership across two constructions) — never measured. The
  80-layer model was never built. `test_two_models_in_one_process` exists and
  has never run. Job 0's O5 stands.
* **L3** — attention decode matmuls confined with `allowed_worker_cores` but
  left on `dense_matmul_program_config`, i.e. legal but on three worker columns.
  Proposed text is in job 1's `REPORT.md` §4.5.
* Everything in plan step 7 — paged KV, prefix cache, concat-32, device
  sampling, long context. That is your brief, and none of it was started here.

## Deliberate omission you should know about

**This job wrote no new device test files**, though its brief asked for them. A
device test file that has never been executed — not even collected against real
weights — would invite you to trust it. The existing
`models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py` and
`test_full_model_wh_galaxy.py` are still there, still never run.

If you want a staged bringup harness for Qwen, port job 1's
`llama33_70b_galaxy/test_bringup_wh_galaxy.py` and its `_stage` context manager.
Job 1 credits it with locating all nine of its defects with no debugger and no
second run. It prints and flushes a stage name *before* each device call, which
is the only thing that tells you which call aborted when the mesh is left
un-drainable and pytest never reaches its summary.

## Suggested order for your night

1. `ls /sys/class/tenstorrent | wc -l`. If it is not 32, report `BLOCKED (infra)`
   and stop. Do not spend recovery attempts — two jobs have now proved the reset
   paths cannot fix this class of fault.
2. If the mesh is back: `test_partition_wh_galaxy.py` (13 s). It has never run
   against a healthy mesh; treat a pass as new information.
3. Decide which model you are working on **by checking which weights exist**.
   Today that answer is Llama.
4. Expect D-B9 to block any decode step for either model. Decide up front
   whether you are verifying the `in0_block_w` hypothesis or doing the ring
   conversion.
5. Only then plan step 7. It sits on a block that is unqualified for both
   models; if you build paged KV on top of an unverified decode you will not be
   able to tell which layer is wrong.

## Do not

* Do not trust `ls /dev/tenstorrent | wc -l` as a mesh health check.
* Do not trust the `_relocate` or embedding ports until you have seen them run.
* Do not trust job 1's `in0_block_w` change until you have seen it run.
* Do not edit `models/common/modules/MILESTONE_A_STATUS.md` or
  `tttv2_2d_modules_plan.md` — job 0 and job 4 own them. This job's proposed O4
  text is in its `REPORT.md` §11.
* Do not touch `models/common/modules/**/*_1d.py` or
  `models/common/llm_runtime/**`. Both greps are empty across every Milestone B
  commit and should stay that way.
* Do not read a passing result into anything in this package that does not say
  "passed" with a log next to it. For Qwen on silicon, there are **none**.
