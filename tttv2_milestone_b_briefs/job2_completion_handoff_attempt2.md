# Job 2 (`mb-qwen`) attempt 2 → `mb-coverage`: completion handoff

Written 2026-08-27 by `mb-qwen` attempt 2, unattended, on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Full account: `tttv2_milestone_b_evidence/qwen/REPORT.md` §A2.
Run-by-run: `tttv2_milestone_b_evidence/qwen/ATTEMPT2.md`.
Machine and mesh facts, costs, exact invocations:
`tttv2_milestone_b_evidence/qwen/ENVIRONMENT.md`.

**Do not read `job2_completion_handoff.md` (attempt 1) except for its Qwen
architecture notes.** Its two headline claims — that the mesh is dead and that
Qwen3-32B is not on this machine — are both false, and both were false when the
correction banner was added to it. This file is later.

## Read this paragraph first

**Both models work on silicon, end to end, and both are qualified with three
fresh processes per gate and bit-identical results.** That is a different world
from the one attempt 1 handed on.

```text
                                    Llama-3.3-70B        Qwen3-32B
weights on this host                yes                  yes
host adaptor qualified              yes                  yes
one block, decode + prefill PCC     0.99975 / 0.99958    0.99936 / 0.99930
both KV caches PCC                  0.99993 / 0.99975    0.99989 / 0.99989
prefill 2048                        0.99962              0.99902
full model + first decode token     yes                  yes
teacher-forced top-1 / top-5        501/511, 511/511     498/511, 511/511
demo, batch 1 and batch 32          fluent, identical    fluent, identical
```

`mb-coverage` can build plan step 7 on either.

## The environment, and the one thing that will silently ruin a night

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data      # reaches BOTH models
```

The value this job inherits from its shell is `/localdev/ctr-apbernal/hf_data/hub`
— one directory too deep — and under it the Hugging Face cache is
`.../hf_data/hub/hub`, which holds only Mistral. `/proj_sw/user_dev/hf_data`
reaches Llama only. Under either wrong value `hf_config_or_skip` turns every
real-checkpoint test into a **`SKIPPED`**, and a run looks green having measured
nothing. This job saw it happen once. Every script under
`tttv2_milestone_b_evidence/qwen/` exports the right value; if you copy one,
check the export survived, and treat a `skipped` in a run you meant to count as
a failure of the run.

Mesh health: `ls /sys/class/tenstorrent | wc -l` must be 32.
**Not `/dev/tenstorrent`** — the device nodes persist after a board falls off the
bus, and they showed a full 32 while eleven boards were gone. The cheapest real
check is 13 seconds:

```sh
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=900 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py
```

## The commands that produce a passing decode and prefill, for each model

All of them want `HF_HOME=/localdev/ctr-apbernal/hf_data`. Wall-clock is a whole
cycle — test, reap, `tt-smi -glx_reset` — with the device weight cache warm.

```sh
Q=models/common/tests/models/qwen3_32b_galaxy
L=models/common/tests/models/llama33_70b_galaxy

# Qwen: one block, prefill 128 + decode batch 32, logits and both caches   ~3 min
$Q/test_model_wh_galaxy.py::test_qwen3_32b_galaxy_one_layer_prefill_and_decode_8x4_qwen3_32b_b32_s128
# Qwen: single-row prefill at the full 2048 recipe                       ~3.5 min
$Q/test_model_wh_galaxy.py::test_qwen3_32b_galaxy_one_layer_prefill_2048_8x4_qwen3_32b_b1_s2048
# Qwen: 64 layers, prefill + first decode token                            ~2 min
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_full_model_prefill_and_first_decode_token
# Qwen: the Milestone B accuracy gate                                     ~13 min
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# Qwen: the demo, batch 1 and batch 32                             ~2.5 / ~3 min
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch1
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination

# Llama: the same five, as job 1 left them                     ~3 / 3.5 / 14 / 21.5 / 3 min
$L/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_and_decode
$L/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_2048
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_full_model_prefill_and_first_decode_token
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch1
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
```

Drive them with `tttv2_milestone_b_evidence/qwen/run3_sequence.sh <manifest>`;
manifest lines are `<wrapper-deadline> <pytest-deadline> <logname> <node-id>`.
That harness never pipes pytest, reaps only the PID it started, refuses to signal
anything whose `comm` is not python, and resets after any non-clean run. Copy it
rather than the Llama one — the `HF_HOME` is already right.

When diagnosing, add
`$Q/test_model_wh_galaxy.py::test_qwen3_32b_galaxy_decode_bisection_8x4_qwen3_32b_b32_s128`:
it reports a PCC at every boundary inside layer 0 against HF forward hooks, in
~2.5 minutes, and every boundary is currently >= 0.9992.

## Shared code this job changed, and what it means for you

Two files outside the Qwen package. Both are declared with their reductions in
`REPORT.md` §A2.8, and **Llama's six device gates were re-run at this commit and
are bit-identical to `mb-llama` attempt 3's numbers** (`a2_40..47_llama_*`).

1. **`models/common/modules/rmsnorm/rmsnorm_2d.py`** — a new optional
   `RMSNorm2DConfig.decode_compute_cores`, and the `HEAD_LOCAL` decode path that
   uses it. Unset, behaviour is unchanged. Llama constructs no `HEAD_LOCAL` norm
   at all, so the branch is unreachable from it.
2. **`models/common/models/galaxy/recipes.py`** — `lm_head_reduce_core_count`
   now reserves `GALAXY_CCL_RESERVED_WORKER_CORES = 4`. Llama resolves 42 before
   and after; Qwen resolves 40 instead of 50. **If you add a model whose padded
   local vocabulary has a divisor in the top four of the worker envelope, this is
   the line that stops it segmentation-faulting.**

## Two defects you now inherit as fixed, and the shape of each

Both are Qwen-only, and both would have cost you a night.

**D-B26 — the per-head Q/K decode norm was unplaceable, three ways.** The
unresolved half of Milestone A's D2. Interleaved DRAM (the module's post-D2
default) spreads `ttnn.rms_norm` over the whole compute grid, which the decode
sub-device manager rejects; the created heads are HEIGHT_SHARDED, which the op
rejects outright; and naming any single sharded placement relocates Q and K onto
the same cores, which the fused QK rotary rejects (`Q and K must not overlap`).
The fix names only the *cores* the kernel may use, derives the shard shape from
the tensor, and returns each tensor to the placement it arrived in.

**The general rule worth carrying:** on this mesh a decode-mode op has three
independent placement constraints — the sub-device it may run on, the layouts the
op accepts, and the layout its *consumer* requires — and satisfying two of them
is not progress. Each of the three failures above satisfied exactly two.

**D-B27 — `all_reduce_async` takes worker cores from what is left over.** Fill
the worker envelope with the reduction's own tensors and it gets none; it warns
and then segmentation-faults the process. Llama escaped by arithmetic accident.
If you place a new collective, leave it room.

## Facts about Qwen settled on hardware, so you do not re-derive them

1. **The 64-head decoupled geometry is qualified on silicon**, prefill and
   decode. Milestone A's Qwen attention evidence was a 40-head fixture where
   `n_heads * head_dim == dim`; that is no longer the only evidence.
   `local_dim=1280, local_attention_dim=1024, local_qkv_size=1280`.
2. **`local_qkv_size == local_dim == 1280`.** A confusion between the fused-QKV
   width and the residual width is shape-invisible on this model.
   `local_attention_dim` is the one that differs.
3. **The ring-width question in the brief's risk 3 is settled and benign.** Qwen
   resolves resource key 800 and placement 960 where Llama resolves 960/960, and
   the decode all-gather finds its resource: the block gate passes three times.
4. **The padded vocabulary is 153600** (19200/device, 600 tiles), and after D-B27
   the LM head reduction runs on 40 cores at 480 columns — exactly 19200, so
   D-B19's invariant holds.
5. **`attention_bias: false`** in the pinned revision `9216db5781bf`. No contract
   change was needed and none was made.
6. **Qwen's decode residual placement is 10 cores**, not Llama's 16
   (`local_dim` 1280 / 128).

## A defect in the Llama package that this job may not fix

`models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py`'s
bisection reads `out.hidden_states[-1]` as "after layer 0". It is not: in
transformers 5.12.1 that entry is the output of the model's **final norm**. So
the bisection compares the residual stream against a normalized reference and the
final norm against the norm applied *twice*, and
`tttv2_milestone_b_evidence/llama/REPORT.md` records the resulting 0.979 and
0.990 as "the bfloat16 residual against an fp32 reference is the floor". **That
explanation is wrong**, and Llama's residual stream is very probably ~0.9995 like
Qwen's. Verified on the Qwen checkpoint: `hidden_states[1]` is PCC 1.0 against
`norm(layer0_output)` and 0.9178 against the layer output.

The fix is four lines — take the layer output from its own forward hook and the
final norm from `hidden_states[-1]` — and the corrected Qwen version is in
`_reference_decode_stages` in the Qwen file. The brief forbade this job from
modifying the Llama package.

## Still open

* **L1's remaining half — prefill after a decode.** Untouched, inherited,
  identical for both models. Production has the same property; `mb-llama`
  attempt 3 implemented and then refuted the obvious fix on hardware
  (`Prefetcher2DConfig.release_global_cb_on_prefill`; the L1 base address is
  unchanged because a `global_circular_buffer` has no `deallocate`). **Do not
  spend that run again.** The open hypothesis is to confine the prefill mode plan
  to the worker cores. No Milestone B gate needs it, but **step 7's prefix-cache
  and chunked-prefill work may**: any sequence that prefills, decodes, and then
  prefills again in one process hits it.
* **L1's global-CB ownership across two constructions**
  (`test_two_models_in_one_process`) — still never run, for either model.
* **D-B9** — the attention decode matmul CB clash. Job 1's `in0_block_w`
  `gcd(k, 4)` change has now run for Qwen's `local_qkv_size` 1280 as well as
  Llama's 2048, both without a clash. The structural follow-up - moving the
  attention decode matmuls onto the 24-core ring, which would also let them be
  prefetched again - is unstarted and is a performance item.
* **The head-local norm's four relocations per call** — a decode-latency item.
  The clean fix is a ttnn-level way to pass a `core_range_set` to `ttnn.rms_norm`
  for an interleaved input; the *program factory* already accepts one
  (`layernorm_op_multi_core.cpp:193`) and only the low-level `create_descriptor`
  binding exposes it.
* **Plan step 7 itself.** Not started here.
  `models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py`
  and the Llama equivalent exist and have still never been run.

## Do not

* Do not trust `ls /dev/tenstorrent | wc -l` as a mesh health check.
* Do not run without `HF_HOME=/localdev/ctr-apbernal/hf_data`, and do not accept
  a `skipped` in a run you meant to count.
* Do not pass `models/common/tests/modules` as a directory to pytest — it
  collects the 1D device suites and takes the mesh for ten minutes. The working
  host gate is `tttv2_milestone_b_evidence/qwen/host_gate.sh` (570 passed).
* Do not start a device cycle before the previous one has actually exited. A
  decode-mode `TT_FATAL` hangs the `mesh_device` fixture teardown **before**
  pytest writes any per-test verdict, so a verdict-keyed grace timer never arms
  and only the wrapper's full deadline reaps it. Watch the log's mtime, not its
  contents.
* Do not edit `models/common/modules/MILESTONE_A_STATUS.md` or
  `tttv2_2d_modules_plan.md` — job 0 and job 4 own them. This job's proposed text
  is in `REPORT.md` §A2.11.
* Do not touch `models/common/modules/**/*_1d.py` or `models/common/llm_runtime/**`.
  Both greps are empty across every Milestone B commit and should stay that way.
