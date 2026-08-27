# Milestone C brief — executors, runtime integration, tracing, vLLM

Written 2026-08-27 by `mb-signoff`, at commit `9d3ec5799ef` on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

> **Read this before you plan anything.** The plan says *"Do not begin executor/vLLM integration
> until both models pass Milestone B."* **Neither model passed.** Milestone B's exit gate is
> **NOT PASSED** — see [`models/common/models/MILESTONE_B_STATUS.md`](models/common/models/MILESTONE_B_STATUS.md).
> This brief exists so the handoff is written down while it is fresh, **not** as authorisation to
> start. The gate is the point of the gate.
>
> The blocker is infrastructure, not code: **eleven of the 32 Galaxy boards have been off the PCIe
> bus since 2026-08-26**, and three consecutive device jobs produced no numerical result from
> silicon for either model. Milestone C's first task is not code. It is:
>
> 1. an IPMI power cycle or host reboot of `wh-glx6u-05` (`tt-smi` cannot fix this — see below);
> 2. fetching the Qwen3-32B checkpoint (~65 GB) into `/proj_sw/user_dev/hf_data`;
> 3. re-running Milestone B's device gates and getting the two accuracy numbers.
>
> Until those three are done, everything below is a plan against unqualified code.

---

## 1. The two hard blockers, in the detail you need

### 1.1 The mesh

Verified again by this job at 2026-08-27T03:34Z:

```sh
ls /sys/class/tenstorrent | wc -l     # 21   <- authoritative
ls /dev/tenstorrent   | wc -l         # 32   <- LIES. Do not use it.
```

Missing boards: `0 1 2 3 4 5 6 7 10 11 14`. Every `ttnn` cluster open dies at
`TTDevice::is_pcie_hung — Read 0xffffffff over PCIe ID 17`.

**Do not spend recovery attempts.** Four were spent across `mb-llama` and `mb-qwen` and all four
failed. `tt-smi -glx_reset` fails with `[Errno 6] No such device or address: '/dev/tenstorrent/7'`
— the reset path needs the very node that is gone — and `tt-smi -r` fails on the same PCIe read.
`tt-smi -ls` aborts inside `tt_umd` and lists zero boards. `dmesg` shows a kernel oops with
`irqs disabled`. This needs power, not software.

A working theory, offered as a theory: the ETH instability that preceded the PCIe failure correlated
with `TT_FATAL` aborts inside multi-sub-device programs, which leave the mesh un-drainable
(teardown blocks in `FDMeshCommandQueue::~FDMeshCommandQueue`) and required ~23 resets in one
session. Whether those resets *caused* the PCIe failure or merely preceded it is not determined and
is not claimed. If it is causal, it is a real operational risk for Milestone C, which will run far
more device hours than Milestone B did.

### 1.2 Qwen's weights

The HF cache entry for Qwen3-32B is **config-only** (12 KB, `config.json`). There is no Qwen3-32B in
`/proj_sw/user_dev/hf_data`. A healthy mesh does **not** unblock the Qwen accuracy gate.

Llama-3.3-70B *is* present — 31 safetensors shards, 368 GB, at
`/proj_sw/user_dev/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct`.

**Export `HF_HOME=/proj_sw/user_dev/hf_data`.** It is unset in the inherited environment and the
failure mode is a silent **skip**, not a failure: `snapshot_download` falls through to the network
and takes a 401 on a gated repo, and the real-checkpoint tests skip. A gate that skips looks green.

---

## 2. What Milestone C inherits as working

Everything in this section is **host** evidence unless it says otherwise. Every command below was
run at commit `9d3ec5799ef`.

| What | Evidence | Command |
| --- | --- | --- |
| Llama weight conversion + Llama 3 scaled RoPE, against the real checkpoint | 9 tests, 3 fresh processes, **0 skips** (audited: the real-checkpoint cases genuinely ran) | `HF_HOME=/proj_sw/user_dev/hf_data pytest models/common/tests/models/llama33_70b_galaxy/test_hf_conversion_host.py` |
| Qwen weight conversion, Q/K norm, and the **64-head decoupled geometry** | 13 tests, 3 fresh processes; attention rebuilt from converted tensors alone reproduces unmodified HF `Qwen3Attention` at PCC ≥ 0.9999 | `pytest models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py` |
| Both models' host model wiring | 59 passed | `pytest models/common/tests/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/test_model_host.py` |
| Step-7 host coverage: paged KV, concat-32, prefix cache, sampling, long context, repeat/cleanup | **162 tests, 3 fresh processes, identical** | `pytest models/common/tests/models/galaxy/test_step7_*.py` |
| `llm_runtime` — untouched by Milestone B and green | 1032 passed, 1 skipped | `pytest models/common/tests/llm_runtime` |
| The decode sub-device partition, **on silicon** | 5 passed in 12.8 s, no checkpoint needed | `pytest models/common/tests/models/galaxy/test_partition_wh_galaxy.py` |
| A one-layer Llama model builds, seals, resolves CCL and tears down, **on silicon** | **PASSED once** in 109 s with real layer-0 weights — a single run, not qualified | `pytest models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py::test_one_layer_model_constructs_and_closes` |

**`test_partition_wh_galaxy.py` is the cheapest useful thing on this list.** It needs no checkpoint,
runs in 13 s, and tells you the worker envelope is not contiguous and that sender ∪ worker does not
cover the compute grid. Run it first whenever a decode program aborts on placement — most of
Milestone B's nine silicon defects were that fact, rediscovered nine times.

---

## 3. What Milestone C inherits as broken or unqualified

### 3.1 Never measured, by anybody, at any tree

- **Llama teacher-forced accuracy** (top-1 ≥ 91%, top-5 ≥ 99%). No number exists.
- **Qwen teacher-forced accuracy** (top-1 ≥ 89%, top-5 ≥ 97%). No number exists.
- **Any block-level PCC**, either model, either mode. Decode never reached the LM head.
- **Any KV-cache PCC.** Nothing in this tree has ever compared the paged and contiguous layouts.
- **Any demo output**, any batch.
- The **4K / 32K / 128K** functional smokes. Capacity was accounted arithmetically instead; the
  numbers are in `MILESTONE_B_STATUS.md`.
- **Prefix-cached vs uncached** agreement.
- **L1 / global-CB ownership at model scale.** The 80-layer model has never been built, so no one
  has ever observed a second Galaxy model construction in one process.

### 3.2 Open defects, carried forward

Full descriptions and "how it hid" are in `MILESTONE_B_STATUS.md`. The short list:

| ID | Where | Why it matters to Milestone C |
| --- | --- | --- |
| **D-B9** | `recipes.dense_matmul_program_config` | The attention decode matmul's CBs clash with L1 by ~20 kB. A candidate fix (`in0_block_w` `gcd(k,8)`→`gcd(k,4)`) is **in the tree, host-green, and has never run on hardware.** Treat it as a hypothesis, not a fix. |
| **D-C1** | `attention_2d._validate_decode_page_table` | A prefill-shaped page table is **accepted** by decode. Needs a contract decision (see §4). |
| **D-C2** | `sampling_2d._seed_digest` | A seeded request that migrates slots changes its stream. Needs a **product** decision before vLLM serving is built on it (see §4). |
| **G-C1** | `direct_runner.prefill_batched` | Concat-32 needs all 32 slots active; it does not compose with the `active_slots < 32` sink-block mechanism. Constrains the DP=4 / `max_num_seqs 8` serving shape. |
| **G-C2** | `direct_runner.prefill_batched` | An empty row is rejected one call too late — after the whole concatenated graph has run. |
| **G-C3** | `attention_2d._validate_prefill` | Dead guard: a chunk table alone already selects `PREFIX_CHUNKED`, so a chunk table with no chunk start silently runs the chunked recipe from token 0. |
| **F-C2** | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster (`ttnn.SubDevice` constructs the `MetalContext`). On a healthy mesh these 13 should pass — **and if they do not, that is a finding.** |
| **O2** | 5 1D demo-contract tests | FAIL, and **not Milestone B's** — proven byte-identical to Milestone A. Someone must own them. |

### 3.3 One host assumption a mesh must settle, first

`step7_harness.py` models a non-obvious `ttnn` fact — a distributed tensor's `.shape` is the
**shard** shape, not the global one — read out of `TensorToMesh::Impl::create_tensor`, **not
measured on silicon.** D-C1 rests on it. One line settles it:

```python
t = ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), device=mesh,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(8, 4)),
                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
assert tuple(t.shape) == (8, 64)
```

If it is `(32, 64)` instead, **D-C1 is worse than described**: the device-local-rows branch would be
unreachable for a correctly-mapped table and decode's page table would have no effective validation
at all. Run this in the first hour you have a mesh.

---

## 4. Items routed to Milestone C by name

### L1 — `Prefetcher2D` global-CB ownership redesign

`Prefetcher2D.cleanup()` clears `self._global_cb` **without ever handing it to `deallocate`** —
ttnn exposes no free for a global circular buffer. Confirmed on host by `mb-coverage` with the
module suite's injectable `create_global_cb`/`deallocate`: after `cleanup()` the owner truthfully
reports `owned_resources == ()` while the CB it created was never freed. Two owners in one process
allocate two CBs and free neither, ~55 MB of L1 each.

The **OOM itself needs real L1 and has never been reproduced.** The recommended design fix stands
from Milestone A: make `global_cb` a property on the context rather than a stored handle.

This belongs to Milestone C because the executor work is what will actually exercise repeated owner
lifecycles — and because the plan's own functional gate asks for *"repeated startup, serving, and
cleanup without retained TT resources"*, which is exactly this contract. `test_two_models_in_one_process`
exists in `llama33_70b_galaxy/test_bringup_wh_galaxy.py` and **has never run.**

### D-A — physical-32 real-device trace

Milestone A deferred this here and the reason is now demonstrably right rather than merely stated:
it needs a model-owned executor with `TraceCompiler`/`TracedExecutor` running a 2D model at batch 32
on `(8, 4)`. Executors are Milestone C and the 2D models are Milestone B, so **there was genuinely
nothing on the Galaxy to trace before now** — and, as it turned out, there still is not.

Milestone A also noted a cheaper half worth separating: the batched-prefill delegation has **no
device evidence of any kind**, and could be exercised on N150/T3K with an existing 1D model to prove
the default is byte-for-byte preserved. That does not need a Galaxy and does not need Milestone B.
See `tttv2_milestone_a_gap_briefs/gap3_batched_prefill_physical32_trace.md`.

### The Galaxy CCL / `tt_ccl.py` merge evaluation

Deferred by the plan until **both models pass**. They have not, so this is still deferred — recorded
here so it is not lost, not so it can be started.

When it is evaluated, two inputs are already on record:

- the `semaphore_cores` invariant from Milestone A's D3: narrowing a mode's semaphore allocation
  below its worker sub-device is safe **only** for a collective that binds its semaphore to a grid
  it owns (as the fused RMS all-gather does). The generic async CCLs choose senders from the
  sub-device and must keep the default, or they hang on uninitialised L1. Milestone B promoted this
  into `GalaxyModePlan` validation with an explicit `allow_narrow_semaphore_cores=True` opt-out, so
  a narrowing is now a stated one rather than a silent one;
- `ttnn.reduce_scatter` **cannot run on this partition at all.** Its program factory takes
  `worker_cores(TENSIX, sub_device_id).bounding_box()` and lays workers out from that rectangle's
  origin — and the Galaxy worker bounding box spans `x=1..6`, straight across the `x=4` sender
  column. The file carries its own `// interaction with subdevice needs to be investigated`.

### Two contract decisions that need a human, not a fix

Both were measured and deliberately left alone. Neither is a bug to patch.

- **D-C1 — is a decode page table discriminated by shape or by placement?** Shape cannot separate
  the prefill layout from a legitimate 4-core L1-sharded repeat; both present 32 rows. The
  discriminator that would work is `memory_config()`, which the validator never consults. The
  proposed fix is written out in `MILESTONE_B_STATUS.md`, and it requires changing an existing 2D
  module test expectation — which is why no job committed it unilaterally.
- **D-C2 — is a sampling seed per-request, or per-(request, slot)?** The slot is part of the seed
  digest deliberately, so that 32 slots handed one seed by a serving front end do not all emit the
  same token. That protection is proved by a test. It also means a migrating request does not keep
  its stream, which is the opposite of the step-7 requirement. **Put this in front of whoever owns
  the serving contract before building vLLM integration on top of it.**

---

## 5. Performance methodology — set this up first, do not retrofit it

The plan gates Milestone C on paired TTTv1/TTTv2 measurement. **Build the harness before the first
measurement, not after**, because a paired comparison retrofitted onto runs that did not control
these variables is not a comparison.

Every one of these must be identical across the pair:

- same WH Galaxy host;
- same repository commit **and** firmware/runtime environment;
- same checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling and KV setup.

And the procedure:

- **one unmeasured warmup**;
- **three measured runs**;
- **compare medians** (not means, not best-of);
- **retain profiler artifacts and the exact commands.**

Thresholds: no gated TTTv2 metric may regress by more than **3%** from its paired TTTv1 median, and
TTTv2 must also meet the absolute targets:

```text
Llama, batch 32 / sequence 507      TTFT <= 99 ms    decode >= 71.5 tok/s/user   aggregate >= 2288 tok/s
Qwen,  batch 32 / sequence 507      TTFT <= 700 ms   decode >= 60   tok/s/user   aggregate >= 1920 tok/s
```

**Three known performance debts are already outstanding against those numbers**, all incurred in
Milestone B for correctness and all recorded rather than hidden:

1. the attention decode QKV and `wo` matmuls are confined to **three worker columns instead of
   seven** (D-B5). The structural fix is moving them to the 24-core ring/`gather_in0` form the MLP
   already uses — the recipes already contain `attention_qkv_collective_input_memcfg` shaped for
   exactly those 24 ring cores, so the design anticipated it and the matmuls were simply left
   behind. This also dissolves D-B9;
2. **every sharded→sharded relocation now goes through DRAM** (D-B7), one round trip per placement
   hop, because all three obvious in-place spellings reach a full-grid program factory;
3. the ring matmul config still leaves `allowed_worker_cores` **unpopulated** and auto-populates it
   from the full compute grid — eight warnings per decode step, and `ttnn` says it "will become a
   hard error in a future release". It was deliberately not changed on a night with no way to
   re-qualify it. It is the exact hazard behind D-B2 and D-B5.

Measure the baseline *before* fixing these, so the fixes have something to be measured against.

---

## 6. An upstream filing Milestone C should make

Four tt-metal ops choose their cores from the full compute grid or from a sub-device **bounding
box**, neither of which is the worker set. A non-contiguous worker sub-device is a normal Galaxy
configuration, not an edge case, so "bounding box" is a latent bug for every model that partitions
its grid:

```text
copy_default_tilized_program_factory.cpp:44    device->compute_with_storage_grid_size()   (TODO already present)
reshard_program_factory_generic.cpp:80         same
reduce_scatter_program_factory.cpp:107         sub-device BOUNDING BOX  (// interaction with subdevice needs to be investigated)
typecast_program_factory.cpp:109               full grid (the *sharded* typecast factory is fine; the fallback is not)
```

Milestone B worked around all four. Milestone C will hit them again the moment it builds an executor
that composes ops Milestone B did not.

---

## 7. Suggested order

1. **Get the mesh back.** IPMI power cycle or host reboot. Confirm with
   `ls /sys/class/tenstorrent | wc -l` → 32, *not* `/dev/tenstorrent`.
2. **Fetch the Qwen3-32B checkpoint** into `/proj_sw/user_dev/hf_data`. It is on the critical path
   for one of the two accuracy gates and takes hours.
3. **Settle the shard-shape assumption** (§3.3). One line, first hour.
4. **Run `test_partition_wh_galaxy.py`** (13 s) and the one-layer bringup, three times each in fresh
   processes. The one-layer model has passed exactly once; qualify it.
5. **Test the D-B9 hypothesis** — the `in0_block_w` change that has never run on hardware. Until it
   holds, the Llama decode graph does not complete a single step, and nothing downstream of it can
   be measured.
6. **Finish Milestone B.** Block PCC, then the 80-layer model, then the two accuracy gates, then
   step 7's device half. Only then is the Milestone C gate open.
7. **Then** start executors — with the paired measurement harness of §5 already standing.

---

## 8. How to read the evidence

| Package | What it holds |
| --- | --- |
| `tttv2_milestone_b_evidence/reconcile/` | The Milestone A/B rebase, C1–C10 disposition, host gates. 30 logs |
| `tttv2_milestone_b_evidence/llama/` | First silicon. Nine defects, the partition probe, the dead mesh. **109 logs** |
| `tttv2_milestone_b_evidence/qwen/` | Host qualification of the 64-head geometry; `BLOCKED (infra)`. 18 logs |
| `tttv2_milestone_b_evidence/coverage/` | Step-7 host coverage, D-C1/D-C2, capacity accounting. 26 logs |
| `tttv2_milestone_b_evidence/signoff/` | This job's independent re-verification of the exit gate |

Two rules that this project learned the hard way and that Milestone C should keep:

- **a test that passes once has proved nothing on this hardware.** Three of Milestone A's four
  defects presented as intermittent *passes*, not failures, because they read aliased or
  uninitialised L1. Three runs in fresh processes is the minimum bar for a device claim;
- **never read a device test in this tree as evidence unless a log shows it ran.** 33 of them
  (`test_step7_coverage_wh_galaxy.py`, both models) are committed and have **never been executed**.
  They say so in their own module docstrings. Treat a first run as bringup, not as a regression.
