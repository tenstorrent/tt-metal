# Job 1 (`mb-llama`), attempt 2 → whoever runs next: completion handoff

Written 2026-08-27 by `mb-llama` attempt 2, unattended.
Full account: `tttv2_milestone_b_evidence/llama/REPORT.md` §"Attempt 2", with the
run-by-run narrative in `tttv2_milestone_b_evidence/llama/ATTEMPT2.md` and the
environment in `.../ENVIRONMENT.md` §"Addendum — attempt 2".

`job1_completion_handoff.md` is attempt 1's and is still accurate about the code
it describes. **Where the two disagree, this one is later.** The three things it
got wrong, through no fault of its own:

1. **The mesh was recovered.** Board 7 is back; `tt-smi -ls` enumerated all 32
   boards at the start of attempt 2 and `test_partition_wh_galaxy.py` passed 5/5
   on device. Attempt 1's "if it is still broken, say so and stop" was the right
   instruction and the answer was "it is not broken".
2. **The `in0_block_w` hypothesis is confirmed on hardware.** Attempt 1 said "do
   not trust it until you have seen it run". It has now run in four separate
   processes and D-B9 does not recur. **D-B9 is CLOSED.**
3. **The decode graph goes further than attempt 1 could see.** A whole Llama
   layer *and* the final distributed norm execute. The frontier is the LM head.

## Read this first: the generalisation attempt 1 was one step away from

Attempt 1's summary was:

> A decode-mode program touched a core the loaded sub-device manager does not own.

That is right but incomplete, and the missing half cost attempt 2 two device
runs. The complete statement is:

> **A decode-mode program was placed on cores the sub-device manager does not
> own, _or was never told which sub-device it runs under_ — and several ttnn ops
> do not default to "the whole grid" when untold. They default to sub-device
> ZERO, which on this mesh is the prefetch sender set.**

Sub-device zero is `SubDevice([sender_cores])` — `x=0` and `x=4`. It is the one
group of cores a compute program must never use. So the untold default is
*strictly worse* than the whole-grid default, because the whole grid at least
contains the right cores. The symptom is not the familiar

```text
TT_FATAL ... Kernel group cores do not match sub device cores
```

but

```text
TT_FATAL ... Expecting a non-empty CoreRangeSet!   (program.cpp:1858)
  tt::tt_metal::CreateSemaphore(...)
```

because the op intersects its own core set with the sub-device's, gets the empty
set, and dies creating semaphores. If you see "non-empty CoreRangeSet", look for
a missing `sub_device_id` before you look at your placement.

The mechanism, verbatim, in
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`:

```cpp
auto subdevice_cores = device->worker_cores(
    HalProgrammableCoreType::TENSIX,
    sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
```

**Audit every op you call for a `sub_device_id` / `subdevice_id` parameter, and
pass it.** Two were missing in this tree and both are now fixed:
`LMHead2D`'s `ttnn.linear` (D-B13) and `GalaxyColumnAllReduce`'s
`ttnn.all_reduce` — the latter forwards straight to
`ttnn::experimental::all_reduce_async`
(`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`), so the plain
`all_reduce` name hides an op that very much cares.

## The decode LM head: what it is now, and why

This is the substantive change of attempt 2 and **Qwen needs the same one**.
Qwen's LM head is in exactly the state Llama's was: `LMHead2DConfig` built with
no `decode_program_configs`, `decode_input_memcfg=ttnn.L1_MEMORY_CONFIG`,
`decode_output_memcfg=ttnn.L1_MEMORY_CONFIG`, and no sub-device id — so it will
fail in the same four ways, in the same order.

**The decode LM head is a 24-core `gather_in0` ring matmul on the same ring the
MLP uses.** It always was in production: `_RING_CORE_COORDS` and
`_RING_RECEIVER_COORDS` in `models/common/models/galaxy/recipes.py` are,
coordinate for coordinate, `LM_HEAD_INPUT_GRID` and `LM_HEAD_OUTPUT_GRID` of
`models/demos/llama3_70b_galaxy/tt/model_config.py`. The recipes copied the ring
and wired only the MLP to it.

Four things must be true at once. Getting three right still fails:

1. **The program config** must be the ring, not `dense_matmul_program_config`.
   Decode presents **one row tile**, so a 2D multicast matmul can only use
   `grid_y = 1` — three cores, the worker envelope's width — and the local LM
   head output is 501 tiles wide, so `per_core_N = 167` and the in1 circular
   buffer is 668 tiles. The ring gives `per_core_N = 21` and 42 tiles.
2. **in0 and the output must be on the same cores in the same order.** Use
   `ring_cores()` for both. `ring_cores()` and `ring_receiver_cores()` are the
   **same 24 cores in a different order**, and that is enough to fail

   ```text
   TT_FATAL: ... Input tensor A and output tensor must be sharded on the same
             cores when using gather_in0 and in1 is DRAM_INTERLEAVED
   ```

   The MLP legitimately uses `receivers` for its output because its in1 arrives
   through the prefetcher's **global circular buffer**, so it never takes the
   `in1 is DRAM_INTERLEAVED` path. Do not copy the MLP's output placement.
3. **`sub_device_id`** — see the section above.
4. **`num_global_cb_receivers = 1`**, because the LM head is *not* prefetched.
   The MLP's qualified value is 2. It is now a parameter of
   `ring_matmul_program_config` (`global_cb_receivers`, default 2) so the MLP's
   value is untouched.

Recipe fields you inherit and should simply use, already resolved for any
geometry: `decode.lm_head_input_memcfg`, `decode.lm_head_output_memcfg`,
`decode.lm_head_program_config`.

## Two silent-failure traps worth more than the defects themselves

**1. `*_weights_memcfgs` in `LMHead2DConfig` is inert.** Setting it does nothing
for any weight whose `LazyWeight` already names a memory config, which is every
weight in both Galaxy models:

```python
def resolve_lazy_weight(weight, **kwargs):
    """Resolve the None fields of `weight`; do not override non-None fields"""
    to_set = {k: v for k, v in kwargs.items() if getattr(weight, k, None) is None}
```

and `model.py::_lazy` defaults `memory_config=ttnn.DRAM_MEMORY_CONFIG`. Attempt 2
spent a device run on a DRAM-width-sharded LM head weight that was never applied;
the `in1 is DRAM_INTERLEAVED` in the abort above is the proof. If you want a
weight placed differently, set it on the `LazyWeight` at construction — the way
every layer projection already does with its `ring_memory_config` — or supply a
separate mode weight (`prefill_wqkv` is the established pattern).

**2. `galaxy_hardware.load_reference_tokens` returned a `(1, 1024)` tensor** while
every consumer, here and in the 1D demos, treats the sequence as flat. `len()` was
therefore **1**, and a caller asking for a 512-token prompt got
`pytest.skip("reference sequence has 1 tokens")`. **The Milestone B accuracy gate
could not have run, for either model, and it failed _open_** — a skip, not a
failure. Fixed in the loader. If you inherit an accuracy number from anywhere,
check it was not a skip.

## Harness

`run_sequence.sh <manifest>` is new, next to attempt 1's scripts: it runs a list
of device cycles strictly sequentially, one pytest on the mesh at a time, each
followed by `cycle.sh`'s reap-and-reset. Manifest lines are
`<deadline-seconds> <logname> <node-id>`.

Attempt 1's three harness lessons all held. Three additions:

* **`[stage] leave close model` is not evidence the process will exit.** The
  model's own teardown completes and the process still hangs, because the hang is
  in the `mesh_device` *fixture* teardown after the test body returns. Every
  `TT_FATAL` run in attempt 2 still needed a reap.
* **`host_gate.sh` takes the mesh** — 64 `/dev/tenstorrent` fds. Both
  `test_recipes.py` and `test_lm_head_2d.py` open a device, and `test_plans.py`
  opens the whole cluster *despite a `MagicMock` mesh*, because
  `ttnn.SubDevice([cores])` reaches
  `MetalContext::create_default_instance_implicit_locked()`. A mocked mesh does
  not make a test host-only. Never run it beside a device cycle.
* **Do not pass `models/common/tests/modules/lm_head` as a directory.** It
  collects `test_lm_head_1d.py`, a real 8-device suite that walks a dozen
  checkpoints for well over ten minutes. `host_gate.sh` names the 2D file.

Also: a dirty fabric presents as 13 `test_plans.py` failures with
`Timed out waiting for ETH heartbeat on device ASIC ID ... ETH core e2-0`. That
is the mesh, not your code; `tt-smi -glx_reset` clears it. And under `-q` the
failures print as bare `FAILED` lines with no test id, because the device-open
log interleaves — the ids are only in the `-rA` summary at the end.

## What you inherit in the tree

Attempt 2's changes, on top of attempt 1's three commits:

```text
models/common/models/galaxy/recipes.py          LM head decode ring placements;
                                                `global_cb_receivers` parameter
models/common/models/galaxy/collectives.py      GalaxyColumnAllReduce subdevice_id
models/common/models/llama33_70b_galaxy/model.py  _relocate one-hop interleaved
                                                target; LM head wired to the ring
models/common/modules/lm_head/lm_head_2d.py     [SHARED] sub-device + mask-placement
                                                config surface
models/common/tests/models/galaxy/galaxy_hardware.py  load_reference_tokens squeeze
models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py
                                                KV-cache PCC; prefill 2048;
                                                cheap layer-subset reference
models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py
                                                composed-logits assertion
tttv2_milestone_b_evidence/llama/after_device_run.sh  reset timeout 600 -> 900
tttv2_milestone_b_evidence/llama/run_sequence.sh      new
```

### What Qwen inherits automatically, on top of attempt 1's list

1. **`_relocate` handles an interleaved non-DRAM target in one hop.** Qwen's own
   `_relocate` is a separate copy and attempt 1 already told you to port Llama's;
   port the *current* one, which has this fix too.
2. **`LMHead2DConfig` has `decode_sub_device_id` / `prefill_sub_device_id` and
   `decode_stage_mask` / `prefill_stage_mask`.** The defaults are inert, so Qwen's
   LM head behaves exactly as before until you set them. **You must set them.**
3. **`ring_matmul_program_config` takes `global_cb_receivers`** (default 2, the
   MLP's qualified value). Pass 1 for a non-prefetched matmul.
4. **`recipes.py` resolves `lm_head_input_memcfg`, `lm_head_output_memcfg` and
   `lm_head_program_config` for any geometry.** Qwen gets them free; wire them.
5. **`load_reference_tokens` returns a 1-D sequence.** Any accuracy number you
   measure can now actually run rather than skipping.

### One Qwen-specific warning attempt 1 could not give you

Qwen's `padded_vocab_size` is **not** equal to its `vocab_size`: 151936 pads to
152064. Llama's are equal (128256), which means **Llama's invalid-logits mask is
identically zero and Qwen's is not.** So the mask add is load-bearing for Qwen
and vacuous for Llama, and the mask placement (`decode_stage_mask`) is the one
piece of the LM head change that Llama cannot have qualified for you. Check it
deliberately: `_project` places the interleaved DRAM mask into the sharded output
placement with `interleaved_to_sharded` before adding, and the shard height is a
whole tile while the mask has one logical row. If that add rejects the mismatch,
that is a real defect and it is yours, not an inherited one.

## What is proven, and the boundary of it

Attempt 1's two proven items still stand, unchanged:

1. **The Llama adaptor is numerically correct on host** — 9 tests, 3 fresh
   processes. Attempt 2 did not re-run or alter this.
2. **A one-layer Llama model constructs, seals its prefetcher, resolves both CCL
   contexts, binds and unbinds a KV cache, and tears down cleanly on the mesh.**

Attempt 2 adds, with a log for each:

3. **`D-B9` is closed on hardware.** Four processes, no recurrence.
4. **A whole Llama decoder layer and the final distributed norm execute on
   silicon** with real layer-0 weights, at batch 32: distributed norm, QKV
   matmul, production `rotary_embedding_llama` on real Q/K, SDPA, `wo`, the
   attention all-reduce, all three MLP ring matmuls, the axis-0 all-reduce, and
   the final norm. This closes attempt 1's "RoPE composed with `Attention2D`" and
   L3 risks as *execution* questions.

**Read the boundary carefully.** Item 4 is "these programs build, launch and
complete without aborting". **No output was compared to a reference on the mesh
in attempt 2 either.** There is still no Llama PCC number, no accuracy number and
no demo text. The gates in `job1_llama.md` §Step 2 and §Step 3 are **not met**,
and the exact state of the LM head frontier at the end of attempt 2 is in
`REPORT.md` §"Attempt 2" — read that, not this summary, before you assume
anything about it.

The tests that would produce those numbers now exist and are correct as far as
static reasoning goes — `test_model_wh_galaxy.py` (one-layer prefill 128 +
decode + KV-cache PCC, and prefill 2048), `test_full_model_wh_galaxy.py` (80
layers, teacher-forced accuracy, batch 32) and `llama33_70b_galaxy/demo.py`. What
they have never had is a decode graph that runs to the logits.

## Suggested order for your night

1. **Check the mesh.** `tt-smi -ls` must exit 0 with 32 boards and no
   `ARC startup error`; then `test_partition_wh_galaxy.py` (about 50 s including
   mesh open). If `tt-smi -ls` hangs for five minutes and then reports
   `ARC startup error at core 0-10 over NOC0`, a chip's ARC controller is wedged;
   `timeout 900 tt-smi -glx_reset` fixed it once. **Use 900 s, not 600 s.**
2. **Read `REPORT.md` §"Attempt 2" for the LM head frontier** and continue from
   exactly there. Do not re-derive the four LM head constraints; they are listed
   above and each cost a device run.
3. **Then the step-2 gate**: `test_model_wh_galaxy.py`. It is now cheap per
   process (about 12 GB of checkpoint, not 141 GB), so the three-runs rule is
   affordable — use it.
4. **Only then step 3.** Budget for it honestly: `from_pretrained` loads the whole
   141 GB checkpoint eagerly, once per process, and each of
   `test_full_model_wh_galaxy.py`'s tests calls `_load` separately. There is no
   shared-model fixture. If you need several 80-layer runs, that cost dominates
   the night and is worth fixing first.

## Two dependencies to record rather than absorb

* **Step 3 cannot avoid paged KV.** `from_pretrained` does
  `paged = paged_attention_config or default_paged_attention_config(params)`, so
  passing `None` yields the *default paged* geometry, not a contiguous cache;
  there is no argument that selects contiguous. Every 80-layer path — the
  full-model test, the accuracy gate, the demo — is therefore paged, while
  `job1_llama.md` puts paged KV in step 7 and `test_model_wh_galaxy.py`'s own
  docstring records paged decode as unqualified. `GalaxyDirectRunner` supports
  both (`self.paged = spec.paged_attention_config is not None`); the model
  loader does not expose the choice. **This is a real scope dependency between
  steps 3 and 7 and it is stated here rather than absorbed silently.**
* **`num_global_cb_receivers` on the MLP ring config is still unset for
  `allowed_worker_cores`** — attempt 1's §4.2 recommendation stands untouched, on
  purpose: it currently works and a qualified path should not be changed on a
  night with no way to re-qualify it. The warning is in every log.
