# `mb-llama`, attempt 2 — running log

Attempt 1 (`REPORT.md`, `logs/`) ended `BLOCKED (infra)`: board 7 had dropped off
the PCIe bus and `tt-smi -glx_reset` could not recover it. **That is no longer
true.** This attempt's first act was to re-check, and the mesh is healthy.

Everything below is this attempt. Attempt 1's logs are in `logs/`; this
attempt's are in `logs2/`, and no log from either is ever overwritten.

## The mesh was recovered between the two attempts

Established, not assumed:

```text
ls /dev/tenstorrent | wc -l    -> 32
tt-smi -ls                     -> 32 Wormhole boards enumerate, including
                                  board 7 at 0000:08:0x (exit 0; attempt 1 saw
                                  this abort inside tt_umd and list zero boards)
```

and then, on device, `logs2/a2_00_partition.log`:

```text
5 passed in 51.38s
```

with a clean open and close of all 32 chips and `holders_after: 0`. The handoff
was right about the tree and right about the mesh *at the time it was written*;
the machine has since been power-cycled by someone outside this job.

## Runs

| # | Log | What | Result |
| --- | --- | --- | --- |
| 00 | `logs2/a2_00_partition.log` | partition + mesh health | **5 passed** |
| 01 | `logs2/a2_01_decode_step.log` | one decode step, batch 32, real layer-0 weights | **FAILED** — D-B9 confirmed *fixed*; new defect **D-B10** |
| 02 | `logs2/a2_02_host_gate.log` | host regression gate | 2 failed, 559 passed — both mine, see below |
| 03 | `logs2/a2_03_lm_head_host.log` | `test_lm_head_2d.py` after the correction | **20 passed** |
| 04 | `logs2/a2_04_host_gate.log` | host regression gate | 13 failed, 548 passed — **all** the dirty-fabric ETH symptom, not code |
| 05 | `logs2/a2_05_reset.log` | `tt-smi -glx_reset` | `Re-initialized 32 boards` |
| 06 | `logs2/a2_06_decode_step.log` | one decode step | **FAILED** — D-B10 fixed; new defect **D-B12** |
| 07 | `logs2/a2_07_decode_step.log` | one decode step | **FAILED** — D-B12 fixed; new defect **D-B13** |
| 08 | `logs2/a2_08_lm_head_host.log` | `test_lm_head_2d.py` after D-B13's fix | **20 passed** |
| 09 | `logs2/a2_09_tt_smi_after_failed_reset.log` | mesh health after the timed-out reset | **exit 1** — `ARC startup error at core 0-10` |
| 10 | `logs2/a2_10_recovery1_glx_reset.log` | recovery attempt 1, `glx_reset` at 900 s | **`Re-initialized 32 boards`** |
| 11 | `logs2/a2_11_tt_smi_after_recovery.log` | mesh health after recovery | **exit 0**, 32 boards, 0 errors |
| 12 | `logs2/a2_12_decode_step.log` | one decode step | **FAILED** — D-B13 fixed; new defect **D-B14** |
| 13 | `logs2/a2_13_host_quick.log` | `test_lm_head_2d.py` + `test_model_host.py` after D-B14's fix | **50 passed** |
| 14 | `logs2/a2_14_decode_step.log` | one decode step | **FAILED** — D-B14 fixed; new defect **D-B15** |
| 15 | `logs2/a2_15_host_quick.log` | host after D-B15's fix | **58 passed** |
| 16 | `logs2/a2_16_decode_step.log` | one decode step | **FAILED** — D-B15 fixed; new defect **D-B17** |
| 17 | `logs2/a2_17_lm_head_host.log` | `test_lm_head_2d.py` | **20 passed** |
| 18 | `logs2/a2_18_host_quick.log` | host after D-B17's fix | **50 passed** |
| 19 | `logs2/a2_19_decode_step.log` | one decode step | **FAILED** — resource key width; **D-B16 struck** |
| 20 | `logs2/a2_20_host_quick.log` | host after the key fix and the D-B16 revert | **50 passed** |
| 21 | `logs2/a2_21_decode_step.log` | one decode step | **FAILED** — key resolves, reduction reached; new defect **D-B18** |
| 22 | `logs2/a2_22_host_quick.log` | host after D-B18's fix | **50 passed** |
| 24 | `logs2/a2_24_accuracy_plumbing_host.log` | accuracy-gate plumbing, host only | **runnable**: 512/511 resolves, perfect prediction scores (1.0, 1.0) |

Every device run so far has failed *later* than the one before it, at a
different op, and each failure was a placement or sub-device fault with an exact
site. None reproduced a previously fixed defect.

## Result 01 — D-B9 is fixed, and the graph now reaches the LM head

The change attempt 1 left in the tree as an unverified hypothesis — `in0_block_w`
from `gcd(k_tiles, 8)` to `gcd(k_tiles, 4)` in `dense_matmul_program_config` —
**works on hardware.** The `TT_THROW ... Statically allocated circular buffers
in program 320 clash with L1 buffers on core range [1-0 - 3-0]` of D-B9 does not
occur; the log contains no `clash` at all, and execution proceeded past both
attention projections into the MLP ring matmuls and out the other side.

Stages reached, from the log's `[stage]` markers:

```text
[stage] leave build
[stage] leave allocate kv cache
[stage] leave bind kv cache
[stage] leave activate decode (starts the persistent DRAM prefetch program)
[stage] leave prepare decode rot mats (RotarySetup2D.decode_forward)
[stage] leave stage current positions
[stage] leave embed decode
[stage] enter decode forward      <- aborted inside
[stage] leave close model         <- model teardown itself completed
```

So the whole of one Llama layer *and* the final distributed norm executed. The
abort is the first op after them.

### D-B10 — `_relocate` reached `ttnn::prim::copy` for an *interleaved* target

The abort, from `logs2/a2_01_decode_step.log`:

```text
File ".../llama33_70b_galaxy/model.py", line 1389, in decode_forward
  return self.lm_head.decode_forward(_relocate(normed, self.config.lm_head_config.decode_input_memcfg))
File ".../llama33_70b_galaxy/model.py", line 1100, in _relocate
  placed = ttnn.to_memory_config(staged, target_memcfg)
RuntimeError: TT_FATAL @ .../program.cpp:2205: num_intersections == num_cores
Kernel group cores do not match sub device cores for programmable core type TENSIX
backtrace:
 --- ttnn::prim::copy(...)
 --- ttnn::to_memory_config(...)
```

**Root cause.** `_relocate`'s sharded-source branch always staged into
*DRAM* interleaved and then, if the target was also interleaved, asked
`to_memory_config` to finish the move. The LM head's `decode_input_memcfg` was
`ttnn.L1_MEMORY_CONFIG` — interleaved, but **L1**, not DRAM — so that second hop
was an interleaved-to-interleaved move, which is exactly the
`ttnn::prim::copy` full-compute-grid factory that D-B7 was written to avoid.
D-B7 fixed the *sharded* targets and left this one, because nothing had ever
reached an interleaved non-DRAM target: the decode graph had always aborted
earlier.

Attempt 1 catalogued this op as unsafe and its own docstring names it. The path
was simply not covered.

**Fix.** `sharded_to_interleaved` takes the interleaved destination directly and
runs on its *input's* `shard_spec.grid`, so it is worker-confined whatever the
destination buffer type is. One hop replaces two, and the `prim::copy` call is
gone. This also removes a DRAM round trip.

### D-B11 — the decode LM head had no program config at all

D-B10 was only the staging op. The matmul behind it is the real defect, and it
would have aborted one op later.

`LMHead2DConfig.decode_program_configs` resolves to `(None,)` when a model does
not supply one, and the Llama model did not. With no program config `ttnn`
auto-selects a work grid, which is the full seven-column compute grid — the same
class of fault as D-B2 and D-B5.

It cannot be fixed with `dense_matmul_program_config`, the way the attention
projections were. Decode presents **one row tile**, so a 2D multicast matmul can
only use `grid_y = 1`: three cores, the width of the worker envelope. The local
LM head output is 501 tiles wide, giving `per_core_N = 167` and an in1 circular
buffer around 240 kB per core before double buffering — it cannot fit beside the
resident decode activations, which is D-B9 again and worse.

**Fix — use the ring that was always the LM head's ring.** The decisive evidence
is in this tree already: `_RING_CORE_COORDS` and `_RING_RECEIVER_COORDS` in
`models/common/models/galaxy/recipes.py` are, coordinate for coordinate, the
production `LM_HEAD_INPUT_GRID` and `LM_HEAD_OUTPUT_GRID` of
`models/demos/llama3_70b_galaxy/tt/model_config.py`, whose LM head is a 24-core
`gather_in0` ring at this exact geometry (`LM_HEAD_RING_SIZE = 24`,
`LM_HEAD_TG_RING_PROGCFG`, `k = dim // 4`, `n = padded_vocab // 8`). The
Milestone B recipes copied the ring but wired only the MLP to it.

Measured geometry (`local_dim` 2048, `local_padded_vocab_size` 16032):

```text
                 local_k  local_n  padded_n  in0_block_w  per_core_N
mlp w1/w3          2048     7168      7680        2           10
mlp w2             7168     2048      2304        9            3
lm_head  (new)     2048    16032     16128        2           21
```

`per_core_N = 21` against the dense form's 167, and an in1 buffer of 42 tiles
against 668. The LM head's in0 is the final-norm output at `local_dim`, which is
the **same width the MLP feeds its ring**, so it reuses `mlp_input_memcfg`'s
placement exactly rather than inventing one.

Also fixed alongside, all in the same failure class:

* `num_global_cb_receivers` is now a parameter of `ring_matmul_program_config`.
  The MLP keeps the qualified 2; the LM head passes 1, because it is not
  prefetched — it streams a DRAM width-sharded weight, exactly as
  `LM_HEAD_TG_RING_PROGCFG` does with `prefetch=False`.
* `GalaxyColumnAllReduce` never passed `subdevice_id`. `ttnn.all_reduce`
  forwards straight to `ttnn::experimental::all_reduce_async`
  (`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`), which places
  workers on the whole grid when given no sub-device. It now takes a *callable*,
  because the sub-device id belongs to the live context, which is created after
  the collective and differs between prefill and decode.
* `LMHead2D._project` places the invalid-logits mask into the output's placement
  before adding it. The mask is one module-owned interleaved DRAM tensor shared
  by both modes, and decode's output is now sharded.

### A note on how D-B11's fix was kept inside the module's contract

The first version of the `_project` change asked `output_memcfg.is_sharded()`
inline. That broke two `test_lm_head_2d.py` tests
(`test_lm_head_2d_deallocates_projection_transients`,
`test_lm_head_2d_repeat_projection_and_collective_failure_cleanup`,
`logs2/a2_02_host_gate.log`: 2 failed, 559 passed) — and it broke them for a
good reason, not a nuisance one: those tests drive `_project` with opaque
`object()` sentinels for the memory config, deliberately, so that the module's
deallocation plumbing is testable without a mesh. Interrogating an argument's
type inside the hot path took that away.

So the decision moved to `_resolve_lm_head2d_config`, where the real memory
configs are, and arrives at `_project` as a resolved `bool`
(`decode_stage_mask` / `prefill_stage_mask`). The tests were **not** changed:
20/20 pass unmodified (`logs2/a2_03_lm_head_host.log`). A test that fails
because a change was structurally wrong is doing its job.

### A host-gate failure that was the mesh, not the code

`logs2/a2_04_host_gate.log` came back **13 failed, 548 passed**, every failure in
`models/common/tests/models/galaxy/test_plans.py`. None of them were a
regression:

```text
RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID:
              87032054158471756, ETH core e2-0 (NOC0) to advance.
              Stuck at 0xabcd68c1
  tt::umd::TopologyDiscovery::eth_heartbeat_running(...)
  ...
  tt::tt_metal::MetalContext::create_default_instance_implicit_locked()
```

This is exactly the dirty-fabric symptom attempt 1's `after_device_run.sh` was
written to prevent, quoted in its own comment. A `tt-smi -glx_reset` clears it.
Two things are worth recording because both cost time:

1. **`test_plans.py` opens the whole 32-chip cluster** even though its mesh is a
   `MagicMock`. `ttnn.SubDevice([cores])` reaches
   `MetalContext::create_default_instance_implicit_locked`, which constructs the
   cluster. A mocked mesh does not make a test host-only.
2. **The failure presents as 13 assertion-free `FAILED` lines with no test id on
   them** under `-q`, because the device-open log interleaves. The ids only
   appear in the `-rA` short summary at the end, and the cause only in the
   `FAILURES` section, so an early `grep FAILED` is actively misleading.

## Result 06 — the LM head ring matmul reached execution

`logs2/a2_06_decode_step.log`. D-B10 is fixed (the `_relocate` staging no longer
aborts) and the LM head's own matmul now runs and rejects its *placement*, which
is a much later and much more specific failure:

```text
TT_FATAL: MatmulMultiCoreReuseMultiCast1DProgramConfig: Input tensor A and
          output tensor must be sharded on the same cores when using gather_in0
          and in1 is DRAM_INTERLEAVED.
  matmul_device_operation.cpp:1835:
  input_tensor_a.shard_spec().value().grid == attributes.output_mem_config.shard_spec().value().grid
```

### D-B12 — two independent mistakes behind one message

**1. `ring_cores()` and `ring_receiver_cores()` are the same 24 cores in a
different order.** Both sets are, per column: 4 cores on `x=1`, 4 on `x=2`, 8 on
`x=5`, 8 on `x=6`. They differ only in the order the single-core ranges are
listed, and a `CoreRangeSet` built from ordered single-core ranges compares
unequal. The LM head output was on `receivers` by analogy with the MLP; it is now
on `ring`, matching its own in0.

The MLP gets away with `receivers` because its in1 arrives through the
prefetcher's **global circular buffer**, so it never takes the
`in1 is DRAM_INTERLEAVED` path at all. The LM head is not prefetched. Copying a
qualified placement from a neighbouring op is exactly the kind of reasoning that
looked safe and was not.

**2. `decode_weights_memcfgs` was dead config.** The intent was a DRAM
width-sharded weight, as the production LM head uses. It never took effect:

```python
def resolve_lazy_weight(weight, **kwargs):
    """Resolve the None fields of `weight`; do not override non-None fields"""
    to_set = {k: v for k, v in kwargs.items() if getattr(weight, k, None) is None}
```

and `model.py::_lazy` already defaults `memory_config=ttnn.DRAM_MEMORY_CONFIG`,
so the LM head weight arrived at `LMHead2D` with its placement *already set* and
the config's value was silently discarded. The `in1 is DRAM_INTERLEAVED` in the
assertion is the proof.

That is worth flagging beyond this defect: **every `*_weights_memcfgs` entry in
`LMHead2DConfig` is inert for any weight whose `LazyWeight` already names a
memory config**, which is all of them in both Galaxy models. It fails silently,
not loudly. The model now says `ttnn.DRAM_MEMORY_CONFIG` there so the config
states what actually happens, and the placement is satisfied on the output side
instead.

## Result 07 — the ring matmul's core set is intersected with sub-device *zero*

`logs2/a2_07_decode_step.log`. D-B12 is fixed; the LM head matmul now passes
placement validation and fails building its program:

```text
TT_FATAL: Expecting a non-empty CoreRangeSet!    (program.cpp:1858)
  tt::tt_metal::CreateSemaphore(...)
  ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized(...)
  ...
  File ".../modules/lm_head/lm_head_2d.py", line 172, in _project
    partial = ttnn.linear(
```

### D-B13 — `LMHead2D` never passed `sub_device_id` to `ttnn.linear`

Read the factory
(`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`,
the `gather_in0` path):

```cpp
CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
CoreRangeSet non_idle_cores  = all_worker_cores.merge(hop_cores);
auto subdevice_cores = device->worker_cores(
    HalProgrammableCoreType::TENSIX,
    sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
for (const auto& cr : subdevice_cores.ranges()) {
    auto intersection = non_idle_cores.intersection(cr);
    if (intersection.empty()) continue;
    ...
}
all_cores = CoreRangeSet(non_idle_cores_vec);
...
auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
```

The op's core set is **the ring intersected with the named sub-device**, and with
no `sub_device_id` it falls back to `get_sub_device_ids().at(0)`. Under the
Galaxy decode manager, sub-device 0 is the **prefetch sender** set
(`sub_devices=(SubDevice([sender_cores]), SubDevice([worker_cores]))`), which is
`x=0` and `x=4` — completely disjoint from the ring. So `all_cores` came out
empty and `CreateSemaphore` refused it.

This is the same defect family as every other one in this milestone, but with a
new twist worth stating plainly: **the default is not "the whole grid", it is
"sub-device zero"**, and on this mesh sub-device zero is the one set of cores a
compute program must never use. A silent wrong-sub-device default is more
dangerous than a silent whole-grid default, because the whole-grid case at least
*contains* the right cores.

`MLP2D` already does this correctly, via `_prefetch_kwargs`:

```python
return {
    "global_cb": getattr(context, "global_cb", None),
    "sub_device_id": getattr(context, "worker_sub_device_id", getattr(context, "sub_device_id", None)),
}
```

which is why the MLP ring matmuls have worked on silicon since attempt 1 while
the LM head could not. `LMHead2DConfig` had no equivalent, so it is added:
`decode_sub_device_id` / `prefill_sub_device_id`, callables defaulting to a
`_no_sub_device` that returns `None` — so the field is never `None` (the config's
`is_resolved()` requires that) and any caller that does not set it keeps ttnn's
present behaviour exactly.

20/20 `test_lm_head_2d.py` still pass (`logs2/a2_08_lm_head_host.log`).

## A note on the harness, for whoever runs next

Two additions to attempt 1's scripts, both in this directory:

* **`run_sequence.sh <manifest>`** — runs a list of device cycles strictly
  sequentially, one pytest on the mesh at a time, each followed by `cycle.sh`'s
  reap-and-reset. Manifest lines are `<deadline-seconds> <logname> <node-id>`.
  It exists so a chain of runs can be launched once and left alone; the
  per-run deadline is explicit because a 1-layer bring-up and an 80-layer
  teacher-forced run differ by more than an order of magnitude.
* `logs2/` for this attempt's logs, so attempt 1's `logs/` is never touched.

Attempt 1's three harness lessons all held. One new observation: after a
`TT_FATAL` inside a multi-sub-device program, `[stage] leave close model` can
print — the *model's* teardown completes — and the process still hangs, because
the hang is in the `mesh_device` **fixture** teardown after the test body
returns. So a clean-looking `close model` is not evidence the process will exit,
and every one of these runs still needed a reap.

`tt-smi -glx_reset` timing is not constant: the resets after runs 05 and 06 took
about two minutes each, the one after run 07 sat between `All 32 chips found` and
`Issuing POST_RESET` for several minutes. `after_device_run.sh` caps it at 600 s.
Do not interrupt one that is mid-flight.

## An infrastructure interruption after run 07

The `tt-smi -glx_reset` that `after_device_run.sh` runs after run 07 **timed
out**: it reached `Issuing POST_RESET on 32 devices` and never reached
`Re-initialized 32 boards`, so `after_device_run.sh` recorded `reset exit=1`
(`logs/reset_a2_07_decode_step.log`). The two resets before it, after runs 05
and 06, each completed in about two minutes; this one was killed at the 600 s
cap.

All 32 `/dev/tenstorrent` nodes remained present, so this is not attempt 1's
failure — no board left the PCIe bus. But a reset killed part-way through
`POST_RESET` leaves the boards half-initialised, and `tt-smi -ls` then stopped
returning (`logs2/a2_09_tt_smi_after_failed_reset.log`).

Recovery attempted, and the outcome, is recorded below. The house rule is a
maximum of two recovery attempts before `BLOCKED (infra)`; the failed reset above
is not counted as one, because it was routine post-run hygiene rather than a
response to a fault.

### The fault, named exactly

`logs2/a2_09_tt_smi_after_failed_reset.log` — `tt-smi -ls` exit 1 after about
five minutes of silence:

```text
Error in detecting devices!
ARC startup error at core 0-10 over NOC0: scratch_status=0xaabc,
  postcode=0xc0de0023, message_id=0xbc (Timed out after 300000 ms)
Location: /project/device/tt_device/wormhole_tt_device.cpp:404
  tt::umd::TTDevice::init_tt_device(...)
  tt::umd::TopologyDiscovery::get_connected_devices()
  tt::umd::TopologyDiscovery::create_ethernet_map()
```

**A chip's ARC firmware did not come up.** That is what the interrupted
`POST_RESET` left behind, and it also explains the five-minute silence: UMD waits
300 s for ARC startup before giving up. All 32 PCIe nodes are still present, so
this is *not* attempt 1's fault (a board leaving the bus) — the boards are there
and one or more of their management controllers is wedged.

An IPMI-level tray reset is the correct recovery for this and is what
`tt-smi -glx_reset` does. Recorded below as recovery attempt 1.

### Recovery attempt 1 — succeeded

```sh
timeout 900 tt-smi -glx_reset      # logs2/a2_10_recovery1_glx_reset.log
```

```text
recovery1 exit=0
Issuing POST_RESET on 32 devices after IPMI reset
Re-initialized 32 boards after reset
devices=32
```

and then, independently (`logs2/a2_11_tt_smi_after_recovery.log`):

```text
tt-smi -ls exit=0
32 boards listed, 0 occurrences of "Error in detecting devices" or "ARC startup error"
```

One recovery attempt, one success. The mesh was live again and device work
resumed at run 12. **Note the timeout mattered**: the reset that failed had 600 s
(`after_device_run.sh`'s cap) and this one had 900 s. On this machine a
`glx_reset` after a wedged ARC needs more than ten minutes, so
`after_device_run.sh`'s 600 s cap is too tight for the case where it is most
needed. That is a harness finding, not a hardware one, and it is the one change
the next session should make to these scripts.

## Result 12 — the LM head matmul runs; the collective is the next frontier

`logs2/a2_12_decode_step.log`. D-B13 is fixed: the 24-core `gather_in0` LM head
matmul **builds and executes**. The abort moved one op further, into the column
reduction:

```text
File ".../modules/lm_head/lm_head_2d.py", line 203, in _project
  reduced = collective(partial)
File ".../models/galaxy/collectives.py", line 113, in __call__
  return ttnn.all_reduce(
RuntimeError: TT_FATAL @ program.cpp:2205: num_intersections == num_cores
Kernel group cores do not match sub device cores for programmable core type TENSIX
backtrace:
  ttnn::prim::concat(...)
  ttnn::operations::data_movement::concat_impl(...)
  ttnn::concat(...)
  composite_common::composite_all_gather(...)
  ttnn::all_reduce(...)
```

### D-B14 — `ttnn.all_reduce` falls back to a composite that is not sub-device aware

`ttnn.all_reduce` forwards to the `all_reduce_async` overload that takes **no
persistent buffer and no semaphores** — `all_reduce.cpp` passes `std::nullopt`
for `barrier_semaphores`, `rs_global_semaphores` and `ag_global_semaphores`. That
overload falls back to `composite_common::composite_all_gather`, which calls
`ttnn::concat`. `concat` accepts an optional `sub_core_grids` and is handed none,
so it builds over the full compute grid — and the sub-device id **was** passed
correctly all the way down; it just does not reach the concat.

So D-B13's fix was necessary and not sufficient: `subdevice_id` is honoured by the
*fused* path and ignored by the *composite fallback*.

**Fix — the production recipe, followed exactly.** The decode LM head now uses the
persistent-buffer overload of `all_reduce_async` against its own keyed resource,
which is what `tt_ccl.line_all_reduce(..., lm_head=True, buffer_key="LM_HEAD")`
does in `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`. Three parts:

1. **A dedicated resource and buffer** (`plans.py`), keyed
   `("all_reduce", 1, (1, 1, 32, padded_local_vocab))`. It cannot borrow the
   axis-0 buffer: that one is sized for the `local_dim`-wide residual stream, and
   `all_reduce_async` validates
   `buffer_shard_volume >= output_shard_volume * ring_size` against the *logits*.
   (That validation rule is stated in a comment in the reference's own Qwen
   branch, which is the only place it is written down.)
2. **Staging onto 32 cores**, not the 24 the matmul ran on. On 24 cores a
   4-device axis-1 reduction needs a buffer shard a third larger, and it does not
   fit beside the resident decode activations. The reference reshards for exactly
   this reason — `LM_HEAD_OUT_RING_RESHARD_MEMCFG` is `(32, width // 32)` and
   `num_cores_after_lm_head = 32`, commented "Use 32 cores instead of 16 to
   reduce L1 memory usage per core".
3. **`fp32_dest_acc=True`.** This one is not a placement matter and I would not
   have guessed it. The reference's comment:

   > fp32 dest accumulation for the LM-head all_reduce only: its bf16
   > cross-device sum was order-dependent (ETH ring arrival order) -> per-row
   > logit non-determinism -> greedy flips.

   A bfloat16 cross-device sum of the logits is **not reproducible across runs**.
   That is precisely the failure mode this project's three-runs rule exists to
   catch, and it would have presented as intermittent greedy-decode
   disagreement, not as a crash. Reading the reference was worth more here than
   any number of device runs.

The reduced logits are placed back into the matmul's own placement before
returning, because `LMHead2D` uses one `output_memcfg` for the matmul, the
collective, the concat and the mask add. That costs two DRAM round trips per
token and belongs on the performance list.

## Result 14 — the reduction runs; the buffer must not be resident

`logs2/a2_14_decode_step.log`. D-B14's persistent-buffer reduction **built and
launched** — no composite fallback, no sub-device fault. It failed on L1
capacity instead:

```text
TT_THROW: Statically allocated circular buffers in program 250 clash with L1
          buffers on core range [1-0 - 3-9]. L1 buffer allocated at 392736 and
          static circular buffer region ends at 475392
```

### D-B15 — the LM-head all-reduce buffer was allocated resident in L1

The buffer is four times the width of the logits, because
`all_reduce_async` needs one slot per device on the reduced axis. At
`padded_local_vocab = 16128`, 32 cores and bfloat16 that is
`32 x 2016 x 2 = 129,024 B` **per core**, and I had allocated it as an ordinary
persistent resource — resident for the whole decode step, competing with 80
layers of activations on the same worker cores. The overflow is about 81 kB,
which is the same order as the buffer.

**The production code does not keep it resident, and it says so structurally
rather than in a comment.** Three separate places:

```python
# llama_ccl.py  -- created in DRAM
self.tt_lm_head_buffer = ttnn.from_torch(..., memory_config=ttnn.DRAM_MEMORY_CONFIG, ...)

# llama_model.py -- an L1 copy made only after the last layer, just before the LM head
if mode == "decode":
    self.tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(
        self.tt_ccl.tt_lm_head_buffer, self.tt_ccl.lm_head_buffer_mem_cfg)

# llama_ccl.py  -- freed immediately after the reduction
if lm_head:
    persistent_buffer.deallocate(True)
```

with a comment on the second one reading "Pre-allocated output of AllReduce in LM
Head to avoid memory cloberring".

So the fix is a lifetime fix, not a size fix: the keyed resource now allocates the
buffer in **`ttnn.DRAM_MEMORY_CONFIG`**, and the collective brings it into its L1
placement with `interleaved_to_sharded` for the duration of the call and frees the
L1 copy in a `finally`. `interleaved_to_sharded` runs on its *output* shard's
cores, so the materialisation is itself partition-safe.

This is worth stating as a general lesson, because the resource machinery in
`plans.py` invites the opposite: **"persistent" in `persistent_output_specs` means
"the resource owns it across calls", not "it must sit in L1".** For a buffer
sized by the vocabulary rather than by the hidden dimension, resident L1 is not
affordable, and DRAM residency with a per-call L1 view costs one extra
`interleaved_to_sharded` per token.

58 host tests pass after the change (`logs2/a2_15_host_quick.log`).

### D-B16 — the invalid-logits mask was narrower than the logits

Found by arithmetic rather than by a run, and fixed before the run that would
have hit it.

A `gather_in0` ring output is padded to the ring's tile alignment, so the decode
logits are **16128** columns per device while the module's mask is built at
`padded_vocab_size / 8 = 16032`. `_project` adds the two.

That the ring matmul's *logical* width is the padded one, not the logical vocab
width, is not a guess: `GalaxyDenseGeometry.decode_reduce_scatter_width` already
had to settle exactly this question for the MLP, and its comment records the rule —

> The ring matmul emits a 24-core aligned padded width. When that padding is not
> a whole number of shards the collective scatters the padded width; otherwise it
> scatters the logical width. This is the width the resource key is derived from,
> so it must match what TTNN reports for the scattered tensor.

For the MLP: `7168 % (7680 / 24) = 128`, not zero, so the padded 7680 is what
ttnn reports — and that resource key has been working on silicon since attempt 1.
For the LM head: `16032 % (16128 / 24) = 576`, not zero, so 16128. The same rule,
already validated, applied to the new op.

**Fix.** The mask is now built at the widest width any *mode output placement*
spans, rather than at `padded_vocab_size`. Both placements default to interleaved,
which has no shard spec, so a caller that does not place its logits keeps exactly
the old mask — Qwen included, until it wires the ring up. The extra columns are
padding like any other and are masked to `-inf` for the same reason the others
are; the statement the mask makes is unchanged, it just reaches further.

20/20 `test_lm_head_2d.py` still pass unmodified
(`logs2/a2_17_lm_head_host.log`), including the two that assert the mask's exact
metadata for a 152064-wide Qwen-like geometry.

## Result 16 — D-B17: the staging shard must be a whole number of tiles

`logs2/a2_16_decode_step.log`. The DRAM-resident buffer fixed D-B15 — no L1
clash — and the next constraint appeared:

```text
TT_FATAL: Physical shard shape (32, 504) must be tile {32, 32} sized!
  tensor_layout.cpp:168: !shard_align_error.has_value()
```

504 is 15.75 tiles. I had taken the production `num_cores_after_lm_head = 32`
as a constant, and it is not one: it works upstream because *their* padded width
is 16384 — 512 tiles, which 32 divides. Llama's ring-padded width here is 16128,
504 tiles, and 32 does not divide 504.

**Fix.** The core count is now searched for rather than named:
`lm_head_reduce_core_count(padded_local_vocab, available_cores)` returns the
largest count that fits the worker envelope *and* divides the width evenly in
tiles. Because the buffer is exactly `GALAXY_COLUMNS` times the width, one
divisibility condition covers the output shard and the buffer shard together.

Both Galaxy geometries then land on the same tile-aligned shard, which is a
reassuring sign the rule is the right one rather than a fit to Llama:

```text
                width  tiles  cores  shard         buffer shard   L1/core (bf16)
llama-3.3-70b   16128    504     42  384 (12 tiles) 1536 (48)      96 kB
qwen3-32b       19200    600     50  384 (12 tiles) 1536 (48)      96 kB
```

This is the second constant copied from the reference that turned out to be
geometry-specific — `num_global_cb_receivers` was the first. Both are now derived.

## Result 19 — hardware refuted an inference of mine, and D-B16 with it

`logs2/a2_19_decode_step.log`. D-B17's derived core count is fine — the tile
alignment passes — and the run got as far as the resource lookup:

```text
KeyError: 'no all_reduce resources for axis=1, geometry=(1, 1, 32, 16032), sequence=32'
```

**The logits' logical width is 16032, not the ring-padded 16128.** The ring
matmul's output *shard spec* over-covers its tensor — 24 cores x 672 = 16128
columns of spec for a 16032-column tensor — and `select_galaxy_resource` keys on
`tensor.shape`, which reports the logical width.

That contradicts the inference I drew from
`GalaxyDenseGeometry.decode_reduce_scatter_width`, whose comment says the
collective "scatters the padded width" and which is validated on silicon. Both
are true: a **reduce-scatter** output takes the padded width, a **matmul** output
keeps its logical width. I generalised one op's behaviour to another and hardware
said no.

Two consequences, and the second is a correction to this document:

1. The resource key is now `(1, 1, 32, local_padded_vocab_size)` — the logical
   width. The buffer's own spec stays at the physical `padded * GALAXY_COLUMNS`,
   because `all_reduce_async` validates *shard* volumes, not logical widths.
2. **D-B16 was not a defect and its fix was wrong.** The mask at
   `padded_vocab_size / 8 = 16032` matched the logits all along; widening it to
   16128 would have broken the add in the opposite direction. It is reverted, and
   the reasoning is now recorded in the code as a warning against exactly the
   generalisation I made.

D-B16 is struck. It is left in this document rather than deleted because the
mistake is the useful part: the two ops' width conventions differ, nothing says
so, and the only way to find out was to ask the mesh.

## Result 21 — D-B18: the reduction buffer's dtype, and why bfloat8_b is the right one

`logs2/a2_21_decode_step.log`. The resource key now resolves, the reduction is
reached, and ~30 programs report the same L1 clash at once:

```text
TT_THROW: Statically allocated circular buffers in program 862 clash with L1
          buffers on core range [5-6 - 6-7]. L1 buffer allocated at 355840 and
          static circular buffer region ends at 376768
```

about 21 kB, on four cores that are in *both* the 42-core reduce staging set and
the 24-core matmul ring — so they carry the ring matmul's circular buffers and the
reduction's L1 buffer view at the same time.

The arithmetic is unforgiving. The buffer is `GALAXY_COLUMNS` times the width of
the logits, so per core it is about `4 * 32 * (16032 / cores) * bytes`:

```text
             bfloat16   bfloat8_b
42 cores      96 kB       49 kB
50 cores      82 kB       41 kB
```

Spreading over more cores cannot get bfloat16 below ~82 kB, and the shortfall is
21 kB against a 96 kB allocation. **The dtype is the variable that matters, not
the core count.**

**bfloat8_b is not a concession here, it is the qualified value.** The production
Galaxy LM head calls `ttnn.linear(..., dtype=ttnn.bfloat8_b)` for both modes and
allocates `tt_lm_head_buffer` at bfloat8_b, and the accuracy gates this milestone
reuses — top-1 >= 91%, top-5 >= 99% from `tt_transformers` — were established
against exactly that. The Milestone B recipe's `decode_output_dtype =
decode_activation_dtype` (bfloat16) was an authoring choice that had never run.
`Llama33_70BGalaxyPrecision.lm_head_dtype` already declares bfloat8_b for the LM
head *weight*; the output now agrees with it, as a named field
(`lm_head_output_dtype`) rather than a borrowed one.

And the accumulation is unaffected: `fp32_dest_acc=True` means the cross-device
sum runs in fp32 regardless. Only the stored logits are bfloat8_b — the same
arrangement upstream ships. For reference, the upstream unit test for this exact
op reports `LMHead1D vs reference: 0.99987` at bfloat8_b, comfortably above this
job's PCC >= 0.99 gate.

**This is the one precision value attempt 2 changed, and it is called out here
rather than buried because changing precision to make something fit would
otherwise be exactly the wrong move.** The justification is that it moves *to*
the qualified value, not away from a gate; the accuracy measurement is the arbiter
and it is still to be taken.

## Result 23 — D-B19, OPEN: the axis-1 LM head reduction hangs on device

`logs2/a2_23_decode_step.log`, and the gdb dump in
`logs2/a2_23_hang_gdb_dump1.log`. D-B18's bfloat8_b logits removed the L1 clash —
the run got past it — and then **hung**. It did not abort; it stopped producing
output entirely, was reaped at the 900 s deadline, and needed a reset.

This is the first hang of attempt 2 and it was diagnosed before spending a
recovery attempt, as the house rules require.

**It is a device-side hang, not a host stall, and not a compile.** Established,
in this order:

1. `/proc/<pid>/stat` showed >100% CPU across 296 threads, so the process was
   running, not blocked;
2. no child processes, no new `.elf` artifacts and no JIT cache directories
   touched in three minutes, so it was **not** kernel compilation — which was the
   obvious benign explanation and is what the CPU load looks like;
3. per-thread CPU accounting named one thread burning 151 s of the 260 s;
4. `gdb -p <pid> -batch -ex "thread apply all bt"` on that thread and the main
   thread gave the answer:

```text
Thread 294 (the spinning one):
  tt::tt_metal::SystemMemoryManager::completion_queue_wait_front(...)
  tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue_event(...)
  tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()

Thread 1 (main):
  pthread_cond_wait
  tt::tt_metal::distributed::FDMeshCommandQueue::wait_for_outstanding_reads(...)
  tt::tt_metal::distributed::FDMeshCommandQueue::finish_nolock(...)
  tt::tt_metal::distributed::Synchronize(...)
  <- ttnn.synchronize_device, i.e. this collective's own `finally`
```

So: an enqueued device program **never signalled completion**, the completion-queue
reader spins on it forever, and the main thread waits on the reader. The `finally`
that blocks is `self.resources.synchronize("decode")` in
`GalaxyColumnAllReduce._persistent_all_reduce`, which means the op that hung is one
of the three that collective enqueues — the DRAM-to-L1 buffer materialisation, the
`all_reduce_async` itself, or the placement of the result back onto the ring.

**What was ruled out by reading, not by running:**

* **topology and link count.** The reference selects `Ring`/4 links for a 6U
  Galaxy and `Linear`/3 for 4U. This machine is 6U (`tt-smi` offers `-r` on
  Galaxy 6U; the reset does IPMI tray resets), and every other decode plan in
  `plans.py` uses `Ring`/4 on both axes and works, including three axis-1
  collectives. Not the cause.
* **semaphore count.** For decode the reference's `gather_semaphore_handles`
  holds **one** global semaphore per slot, which is exactly
  `semaphores_per_slot = 1`, the default this plan uses and the value the working
  axis-0 `all_reduce` uses. Not the cause.
* **buffer sizing.** `buffer_shard_volume >= output_shard_volume * ring_size`
  holds with equality — 32x1536 against 4 x 32x384 — which is also how the
  working axis-0 buffer is sized (32x1024 against 8 x 32x128). Not obviously the
  cause.

What remains are the two flags that differ from the axis-0 call proven on this
mesh (`use_optimal_ccl_for_llama`, which axis 0 sets and this does not, and
`fp32_dest_acc`, which this sets and axis 0 does not), and the possibility that
one of the two relocations rather than the reduction is the op that hung.

**Rather than guess between them, the collective is now instrumented.**
`_ccl_trace` prints and flushes a name before each device op it enqueues, gated on
`TTTV2_GALAXY_CCL_TRACE` so it costs nothing when unset, and `run_sequence.sh`
exports it. A hang leaves no traceback and no further log output, so naming each
op before entering it is the only way to turn "one of three ops" into "this op" —
and it turns one device run into a fact instead of a coin toss.
