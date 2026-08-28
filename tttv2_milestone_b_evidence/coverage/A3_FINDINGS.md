
## Findings, attempt 3

Attempt 1's seven and attempt 2's three stand as §A2 leaves them, except where
this section says otherwise. Attempt 3 escalates one, adds one, and closes one.

### D-C5 — **escalated**: the column user selector cannot accept *either* model's decode logits

§A2 records D-C5 as a Qwen failure. It is not model-specific, and the reason is
visible on the host without opening the mesh.

`GalaxyColumnUserSelector.__call__` (`models/common/models/galaxy/collectives.py:445`)
is one `ttnn.matmul(selector, tensor)`. The default multi-core matmul program
config requires **input B interleaved** (`matmul_device_operation.cpp:1233`). The
tensor it is handed is whatever `model.decode_forward` returned, and that comes
from `LMHead2D.decode_forward` with `decode_output_memcfg`, which both models set
from the *shared* Galaxy recipe:

```python
# models/common/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/model.py
decode_output_memcfg=decode.lm_head_output_memcfg
# models/common/models/galaxy/recipes.py:889
lm_head_output_memcfg=width_sharded_memory_config(padded_local_vocab, ring)
```

Resolved on the host for both geometries (`logs3/a3_h6_decode_placements_probe.log`):

| | Llama-3.3-70B | Qwen3-32B |
| --- | --- | --- |
| `lm_head_output_memcfg` layout | **WIDTH_SHARDED**, L1, 24 cores, shard `(32, 672)` | **WIDTH_SHARDED**, L1, 24 cores, shard `(32, 800)` |
| `residual_memcfg` cores | 16 | 10 |
| `local_dim` | 2048 | 1280 |
| LM-head all-reduce cores | 42 | 40 |

So the selector is fed a width-sharded tensor for **both** models, and the
`TT_FATAL` attempt 2 saw for Qwen is reachable for Llama by exactly the same route.
The only reason no log showed it for Llama is that Llama's demo path dies of the
L1 address clash at its second prefill, *before* it ever reaches the sampler
(`a2_g11`). Two independent faults, one hiding the other.

**Measured, not inferred.** The paragraph above was written from the host probe
at 08:06Z; `a3_l_greedy` then ran the Llama step-7 greedy case directly and
closed it on silicon (`logs2/a3_l_greedy.log`, `1 failed in 886.53s`):

```text
models/common/models/galaxy/direct_runner.py:527: in decode_sampled
models/common/models/galaxy/collectives.py:445: in __call__
E   RuntimeError: TT_FATAL @ matmul_device_operation.cpp:1233
E   MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED,
E   got: TensorMemoryLayout::WIDTH_SHARDED
```

Same frame, same assertion, same line as Qwen's `a3_q_greedy`. Both models, and
on the **step-7** path rather than a demo, so the fault is not an artefact of
the demo's two-phase shape either. D-C5 is a two-model, two-entry-point,
shared-code defect.

**And the fix has a precedent in the same file.** `collectives._relocate_sharded`
(line 122) already stages through `ttnn.sharded_to_interleaved(tensor,
ttnn.DRAM_MEMORY_CONFIG)` and documents *why* that op and not
`to_memory_config`: it runs on its input's own `shard_spec.grid`, so it stays
worker-confined under a loaded sub-device manager, whereas a generic reshard
builds over the full compute grid and is illegal there. So the one-line fix for
the selector is not a guess — it is the op two hundred lines above it, chosen
for exactly this constraint. Attempt 3 tested that claim on hardware rather
than asserting it; see `test_{qwen,llama}_device_sampling_claims_behind_dc5_with_interleaved_logits` in the area-4 table.

**Why this is a 2D-module finding and not a model one.** Both the selector
(`collectives.py`) and the LM head placement (`recipes.py`) are shared Galaxy code.
The selector's only guard is a shape check:

```python
if len(shape) != 4 or shape[-2] != self.max_batch_size:
    raise ValueError(f"column user selection expects [1, 1, {self.max_batch_size}, W], got {shape}")
```

Memory layout is unvalidated, so the incompatibility surfaces as a `TT_FATAL`
thrown from inside `ttnn` rather than as a contract error naming the caller — and
it surfaces only when someone composes the LM head with the sampler on a real
model.

**And that is the composition gap.** The selector *does* have a device test,
`models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py`,
including one called `test_column_user_selector_feeds_sampling_2d`. It builds its
input with

```python
memory_config=ttnn.DRAM_MEMORY_CONFIG        # interleaved
```

which is the one layout the matmul accepts and the one layout the real model never
produces. Every module in the chain is green in its own suite; the chain is broken.
This is precisely the class of defect the plan's per-module contracts cannot catch
and the reason step 7 exists.

**Consequence for the exit gate.** Everything in the brief's area 4 —
greedy-vs-host-argmax, the padded-vocabulary claim, seeded slot stability, the
near-zero-temperature check for defect D4, per-slot heterogeneous controls — is
behind `sample_decode`, hence behind this one matmul, for both models. See the
area-4 table for what that measured out as.

**What it needs**, and none of it is this job's to do: either the selector accepts
a sharded input B (a `sharded_to_interleaved` at the boundary, or a matmul program
config that takes width-sharded in1), or `sample_decode` declares the layout it
requires and each model relocates before calling. Both are runtime changes to
shared code. Reported, not made.

### D-C7 — **new**: closing a model does not return its L1, and the second model in a process cannot start

This is the finding attempt 3's second half was told to look for, and it is the
one that changes §A3's L1 story.

`a3_q_two_pools` (`logs2/a3_q_two_pools.log`, commit `2061c126743`, `1 failed,
2 warnings in 571.29s`) builds **Qwen** twice in one process, once per paged
pool, each inside its own `try/finally` that runs

```python
def _close(handle):
    try:
        handle.close()
    finally:
        del handle
        gc.collect()
```

The first pool completed — `[pool] default-2048: block_size=32
max_num_blocks=2048` at log line 331, a full prefill of 32 rows and a decode,
then `close()` and an explicit `gc.collect()`. The second model then **loaded
successfully** (`[pool] explicit-4096: block_size=32 max_num_blocks=4096`, line
11798) and died at its first decode:

```text
models/common/models/galaxy/direct_runner.py:543: in _decode_device_logits
    self.model.activate("decode")
models/common/models/galaxy/resources.py:363: in activate
    self._prefetcher.activate(mode)
models/common/modules/prefetcher/prefetcher_2d.py:431: in activate
    self._ensure_global_cb(context)
...
E   RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:462
E   Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70
E   banks, where each bank needs to store 792064 B, but bank size is 1393472 B
E   (allocated: 923776 B, free: 469696 B, largest free block: 373824 B)
```

**Read the numbers.** At the moment the second model asks for its global
circular buffer, **923776 of 1393472 bytes per L1 bank — 66% — are still
allocated**, with a largest free block of 373824 B against the 792064 B needed.
The first model had been closed *and* garbage-collected. One model alone fits:
`a2_g17`, `a2_g18`, `a2_L1_qwen_repeat_run2/3` and `a2_L1_qwen_batch32_run2/3`
all create exactly one Qwen model and all create the global CB without
complaint, 6/6.

**Why it is a finding and not a restatement of L1.** `Prefetcher2D.cleanup()`
already does everything Python can do: it stops the prefetch, deallocates every
retained resource, sets `self._global_cb = None` and clears `self._contexts`.
`mb-llama` attempt 3 showed that dropping the *last* reference does not return
the buffer's L1 mid-process. This measures the stronger statement — **the L1 is
not returned by full model teardown either**, and quantifies what is left behind.
Milestone A's limitation L1 is written as a prefill-after-decode ordering
problem; this says the residue outlives the owner entirely, which is a lifetime
problem, not an ordering one. No teardown ordering the brief suggests can fix a
buffer that the destructor of a closed object did not free.

**Why it matters more than the Llama clash.** §A3's L1 section, written earlier
in this attempt, concluded that "the address clash is Llama-only at this tree"
and offered Qwen as "a working reference configuration". That is still true of
the two shapes it was measured on, and it is **not** true of L1 in general:
Qwen hits L1 too, in the shape the brief names third — *repeated model
construction and teardown in one process* — with a capacity signature instead of
an address one. The corrected statement is in "L1, corrected" below.

**Consequence for the exit gate.** Area 1's headline claim, "paged fill during
prefill then decode reading the same blocks, PCC ≥ 0.99 against the contiguous
path", was already **not expressible** through the adaptor (D-C4). Its nearest
reachable substitute — two *different* paged pools compared against each other —
needs two models, and D-C7 says a process gets one. Attempt 3's answer is to
compare across processes; see "Area 1" below.

### F-C3 — the model-named import gate is not literally zero over `models/common`

§A2 reported "0 matches" for the brief's "no dependency imports from an existing
model-named implementation package" line. Attempt 3 widened the grep to
`models/common/tests/modules` and found one:

```text
models/common/tests/modules/moe/test_tt_moe_decode.py:33
    from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.test_optimized_moe_decode_block import (
        create_torch_dispatch_input_expert_scores_tensor,
        create_torch_dispatch_input_tensor,
        verify_output,
    )
```

It is **not Milestone B's**: it exists byte-identically at the job-0 base
`bc6ad03bfc2`, was added upstream by `b705bc150e5` ("MoE: (towards) a
configurable e2e decode module (#45041)"), is a *test* importing test helpers,
and is on no Galaxy import path — `git diff --name-only bc6ad03bfc2..HEAD` does
not contain the file at all. Milestone B's own verdict on this gate is a clean
**PASS**. But `mb-signoff` should state the exception rather than assert a bare
zero over `models/common`, because the next person to run the grep will find it
and will not know it is pre-existing.

### D-C8 — **new**: behind D-C5 the selector matmul violates the loaded decode sub-device's core set

This is why the diagnostic was worth a Galaxy quarter-hour.

`a3_q_dc5` (`logs2/a3_q_dc5.log`, commit `152d4c49efb`, `1 failed, 2 warnings in
156.06s`) relocated the decode logits exactly as D-C5's proposed one-line fix
would. **Three fresh processes, byte-identical**: `a3_q_dc5` 156.06s,
`a3_q_dc5_run2` 157.88s, `a3_q_dc5_run3` 154.84s, each printing the same
relocation line and raising the same `TT_FATAL`. On this hardware a passing run
proves nothing, and the same rule applied to a failure says the opposite: this
is not a race and not aliased L1, it is a function of the resolved placement.

```text
[dc5] greedy: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200;
      relocated to TensorMemoryLayout.INTERLEAVED
```

The `INTERLEAVED` assertion is gone; the call gets **further into the same
function** and then dies:

```text
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py:822: in sample
models/common/models/qwen3_32b_galaxy/model.py:1810: in sample_decode
models/common/models/qwen3_32b_galaxy/model.py:1793: in select_decode_column_users
models/common/models/galaxy/collectives.py:445: in __call__
E   RuntimeError: TT_FATAL @ tt_metal/impl/program/program.cpp:2205:
E                 num_intersections == num_cores
E   info:
E   Kernel group cores do not match sub device cores for programmable core type TENSIX
```

`collectives.py:445` is the same line as D-C5 — the bare `ttnn.matmul` — and the
new failure is one layer down: the program the matmul builds spans cores that are
**not in the loaded decode sub-device's core set**. Decode runs under a
sub-device manager (`Prefetcher2D._configure_mode`); a default multi-core matmul
program config resolves its grid from the tensors and the full compute grid, not
from the loaded sub-device, so the two disagree and `program.cpp` refuses the
program.

**So D-C5's fix is not one line, and the file already says why.** Two hundred
lines above the selector, `collectives._relocate_sharded` documents this exact
hazard for a *different* op:

> a direct `to_memory_config` between two shard specs that differ in grid **and**
> width resolves to `reshard_program_factory_generic`, which builds over the full
> compute grid and is illegal under a loaded sub-device manager.
> `sharded_to_interleaved` runs on its input's `shard_spec.grid` and
> `interleaved_to_sharded` on its output shard's cores, and both of those are
> worker-confined here.

The relocation was chosen for worker-confinement and it *is* worker-confined —
that part of the fix works, and `a3_q_dc5` is the hardware evidence. What is not
worker-confined is the **matmul that consumes it**. Making the selector's input
interleaved satisfies the matmul's memory-layout precondition and simultaneously
hands it a placement decision it makes over the wrong grid.

**The reduction, for whoever owns this.** `GalaxyColumnUserSelector` needs *both*:

1. an input B the matmul accepts — interleaved, or a program config that takes
   width-sharded in1; **and**
2. a program config whose core grid is inside the decode worker sub-device, the
   way every other decode-time op in `collectives.py` is.

Neither is expressible from a test, and (2) is the one no amount of relocation
reaches. `GalaxyColumnUserSelector.__init__` already accepts a
`compute_kernel_config` and a `memory_config` and passes both to the matmul; it
accepts no `program_config`, and nothing in it knows which sub-device is loaded.

**And this is the third fault in one stack of three.** The L1 address clash hid
D-C5 for Llama; D-C5 hid D-C8 for both models. The class's own docstring predicted
it in as many words —

> **Unqualified.** This composition has never run on a Galaxy mesh. Qualify it
> with the focused selector test before trusting a device sampling path built on
> it; the alternative is composing the logits to host and calling
> `Sampling2D.sample_host`.

— and the focused selector test it points at
(`test_column_user_selector_wh_galaxy.py`) builds its input with
`memory_config=ttnn.DRAM_MEMORY_CONFIG` **and no loaded sub-device manager**, so
it cannot see either fault. The alternative the docstring offers — compose to host
and call `sample_host` — is what both demos' passing half actually does, and it is
the only sampling path this tree has that works on a Galaxy.

**Consequence for the exit gate.** Area 4 is **BLOCKED**, for both models, by two
stacked defects in shared Galaxy code. Not "unmeasured": measured, twice, with the
first blocker removed at the call site to reach the second. Milestone B's device
sampling does not work end to end on this hardware at this tree, and the report
should say so in those words.
