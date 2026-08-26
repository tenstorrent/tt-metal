# Agent Brief — Gap 2: `Prefetcher2D` / Galaxy CCL hardware qualification

You are closing one named Milestone A coverage gap on a Wormhole Galaxy. Work unsupervised; every
decision that can be made in advance is made below.

## Mission

`Prefetcher2D` and the Galaxy CCL/resource owner have extensive host coverage against mock meshes and
are exercised **incidentally** on silicon as a side effect of MLP2D and one RMSNorm2D test. Nothing
tests their own lifecycle contract on real hardware, and in particular **no device test has ever
performed a prefill↔decode mode transition.** Close that.

## Why this is a gap

The repo says so itself, in
[models/common/modules/README.md:212](../models/common/modules/README.md#L212):

> Integrated Prefetcher2D/Galaxy-resource ownership has host coverage but **is not yet qualified on
> hardware**.

And `tttv2_2d_modules_plan.md`, "Galaxy operation-boundary activation", mandates the exact matrix
that is missing:

> Compilation, capture, replay, and cleanup tests must include transitions:
> decode to prefill; prefill to decode; repeated prefill; repeated decode; failure during
> transition; cleanup from either active mode.

The Milestone A exit gate also requires every module to "pass ownership/cleanup and repeat-invocation
tests" and to "demonstrate that prefetch/CCL/static strategy is resolved before the hot path".

## Current state — this is the whole picture, verified

### Host coverage is mock-only

[models/common/tests/modules/prefetcher/test_prefetcher_2d.py](../models/common/tests/modules/prefetcher/test_prefetcher_2d.py)
is 446 lines built entirely on `FakeMesh` (:14), `FakeTensor` (:55), `ResourceHarness` (:71) and
`MagicMock` (:6). Zero silicon. It covers, against mocks:

| Test | Line |
| --- | --- |
| `test_initialize_creates_only_both_managers_and_is_idempotent` | :198 |
| `test_registration_is_ordered_borrowed_and_compatibility_validated` | :209 |
| `test_seal_loads_decode_manager_before_global_cb_allocation` | :239 |
| `test_default_registration_rejects_duplicate_buffer_and_other_mesh` | :251 |
| `test_seal_derives_cb_size_and_rejects_undersized_configuration` | :262 |
| `test_seal_is_transactional_and_retryable_after_metadata_failure` | :278 |
| `test_activation_starts_stops_and_releases_repeat_results` | :293 |
| `test_borrow_context_requires_exact_sealed_subdevice_policy` | :333 |
| `test_failed_activation_rolls_back_mode_and_running_prefetch` | :357 |
| `test_failed_stop_preserves_active_session_ownership` | :373 |
| `test_failed_stall_transition_restores_previous_mode_without_publishing_target` | :389 |
| `test_cleanup_is_idempotent_releases_owned_results_and_never_weights` | :406 |
| `test_cleanup_continues_after_failure_and_remains_idempotent` | :429 |
| `test_context_manager_cleans_up_on_failure` | :440 |

`models/common/tests/models/galaxy/{test_ccl,test_resources,test_prefetcher_resources_composition}.py`
are host-only in the same way.

### A real prefetcher *is* built on silicon

[models/common/tests/modules/_wh_galaxy_hardware.py:188-263](../models/common/tests/modules/_wh_galaxy_hardware.py#L188-L263),
`_create_hardware_prefetcher`, constructs a genuine `Prefetcher2D`: 12 real sender cores plus 8 dummy
senders, real receiver `CoreRangeSet`s, `global_cb_size=728*1088`, an L1 height-sharded
`address_memory_config` on the sender cores, then `initialize()` → `register_weight()` → `seal()`.

Consumers today:

| Device test | Resource helper | Prefetcher? |
| --- | --- | --- |
| `test_mlp_2d_wh_galaxy.py:509` (decode) | `require_galaxy_hardware_resources` | **real** |
| `test_mlp_2d_wh_galaxy.py:588` (prefill) | `require_galaxy_hardware_resources` | **real** |
| `test_rmsnorm_2d_wh_galaxy.py:244` | `require_galaxy_hardware_resources` | **real** |
| `test_rmsnorm_2d_wh_galaxy.py:205` | `require_galaxy_ccl_hardware_resources` | none (CCL-only) |
| `test_attention_2d_wh_galaxy.py:1106` | `require_galaxy_ccl_hardware_resources` | **none** |

So the happy path — initialize, register, seal, consume — does run on hardware.

### What has never run on hardware

**Every device test pins one mode for its entire lifetime.** Look at `_invoke` in
[test_mlp_2d_wh_galaxy.py:462-482](../models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py#L462-L482):

```python
resources.activate(mode)
output = module(device_input, mode=mode)
resources.synchronize(mode)
```

`mode` is a constant per test. `test_mlp_2d_wh_galaxy_decode_batch_32_repeat` only ever passes
`mode="decode"` (:554); `test_mlp_2d_wh_galaxy_prefill_128_then_2048_repeat` only ever passes
`mode="prefill"`. The `GalaxyResourcesConfig` carries both plans, and both prefetch contexts are
handed to `MLP2DConfig` — but only one is ever activated.

Consequently, on real silicon, these have **never** executed:

- decode → prefill transition
- prefill → decode transition
- repeated transitions in one process
- failure during a transition (the rollback paths at `test_prefetcher_2d.py:357/373/389`)
- cleanup from an active decode mode, or from an active prefill mode
- a second `Prefetcher2D` on the same mesh in the same process after a first one was cleaned up

`activate()` performs `load_sub_device_manager` + `set_sub_device_stall_group`
(see `_CCLOnlySubdeviceOwner.activate`,
[_wh_galaxy_hardware.py:114-119](../models/common/tests/modules/_wh_galaxy_hardware.py#L114-L119)),
and the real `Prefetcher2D.activate`
([prefetcher_2d.py:370](../models/common/modules/prefetcher/prefetcher_2d.py#L370)) additionally
starts and stops the DRAM prefetch producer. None of that is observable through a `FakeMesh`.

## Why this gap is higher-risk than it looks

Both Milestone A defects root-caused on 2026-08-25 were **L1 address/ownership faults that presented
as intermittent passes or hangs, never as clean failures.** Read the last two checkpoints in
`tttv2_2d_modules_work_log.md` in full before starting. In summary:

1. **Fused RMS stats** — `fused_rms_minimal` binds its stats circular buffer to the stats tensor's L1
   address on the norm grid's *first core*. The test placed stats on `x=1` while the grid started at
   `x=2`, so the kernel reduced whatever the allocator had left there. It passed in some processes
   and failed in others with identical seeds. One row recorded as PASSED in the 08-24 evidence run
   was reading plausible aliased L1.
2. **Fused QKV all-reduce** — `galaxy_mode_plan` narrowed `semaphore_cores` below the worker
   subdevice, so `all_reduce_create_qkv_heads` placed a sender on `(0,0)` where the semaphore address
   was never reserved or zeroed. The collective polled uninitialized L1 forever. Same
   pass-or-hang-depending-on-residual-L1 signature.

A leaked subdevice manager, a stall group left set across a mode switch, or a global CB whose address
metadata survives a cleanup is **the same class of defect**. It will not fail loudly the first time.
This is precisely what mocks cannot see and precisely why this gap matters.

**Operational consequence: run every new test at least three times in fresh processes before
recording it as evidence.** A case that flips across processes is a defect, not noise.

## What to build

New file: `models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py`.

Use the **MLP2D geometry as the payload** — it is the qualified one, and it already has a working
prefetch wiring you can lift from
[test_mlp_2d_wh_galaxy.py](../models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py)
(`_resources_config`, `_decode_ring_config`, `_weight_lazies`, `_prefill_weight_lazies`,
`galaxy_prefetch_decode_mode_plan` for decode, `galaxy_mode_plan` for prefill). Prefer importing or
factoring those helpers over re-deriving the geometry; a wrong core grid here costs hours.

Note the shared-helper precedent already set on 2026-08-25: cross-suite test plumbing lives in a
shared module (`models/common/tests/modules/_hf_reference.py`) and both suites import it. If you
factor MLP helpers out, follow that pattern and keep the MLP suite green.

### Test 1 — sealed resources are real on device

After `initialize()` → `register_weight()` ×N → `seal()` on the real mesh:

- `prefetcher.sealed` is `True`, `initialized` is `True`;
- `context("decode")` and `context("prefill")` both return a `Prefetcher2DContext` with a non-`None`
  `global_cb`, `weight_addresses`, and `weight_address_metadata`;
- read back `weight_addresses` and assert each entry matches the corresponding registered tensor's
  actual device buffer address, in registration order — the addresses are the whole point of sealing
  and nothing currently proves they are correct on silicon;
- the address tensor is placed on the 12 sender cores per `address_memory_config`;
- `expected_weight_count` and `address_repeat_count` invariants hold
  ([prefetcher_2d.py:__post_init__](../models/common/modules/prefetcher/prefetcher_2d.py#L76-L100)).

### Test 2 — the mode-transition matrix (the core of this brief)

One process, one sealed `Prefetcher2D`, one `MLP2D` instance. Drive:

```
decode → prefill → decode → prefill → decode
```

repeated at least twice, running **one real MLP2D invocation with a PCC assertion at every step**.
The PCC is what proves the context is actually usable after the transition rather than merely
switched. Reuse the MLP reference (`_reference_mlp` + `get_mlp_weights_from_ref_model` from
`_hf_reference.py`) so the numbers are comparable with the qualified MLP evidence.

Assert at each step:

- the correct subdevice manager is loaded and the stall group matches the target mode's plan;
- PCC ≥ 0.99 against the HF reference;
- `resources.synchronize(mode)` completes.

Also cover **repeated same-mode activation** (`decode → decode`) — the plan lists "repeated prefill;
repeated decode" separately, and an activate that is not idempotent will show up here.

### Test 3 — cleanup from each active mode, and the leak detector

Two cases: clean up while **decode** is the active mode, and clean up while **prefill** is active.

The real assertion is not that `cleanup()` returns — it is that the mesh is genuinely free
afterwards. So in the same process, after cleanup:

1. construct a **second** `Prefetcher2D` on the same mesh, `initialize()` / `register_weight()` /
   `seal()` it, activate a mode, and run one MLP2D invocation to a passing PCC.

If a subdevice manager, stall group, or global CB leaked, this second construction is where it
surfaces. Nothing in the suite does this today.

Also assert on device what the host suite asserts against mocks:

- `cleanup()` is idempotent (call it twice, no error);
- registered **weights are never released by cleanup** — after cleanup the weight tensors must still
  be allocated (host equivalent: `..._never_weights` at `test_prefetcher_2d.py:406`);
- the `__enter__`/`__exit__` context-manager path
  ([prefetcher_2d.py:452](../models/common/modules/prefetcher/prefetcher_2d.py#L452)) cleans up when
  the body raises.

### Test 4 — registration and sealing rejections on real tensors

Port the host rejection contracts to silicon, where the tensors are real:

- duplicate buffer registration is rejected;
- a tensor belonging to a different mesh is rejected;
- `seal()` with an undersized `global_cb_size` is rejected (host: `:262`);
- `borrow_context` with a subdevice policy that does not match the sealed one is rejected
  (host: `:333`; the device analogue matters because a mismatch that the mock rejects cleanly could
  corrupt real subdevice state).

Keep these cheap — they should abort before any kernel runs.

### Test 5 — Attention2D with an active prefetch producer (attempt, and report honestly)

Attention2D is the one module that has **never** run alongside a prefetcher: it uses
`require_galaxy_ccl_hardware_resources` only. Production will run it with prefetched weights, so this
is a real qualification hole.

Attempt one Attention2D decode case using `require_galaxy_hardware_resources` +
`galaxy_prefetch_decode_mode_plan`.

**Expect friction, and understand why before you start.** With `galaxy_prefetch_decode_mode_plan`
([_wh_galaxy_hardware.py:306-342](../models/common/tests/modules/_wh_galaxy_hardware.py#L306-L342))
the worker subdevice is narrowed to `worker_cores` = `x ∈ {1,2,3} ∪ {5,6}` and `semaphore_cores` is
set to the same range — those two are consistent, which is the invariant the 08-25 attention fix
established. But the Attention2D decode QKV/WO matmuls use a `(7,1)` grid spanning `x = 0..6`, which
overlaps the prefetch **sender** cores at `x=0` and `x=4`. That is exactly why the attention test
could not narrow its subdevice and had to keep the full-grid `galaxy_mode_plan` instead.

So this case may require a geometry change to the attention decode matmul grid, or may be infeasible
without one. **If it is infeasible, that is a finding to write up, not a thing to force.** Report
precisely what conflicts with what, and recommend whether it belongs in Milestone A or in the
Milestone B model integration where the production grids are chosen anyway.

Do **not** paper over a hang here by narrowing `semaphore_cores` — read the docstring at
[_wh_galaxy_hardware.py:281-291](../models/common/tests/modules/_wh_galaxy_hardware.py#L281-L291),
which records that invariant explicitly after it cost a full evidence run.

## If you find a defect

Report it. Root-cause it if you can do so without guessing. A module fix is permitted **only** if you
can demonstrate the defect with a failing test first and show the fix on hardware across ≥3 fresh
processes. Never change a threshold or a core grid to make a failure disappear without explaining
why the new value is the correct one.

Distinguish carefully between "the module is wrong" and "the test's resource plan is wrong" — both
2026-08-25 root causes turned out to be the latter in part, and the write-ups are worth imitating.

## Run procedure

Host: the reserved 6U WH Galaxy, 32 devices. Confirm before starting:

```sh
ls /dev/tenstorrent | wc -l      # must be 32
tt-smi -ls
pgrep -af 'pytest|ttnn' | grep -v grep     # must be empty
```

If fewer than 32 device nodes are present, stop and report; do not run a reduced mesh.

**Exactly one pytest process may touch the device at any moment.** Never put pytest in a shell
pipeline — the pipeline can return control while the nested process still holds the device.

```sh
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The agent harness caps a foreground tool call at 600 s, shorter than the 2700 s bound. If a run needs
longer, issue it as a tracked background process and block on its exit before the next one. Re-check
`pgrep` between runs.

On a hang or crash: kill the process tree, confirm the device is free, `tt-smi -glx_reset`, confirm
`Re-initialized 32 boards`, retry. **Maximum 2 recovery attempts**, then record `BLOCKED (infra)`
with logs and move on.

A hang in this work is most likely a subdevice/semaphore ownership fault. Before spending a recovery
attempt, capture a traceback: the 08-25 attention debugging used a repeating
`faulthandler.dump_traceback_later` pytest plugin (diagnostic only, never committed) and located the
stall in two dumps 90 s apart. Do the same rather than guessing from the outside.

Keep every log from every attempt. Never overwrite a log.

## Regression gates

Because you may factor MLP helpers into shared plumbing, the MLP and RMSNorm device suites must stay
green. Before recording your own evidence:

```sh
# host
python -m pytest models/common/tests/modules/prefetcher/test_prefetcher_2d.py \
                 models/common/tests/models/galaxy \
                 models/common/tests/modules/mlp/test_mlp_2d.py -q
pre-commit run --files <every file you touched>

# device, one process at a time
models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py           # expect 4 passed
models/common/tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py   # expect 8 passed
```

The RMSNorm and Attention device suites were fixed on 2026-08-25 and their current expected results
are `8 passed` and `2 passed` respectively — see the last two work-log checkpoints. If you see
anything else, stop and report before continuing; you may have perturbed a shared helper.

## Hard prohibitions

- Do not modify any `models/common/modules/**/*_1d.py` implementation file.
- Do not narrow `semaphore_cores` for a generic async CCL to make a hang go away. That defect is
  documented; reintroducing it silently would invalidate the attention evidence.
- Do not relax a PCC threshold or a tolerance to turn a failure green.
- Do not `git commit`, `push`, `checkout`, `stash`, or `reset`. Leave the tree dirty for review.
- Do not rebuild tt-metal or recreate the venv.
- Do not run the full `models/common/tests` suite or any 1D hardware matrix.
- Do not claim a result you did not observe. `BLOCKED (infra)` with logs is an honest outcome;
  an invented pass is not.

## Deliverables

1. `models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py`.
2. Any shared helpers factored out of the MLP suite, with the MLP suite still green on device.
3. An evidence directory `tttv2_milestone_a_gap2_evidence/` with one raw log per run (including the
   ≥3 repeats and any failures/resets) plus `REPORT.md` covering: node IDs, results, the transition
   matrix actually executed, PCC at each transition step, cleanup/leak-detector outcomes, the
   Test 5 attention-with-prefetcher finding, and anything left open.
4. A `## Hardware checkpoint: Prefetcher2D and Galaxy resource hardware qualification <ISO date>`
   section appended to `tttv2_2d_modules_work_log.md` in that file's terse bullet style.
5. Proposed replacement text for the `Prefetcher2D` and `Galaxy CCL/resources` rows of
   `models/common/modules/MILESTONE_A_STATUS.md`, and for
   [README.md:212](../models/common/modules/README.md#L212) if the hardware-qualification caveat can
   now be dropped. Put the proposed text in your report; do not edit those files — they are being
   rewritten wholesale as a separate task.

## Finish condition

Every test you added has a terminal state with at least one log; the transition matrix named in the
plan (decode→prefill, prefill→decode, repeated prefill, repeated decode, failure during transition,
cleanup from either active mode) is either covered on hardware or has a documented reason it could
not be; the MLP/RMSNorm device suites still show their expected results; `REPORT.md` and the work-log
section are written; the device is clean (`tt-smi -ls` showing 32 boards). Print the absolute path of
`REPORT.md` last.

If you finish early, do not invent extra work. Stop.
