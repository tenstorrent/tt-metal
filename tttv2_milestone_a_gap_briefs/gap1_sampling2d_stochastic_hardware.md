# Agent Brief — Gap 1: `Sampling2D` stochastic hardware coverage

You are closing one named Milestone A coverage gap on a Wormhole Galaxy. Work unsupervised; every
decision that can be made in advance is made below.

## Mission

`Sampling2D` has exactly one hardware test, and it only exercises the **greedy** path. Add real
`(8, 4)` WH Galaxy coverage for the **stochastic** paths — seeded and unseeded — and record the
evidence. You are adding tests to already-shipped module code; you are not expected to change
`sampling_2d.py`, but you may if you find a real defect (see "If you find a defect").

## Why this is a gap

`tttv2_2d_modules_plan.md`, the `Sampling2D` contract:

> - top-k, top-p, temperature, seed, and forced argmax remain per-call values;
> - deterministic seeded requests remain slot-stable;
> - **greedy, seeded stochastic, and unseeded stochastic paths are covered**;

`models/common/modules/MILESTONE_A_STATUS.md` already concedes it —
"Qualified for the required forced-argmax hardware case; **stochastic hardware is not recorded**."

## Current state — read these first, they are the whole picture

### The one existing device test

[models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy.py](../models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy.py)
— 75 lines, one test, `test_sampling_2d_wh_galaxy_exact_padded_vocab_exclusion`. It calls:

```python
sampler.decode_forward(tt_logits, top_k=32, top_p=1.0, temperature=0.0, seed=7, forced_argmax=True)
```

Now look at `_update_call_buffers` in
[models/common/modules/sampling/sampling_2d.py:198-216](../models/common/modules/sampling/sampling_2d.py#L198-L216):

```python
force_greedy = call.forced_argmax[index] or call.temperature[index] == 0.0
k_values[slot]           = 1   if force_greedy else call.top_k[index]
p_values[slot]           = 0.0 if force_greedy else call.top_p[index]
temperature_values[slot] = 1.0 if force_greedy else call.temperature[index]
```

The existing test sets **both** `forced_argmax=True` and `temperature=0.0`, so every slot collapses
to `k=1, p=0, temp=1`. `ttnn.sampling` never takes its stochastic branch, the `top_p` nucleus code
never runs, and `seed=7` is inert because a `k=1` draw has one candidate. **The entire stochastic
path of `Sampling2D` has never executed on silicon.**

### The device execution path you are qualifying

[sampling_2d.py:289-387](../models/common/modules/sampling/sampling_2d.py#L289-L387), `decode_forward`:

1. optional typecast to bf16;
2. `ttnn.add(logits, invalid_vocab_mask)` — **before** topk (lines 333-341);
3. `ttnn.topk(k=max_top_k=32)` per vocab shard, on `sub_core_grid_topk`;
4. two `ttnn.all_gather(dim=3, cluster_axis=0, topology=Linear)` over the 8 vocab shards;
5. `ttnn.add(index_offsets, gathered_indices)` to globalize indices, then `ttnn.untilize`;
6. `ttnn.manual_seed(seeds=self._seeds, user_ids=self._user_ids)`;
7. `ttnn.sampling(gathered_values, global_indices, k=..., p=..., temp=...)`.

Note step 6 runs **inside** `decode_forward`, per call. Note also that with 8 vocab shards × top-32
each, `ttnn.sampling` chooses from **256 gathered candidates**.

### Host coverage that already exists (do not duplicate it)

[test_sampling_2d.py](../models/common/tests/modules/sampling/test_sampling_2d.py) has
`test_seeded_sampling_is_repeatable_and_slot_stable` (:193) and
`test_unseeded_sampling_uses_fresh_randomness` (:215). **Both test `sample_host`, not the device
path.** They prove the torch reference behaves; they say nothing about `ttnn.sampling`.

## The single most important design constraint

**You cannot compare device stochastic tokens against `sample_host`. Do not try.**

- `_device_seed(seed, slot)` is a blake2b digest masked to **31 bits**
  ([sampling_2d.py:621](../models/common/modules/sampling/sampling_2d.py#L621)).
- `_host_seed(seed, slot)` is the same digest masked to **63 bits**
  ([sampling_2d.py:625](../models/common/modules/sampling/sampling_2d.py#L625)).
- The device draws with TTNN's RNG; the host draws with `torch.multinomial` and a CPU generator
  ([sampling_2d.py:264-268](../models/common/modules/sampling/sampling_2d.py#L264-L268)).

Different seed width, different generator, different algorithm. Token-for-token equality between
device and host is **not** a property of this design, and a test asserting it is wrong. If you write
one and it fails, the correct action is to delete the test, not to relax it and not to "fix"
`sampling_2d.py` to make the two agree.

What *is* assertable is stated below.

## Prior art to reuse — read it before writing anything

[models/common/tests/modules/sampling/test_sampling_1d.py](../models/common/tests/modules/sampling/test_sampling_1d.py)
already solved this problem for 1D. Three pieces matter:

| Lines | What it gives you |
| --- | --- |
| 819-844 | `_hf_valid_token_set(logits_row, k, p, temp)` — builds the eligible-token set with HuggingFace's `TemperatureLogitsWarper → TopKLogitsWarper → TopPLogitsWarper`. An auditable reference instead of a hand-rolled filter. |
| 847-907 | `test_sampling1d_token_in_valid_set` — the containment test, parametrized over `(k, p, temp, max_boundary_violations)`. |
| 910-948 | `test_sampling1d_deterministic_with_same_seed` — same seed, two calls, identical tokens. |

Read the docstring at lines 865-876 carefully. It explains `max_boundary_violations`: `ttnn.sampling`
computes softmax and cumsum in **bfloat16** while the HF reference uses float32, so a token sitting
at the top-p nucleus cutoff can land on either side. The 1D calibration is `0` violations when
`p ∈ {0.0, 1.0}` (no nucleus threshold exists) and 2-6 out of 32 when `p ∈ (0, 1)`.

**Reuse `_hf_valid_token_set` rather than copying it.** The 2026-08-25 checkpoint in
`tttv2_2d_modules_work_log.md` established the pattern: shared test reference plumbing goes in
[models/common/tests/modules/_hf_reference.py](../models/common/tests/modules/_hf_reference.py), and
both the 1D and 2D suites import it. Move `_hf_valid_token_set` there and have `test_sampling_1d.py`
import it. Changing a 1D **test** to import a shared helper is permitted and has precedent; changing
a 1D **module implementation** file is not.

## What to build

New file: `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py`
(keep the existing file untouched — its greedy case is recorded Milestone A evidence).

Keep the existing test's geometry exactly, it is the qualified one:

```python
vocab_size, padded_vocab_size, batch = 151936, 152064, 32
sub_core_grids   = CoreRangeSet([CoreRange((1,0),(3,9)), CoreRange((5,0),(6,9))])
sub_core_grid_topk = CoreRangeSet([CoreRange((1,0),(3,9))])
start_core       = CoreCoord(1, 0)
mesh_mapper      = ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4))
device_params    = {"fabric_config": FABRIC_1D, "dispatch_core_axis": DispatchCoreAxis.COL}
```

### Test 1 — support containment under stochastic sampling (highest value)

The 2D analogue of `test_sampling1d_token_in_valid_set`. Parametrize `(k, p, temp)` and assert every
sampled token lies in the HF-derived eligible set, allowing a calibrated boundary tolerance.

Suggested matrix, mirroring the 1D calibration:

| id | k | p | temp | tolerance |
| --- | --- | --- | --- | --- |
| `k1-p0-t1` | 1 | 0.0 | 1.0 | 0 (degenerates to argmax) |
| `k8-p1-t1` | 8 | 1.0 | 1.0 | 0 (pure top-k, no nucleus) |
| `k32-p1-t1` | 32 | 1.0 | 1.0 | 0 (full candidate width) |
| `k32-p0.9-t1` | 32 | 0.9 | 1.0 | calibrate |
| `k32-p0.5-t0.8` | 32 | 0.5 | 0.8 | calibrate |

**Calibrate the tolerance from observation, and say so in the docstring and the report.** Run each
nucleus case 3-5 times, record the violation counts you actually saw, set the bound just above the
maximum, and write the observed numbers into the docstring the way the 1D test does. Do not copy the
1D numbers blindly — 2D gathers 256 candidates across 8 shards, so the bf16 boundary behaviour may
differ. A tolerance you cannot justify from a recorded run is not acceptable.

Do **not** parametrize `k > 32`: `max_top_k` is 32 and `_validate_call`
([sampling_2d.py:185-187](../models/common/modules/sampling/sampling_2d.py#L185-L187)) rejects it.

Containment is exactly valid for `k <= 32`: the true global top-k is always a subset of the union of
the per-shard top-32 sets, so the gather cannot drop an eligible token.

### Test 2 — padded-vocabulary exclusion under stochastic sampling

This is the assertion the current greedy test cannot make. Build logits the same way it does —
valid region low, padded tail `[151936:152064]` set to `+1000.0`
([test_sampling_2d_wh_galaxy.py:44-47](../models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy.py#L44-L47))
— then sample **stochastically** (`forced_argmax=False`, `temperature=1.0`, `top_k=32`, `top_p=1.0`)
across at least 8 invocations, and assert `torch.all(tokens < vocab_size)` every time.

The `invalid_vocab_mask` is added before `ttnn.topk`, so a correct implementation never lets a padded
column into the candidate set. Under argmax a partially-broken mask can still be masked by one
dominant valid logit; under stochastic sampling with a `+1000.0` tail it cannot. **If this test
fails, you have found a real bug — report it, do not weaken the test.**

### Test 3 — seeded determinism and slot stability

Two properties, one test or two, your call:

1. **Repeatability** — same `seed`, same logits, two `decode_forward` calls → identical tokens.
2. **Slot stability** — this is the 2D-specific one the plan names explicitly.
   `_device_seed(seed, slot)` derives per-slot seeds from one integer, and `slot_placement`
   ([sampling_2d.py:147-151](../models/common/modules/sampling/sampling_2d.py#L147-L151)) maps
   slot → `(mesh column, local index)`. So slot *i*'s token must depend only on
   `(seed_i, logits row i)` and nothing else.

   Test it by perturbation: call with all 32 slots seeded and record the tokens. Call again with a
   **different** seed on slots 16-31 only (pass `seed` as a 32-element sequence — `_broadcast` at
   [:596](../models/common/modules/sampling/sampling_2d.py#L596) accepts sequences). Tokens for
   slots 0-15 must be byte-identical; tokens for 16-31 should differ. Slots 0-15 changing means
   cross-slot RNG contamination, which is a serving-correctness bug.

Note `_update_call_buffers` refills **all 32** seed values with `secrets.randbits(31)` on every call
([:203](../models/common/modules/sampling/sampling_2d.py#L203)) and only then overwrites the slots
whose seed is not `None`. So a fully-seeded call is deterministic and a partially-seeded call is
deterministic exactly on its seeded slots — that is the behaviour to pin.

### Test 4 — unseeded freshness

`seed=None` → every slot gets a fresh `secrets.randbits(31)` per call. Assert that across N
invocations on a deliberately flat distribution (so the sampler has real entropy to express) the
tokens are **not** all identical.

Design the bound so a correct implementation fails with probability < 1e-9, and write the
calculation into the docstring. With a flat top-32 and 32 slots, requiring "at least one slot differs
across 8 calls" is astronomically safe. Do not assert anything stronger about the distribution here
— that is Test 5's job.

### Test 5 — per-slot heterogeneous parameters

Production sends different `top_k`/`top_p`/`temperature` per request. `_broadcast` supports it, the
buffers are per-slot, and **no test anywhere exercises it on device.** One call with, say, slots 0-7
greedy (`forced_argmax=True`), slots 8-15 at `k=1`, slots 16-23 at `k=8, p=1.0`, slots 24-31 at
`k=32, p=0.9, temp=0.8`. Assert each group satisfies its own containment property, and that the
greedy group exactly equals `argmax`.

This also pins the `temperature=0.0 without forced_argmax` collapse at
[:206](../models/common/modules/sampling/sampling_2d.py#L206) — include a slot group with
`temperature=0.0, forced_argmax=False` and assert it equals argmax.

### Test 6 — repeat invocation and clean release

Every Milestone A device test invokes at least twice and tears down cleanly. Make sure at least one
of the above loops ≥2 invocations, calls `sampler.release()`
([:392](../models/common/modules/sampling/sampling_2d.py#L392)) in a `finally`, and that the log ends
with the normal `Closing user mode device drivers` → `Cluster destructor completed` sequence.

## Known hazards on this host

Two Milestone A defects were root-caused on 2026-08-25, and **both were L1 address/ownership faults
that presented as intermittent passes rather than failures.** Read the last two checkpoints in
`tttv2_2d_modules_work_log.md` before you start. The lesson that applies here:

> A test that passes once on this hardware has not proved anything. A stats buffer aliased onto the
> wrong core passed in some processes and failed in others with the same seed.

So: **run every new test at least three times in fresh processes** before recording it as evidence,
and run the whole new file at least twice. If a case flips between pass and fail across processes,
that is the signature of reading uninitialized or aliased L1 — treat it as a defect and root-cause
it, do not average over it.

Specific to sampling:

- `ttnn.manual_seed(seeds, user_ids)` mutates **global device RNG state**. It is re-applied inside
  every `decode_forward`, but if you write a test that calls `ttnn.sampling` outside the module, you
  own that ordering.
- The `seed_buffer` mapper is `dims=(None, 0)` — replicated across the 8 rows, sharded over the 4
  columns, 8 seeds per column ([:512-518](../models/common/modules/sampling/sampling_2d.py#L512-L518)).
  `user_ids` is `arange(users_per_shard) = arange(8)`, replicated. If seeded determinism fails, check
  this placement before suspecting the RNG.
- `local_indices` is `uint16` over a local vocab of 19008 — fine for this geometry, but do not change
  `vocab_size` upward without rechecking it.

## If you find a defect

Report it. Root-cause it if you can do so without guessing. A module fix is permitted **only** if you
can demonstrate the defect with a failing test first and show the fix on hardware across ≥3 fresh
processes. Never change a threshold, a tolerance, or a parametrization to make a failure disappear.

## Run procedure

Host: the reserved 6U WH Galaxy, 32 devices. Confirm before starting:

```sh
ls /dev/tenstorrent | wc -l      # must be 32
tt-smi -ls
pgrep -af 'pytest|ttnn' | grep -v grep     # must be empty
```

If fewer than 32 device nodes are present, stop and report; do not run a reduced mesh.

**Exactly one pytest process may touch the device at any moment.** Never put pytest in a shell
pipeline (`| tee`, `| grep`) — the pipeline can return control while the nested process still holds
the device. Redirect with `> LOG 2>&1` and read the file afterwards.

```sh
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The agent harness caps a foreground tool call at 600 s, which is shorter than the 2700 s bound. If a
run needs longer, issue it as a tracked background process and block on its exit before issuing the
next one — the previous evidence run disclosed this same deviation. Re-check `pgrep` between runs.

On a hang or crash: kill the process tree, confirm the device is free, `tt-smi -glx_reset`, confirm
`Re-initialized 32 boards`, retry. **Maximum 2 recovery attempts**, then record `BLOCKED (infra)`
with logs and move on.

Keep every log from every attempt, including failures and resets. Never overwrite a log.

## Host gates before you touch the device

```sh
python -m pytest models/common/tests/modules/sampling/test_sampling_2d.py \
                 models/common/tests/modules/sampling/test_sampling_1d.py -q
pre-commit run --files <every file you touched>
```

The 1D suite must stay green — you are moving `_hf_valid_token_set` into shared plumbing and it
imports from there now.

## Hard prohibitions

- Do not modify any `models/common/modules/**/*_1d.py` implementation file. Sharing a **test** helper
  is fine; changing 1D module source is not.
- Do not relax a threshold, tolerance, or parametrization to turn a failure green. A failing test is
  a result to report.
- Do not assert device/host token equality (see "the single most important design constraint").
- Do not set a `max_boundary_violations` you cannot justify from a recorded run.
- Do not `git commit`, `push`, `checkout`, `stash`, or `reset`. Leave the tree dirty for review.
- Do not rebuild tt-metal or recreate the venv.
- Do not run the full `models/common/tests` suite or any 1D hardware matrix — the work log records
  that expanding to affected-file selection ballooned to 516 hardware cases.
- Do not claim a result you did not observe.

## Deliverables

1. `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py`.
2. `_hf_valid_token_set` moved into `models/common/tests/modules/_hf_reference.py`, with
   `test_sampling_1d.py` importing it and still green.
3. An evidence directory `tttv2_milestone_a_gap1_evidence/` holding one raw log per run (including
   the ≥3 repeat runs and any failures/resets) plus `REPORT.md` with: node IDs, pass/fail, the
   observed boundary-violation counts per parametrization, the calibration you chose and why, repeat
   counts, teardown confirmation, and anything you could not close.
4. A `## Hardware checkpoint: Sampling2D stochastic hardware coverage <ISO date>` section appended to
   `tttv2_2d_modules_work_log.md`, matching that file's terse bullet style.
5. A one-line status correction for `models/common/modules/MILESTONE_A_STATUS.md` row `Sampling2D`
   — write the proposed replacement text in your report; do not edit the status page yourself, it is
   being rewritten wholesale as a separate task.

## Finish condition

Every test you added has a terminal state with at least one log, the stochastic paths named in the
plan (greedy / seeded stochastic / unseeded stochastic) each have recorded device evidence or a
documented reason they could not be closed, `REPORT.md` and the work-log section are written, and the
device is left clean (`tt-smi -ls` showing 32 boards). Print the absolute path of `REPORT.md` last.

If you finish early, do not invent extra work. Stop.
