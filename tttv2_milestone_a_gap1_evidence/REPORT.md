# WH Galaxy Device Evidence — Gap 1: `Sampling2D` stochastic hardware coverage

Commit `de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd` ("add reusable WH Galaxy 2D modules"),
branch `gongyu/tttv2_wh_glx_2d_modules`, run on real hardware on
`wh-glx6u-05-special-ctr-apbernal-for-reservation-116970` (complete 6U WH Galaxy, 32 devices),
2026-08-25.

Brief: [`../tttv2_milestone_a_gap_briefs/gap1_sampling2d_stochastic_hardware.md`](../tttv2_milestone_a_gap_briefs/gap1_sampling2d_stochastic_hardware.md).
Completion handoff: [`../tttv2_milestone_a_gap_briefs/gap1_completion_handoff.md`](../tttv2_milestone_a_gap_briefs/gap1_completion_handoff.md).
Predecessor session notes: [`HANDOFF.md`](HANDOFF.md).

## 1. Summary

**Closing this gap found and fixed a real defect in `Sampling2D`.** `ttnn.sampling`'s `temp`
argument is the *reciprocal* temperature — the kernel multiplies candidate logits by it before the
softmax — but `_update_call_buffers` was writing the raw `T` into the device temperature buffer.
Every request with `T != 1.0` therefore sampled from a distribution warped in the wrong direction.
The fix is a one-line change to
[`sampling_2d.py:213`](../models/common/modules/sampling/sampling_2d.py#L213).

The gap is **closed**. `Sampling2D`'s stochastic path is now qualified on the `(8, 4)` 6U WH Galaxy
across nine device cases — top-k/top-p containment over five parametrizations, padded-vocabulary
exclusion under stochastic sampling, seeded repeatability with per-slot seed stability, unseeded
freshness, and per-slot heterogeneous `k`/`p`/`temperature` — with **`9 passed` in each of three
fresh post-fix processes** (`run03`, `run04`, `run05`) and **zero boundary violations on all 23
report lines in all three**.

The defect was demonstrated failing *before* the fix
([`run02_prefix_defect_demo.log`](logs/run02_prefix_defect_demo.log): `2 failed, 7 passed`), and the
two failing cases are exactly the two `T != 1.0` cases. That pre-fix/post-fix correlation — described
in §5 — is the strongest single piece of evidence here.

The defect was **structurally unreachable by the pre-existing coverage**: `1.0` is its own
reciprocal, so the bug is invisible at `T = 1.0`, and the greedy path forces `temp = 1.0`
unconditionally. The only pre-existing hardware test for `Sampling2D` is greedy. That is the
argument for this gap having been worth closing.

## 2. Environment

| Item | Value |
| --- | --- |
| Commit | `de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd` (`gongyu/tttv2_wh_glx_2d_modules`) |
| Host | `wh-glx6u-05-special-ctr-apbernal-for-reservation-116970` |
| Devices | 32 `/dev/tenstorrent` nodes; all Wormhole `tt-galaxy-...L` boards ([`logs/99_tt_smi_after.log`](logs/99_tt_smi_after.log)) |
| Mesh | `(8, 4)`, `FabricConfig.FABRIC_1D`, `DispatchCoreAxis.COL` |
| Arch | `wormhole_b0` |
| Build | `Release` (`build` → `build_Release`) |
| Python | 3.10.21 in `python_env/` |

**The working tree is dirty by design.** The reciprocal-temperature fix and the new device tests
*are* the deliverable of this job; they are deliberately left uncommitted for review. No
`git commit`, `push`, `checkout`, `stash`, or `reset` was run at any point.

Files changed or added by this gap job:

| File | State |
| --- | --- |
| `models/common/modules/sampling/sampling_2d.py` | The reciprocal-temperature fix — one functional line — plus a four-line explanatory comment at [`:202-205`](../models/common/modules/sampling/sampling_2d.py#L202-L205). `git diff --numstat`: `+5 −1` |
| `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py` | **New**, untracked. The 9 device cases |
| `models/common/tests/modules/_hf_reference.py` | Untracked. **Pre-existing** — created by the earlier "2D module tests share the 1D reference" checkpoint for the attention/MLP helpers. This job *added* `hf_valid_token_set` to it ([`:310`](../models/common/tests/modules/_hf_reference.py#L310)) so the 1D and 2D sampling suites share one HuggingFace-derived reference |
| `models/common/tests/modules/sampling/test_sampling_1d.py` | `+2 −29`: local `_hf_valid_token_set` removed, imports the shared one and updates the single call site. No 1D behaviour changed, and no tolerance touched |
| `models/common/tests/modules/sampling/test_sampling_2d.py` | `+29 −1`: new `test_device_temperature_buffer_holds_reciprocal_temperature`, and one pre-existing assertion at [`:168`](../models/common/tests/modules/sampling/test_sampling_2d.py#L168) corrected — it had pinned the buggy raw-`T` value |

Other modified files in `git status` (`rmsnorm_2d.py`, the attention/mlp/rmsnorm tests,
`_wh_galaxy_hardware.py`) belong to earlier checkpoints on this branch and were **not** touched by
this job.

## 3. The defect

### What was wrong

`ttnn.sampling` takes `temp` as a **reciprocal** temperature: its kernel *multiplies* the candidate
logits by `temp` before the softmax. `Sampling2D._update_call_buffers` was writing the raw requested
temperature `T` into the device buffer:

```python
temperature_values[slot] = 1.0 if force_greedy else call.temperature[index]   # WRONG
```

For a request at `T`, the device therefore evaluated `softmax(logits * T)` where the contract is
`softmax(logits / T)`. The distribution is warped in the **inverse** direction: `T = 0.8` (intended
to *sharpen*) instead *flattened* the distribution, admitting tokens that the top-p nucleus should
have excluded.

### The fix

[`sampling_2d.py:213`](../models/common/modules/sampling/sampling_2d.py#L213):

```python
temperature_values[slot] = 1.0 if force_greedy else 1.0 / call.temperature[index]
```

with the reasoning recorded in the buffer comment at
[`:202-205`](../models/common/modules/sampling/sampling_2d.py#L202-L205) so the convention is not
re-broken.

`temperature[index] == 0.0` is folded into `force_greedy` on the line above, so the reciprocal is
never evaluated at zero.

### The host path was already correct

`sample_host` **divides** — `torch.topk(row / call.temperature[row_index], k=k)` at
[`:260`](../models/common/modules/sampling/sampling_2d.py#L260) — so it always implemented the
documented contract. Host and device now agree semantically. **No host-path change was needed**, and
none was made.

### Why the existing hardware test could not catch it

Three independent reasons, all of which had to hold at once:

1. **`1.0` is its own reciprocal.** At `T = 1.0`, `1/T == T`, so the wrong expression produces the
   right buffer value. The bug is exactly invisible there.
2. **The greedy path forces `temp = 1.0` unconditionally**, via the `force_greedy` branch — so it is
   pinned to the one value where the defect cannot show.
3. **The only pre-existing `Sampling2D` hardware test is greedy.** `test_sampling_2d_wh_galaxy.py`
   passes both `forced_argmax=True` and `temperature=0.0`, which collapses every slot to
   `k=1, p=0.0, temp=1.0`. `ttnn.sampling` never takes its stochastic branch in that file at all.

So the defect was not *missed* by the existing coverage — it was **unreachable** by it. Any test
that could have caught it had to be stochastic *and* use a temperature other than `1.0`. Both of the
new cases that satisfy that condition caught it immediately.

A host regression test now pins the convention directly:
`test_device_temperature_buffer_holds_reciprocal_temperature` in `test_sampling_2d.py` asserts the
buffer holds `1/T`, so the fix cannot silently regress without device time.

## 4. Results table

Nine device node IDs, five runs, 45 rows. Every `Run` cell names a real file in
[`logs/`](logs/). Violation counts are extracted from the tests' own `[sampling2d-stochastic]`
report lines (emitted by `_report`, `test_sampling_2d_wh_galaxy_stochastic.py:126`), so every
passing run records its own calibration in its log. `Tol` is the `max_boundary_violations` in force
for that run. `call` is the pytest `call`-phase duration from each log's `slowest 25 durations`
block.

All node IDs are prefixed `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py::test_sampling_2d_wh_galaxy_`.

| Run | Node ID | Result | Observed violations (per invocation) | Tol | `call` |
| --- | --- | --- | --- | --- | --- |
| `run01_calibration.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k1-p0-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 6.55s |
| `run01_calibration.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k8-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.32s |
| `run01_calibration.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.31s |
| `run01_calibration.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.9-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 32 | 0.52s |
| `run01_calibration.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.5-t0.8-8x4-device_params0]` | **PASSED** | inv0: **2/32**; inv1: **9/32** | 32 | 0.55s |
| `run01_calibration.log` | `stochastic_excludes_padded_vocab`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32**; inv2: **0/32**; inv3: **0/32**; inv4: **0/32**; inv5: **0/32**; inv6: **0/32**; inv7: **0/32** | 0 | 0.62s |
| `run01_calibration.log` | `seeded_sampling_is_repeatable_and_slot_stable`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | n/a — determinism assertions only | — | 0.32s |
| `run01_calibration.log` | `unseeded_sampling_uses_fresh_randomness`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | 8 distinct vectors / 8 invocations | >1 | 0.36s |
| `run01_calibration.log` | `per_slot_heterogeneous_parameters`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | k32-p0.9-t0.8 slots 24-31 inv0: **1/8**<br>k32-p0.9-t0.8 slots 24-31 inv1: **0/8**<br>k8-p1-t1 slots 16-23 inv0: **0/8**<br>k8-p1-t1 slots 16-23 inv1: **0/8** | 0/8 | 0.30s |
| `run02_prefix_defect_demo.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k1-p0-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 6.97s |
| `run02_prefix_defect_demo.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k8-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.33s |
| `run02_prefix_defect_demo.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.34s |
| `run02_prefix_defect_demo.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.9-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.52s |
| `run02_prefix_defect_demo.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.5-t0.8-8x4-device_params0]` | **FAILED** | inv0: **4/32** | 1 | 0.37s |
| `run02_prefix_defect_demo.log` | `stochastic_excludes_padded_vocab`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32**; inv2: **0/32**; inv3: **0/32**; inv4: **0/32**; inv5: **0/32**; inv6: **0/32**; inv7: **0/32** | 0 | 0.55s |
| `run02_prefix_defect_demo.log` | `seeded_sampling_is_repeatable_and_slot_stable`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | n/a — determinism assertions only | — | 0.27s |
| `run02_prefix_defect_demo.log` | `unseeded_sampling_uses_fresh_randomness`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | 8 distinct vectors / 8 invocations | >1 | 0.35s |
| `run02_prefix_defect_demo.log` | `per_slot_heterogeneous_parameters`<br>`[wormhole_b0-8x4-device_params0]` | **FAILED** | k32-p0.9-t0.8 slots 24-31 inv0: **2/8**<br>k8-p1-t1 slots 16-23 inv0: **0/8** | 0/1 | 0.28s |
| `run03_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k1-p0-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 6.13s |
| `run03_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k8-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.32s |
| `run03_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.30s |
| `run03_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.9-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.50s |
| `run03_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.5-t0.8-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.51s |
| `run03_postfix.log` | `stochastic_excludes_padded_vocab`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32**; inv2: **0/32**; inv3: **0/32**; inv4: **0/32**; inv5: **0/32**; inv6: **0/32**; inv7: **0/32** | 0 | 0.55s |
| `run03_postfix.log` | `seeded_sampling_is_repeatable_and_slot_stable`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | n/a — determinism assertions only | — | 0.27s |
| `run03_postfix.log` | `unseeded_sampling_uses_fresh_randomness`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | 8 distinct vectors / 8 invocations | >1 | 0.34s |
| `run03_postfix.log` | `per_slot_heterogeneous_parameters`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | k32-p0.9-t0.8 slots 24-31 inv0: **0/8**<br>k32-p0.9-t0.8 slots 24-31 inv1: **0/8**<br>k8-p1-t1 slots 16-23 inv0: **0/8**<br>k8-p1-t1 slots 16-23 inv1: **0/8** | 0/1 | 0.32s |
| `run04_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k1-p0-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 6.05s |
| `run04_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k8-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.29s |
| `run04_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.30s |
| `run04_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.9-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.52s |
| `run04_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.5-t0.8-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.52s |
| `run04_postfix.log` | `stochastic_excludes_padded_vocab`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32**; inv2: **0/32**; inv3: **0/32**; inv4: **0/32**; inv5: **0/32**; inv6: **0/32**; inv7: **0/32** | 0 | 0.57s |
| `run04_postfix.log` | `seeded_sampling_is_repeatable_and_slot_stable`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | n/a — determinism assertions only | — | 0.27s |
| `run04_postfix.log` | `unseeded_sampling_uses_fresh_randomness`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | 8 distinct vectors / 8 invocations | >1 | 0.34s |
| `run04_postfix.log` | `per_slot_heterogeneous_parameters`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | k32-p0.9-t0.8 slots 24-31 inv0: **0/8**<br>k32-p0.9-t0.8 slots 24-31 inv1: **0/8**<br>k8-p1-t1 slots 16-23 inv0: **0/8**<br>k8-p1-t1 slots 16-23 inv1: **0/8** | 0/1 | 0.36s |
| `run05_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k1-p0-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 8.14s |
| `run05_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k8-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.30s |
| `run05_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p1-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 0 | 0.32s |
| `run05_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.9-t1-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.51s |
| `run05_postfix.log` | `stochastic_token_in_valid_set`<br>`[wormhole_b0-k32-p0.5-t0.8-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32** | 1 | 0.52s |
| `run05_postfix.log` | `stochastic_excludes_padded_vocab`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | inv0: **0/32**; inv1: **0/32**; inv2: **0/32**; inv3: **0/32**; inv4: **0/32**; inv5: **0/32**; inv6: **0/32**; inv7: **0/32** | 0 | 0.65s |
| `run05_postfix.log` | `seeded_sampling_is_repeatable_and_slot_stable`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | n/a — determinism assertions only | — | not in slowest-25 |
| `run05_postfix.log` | `unseeded_sampling_uses_fresh_randomness`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | 8 distinct vectors / 8 invocations | >1 | 0.34s |
| `run05_postfix.log` | `per_slot_heterogeneous_parameters`<br>`[wormhole_b0-8x4-device_params0]` | **PASSED** | k32-p0.9-t0.8 slots 24-31 inv0: **0/8**<br>k32-p0.9-t0.8 slots 24-31 inv1: **0/8**<br>k8-p1-t1 slots 16-23 inv0: **0/8**<br>k8-p1-t1 slots 16-23 inv1: **0/8** | 0/1 | 0.29s |

### Run-level summary, exit codes and teardown

| Log | Wall clock (from log timestamps) | pytest summary | Exit | Teardown |
| --- | --- | --- | --- | --- |
| [`run01_calibration.log`](logs/run01_calibration.log) | 17:34:20 → 17:35:06 | `9 passed in 30.71s` | `exit=0` | clean |
| [`run02_prefix_defect_demo.log`](logs/run02_prefix_defect_demo.log) | 17:39:37 → 17:40:25 | `2 failed, 7 passed in 32.33s` | `exit=1` | clean |
| [`run03_postfix.log`](logs/run03_postfix.log) | 17:43:09 → 17:43:53 | `9 passed in 29.46s` | `exit=0` | clean |
| [`run04_postfix.log`](logs/run04_postfix.log) | 17:44:16 → 17:45:00 | `9 passed in 29.36s` | `exit=0` | clean |
| [`run05_postfix.log`](logs/run05_postfix.log) | 17:45:24 → 17:46:12 | `9 passed in 32.61s` | `exit=0` | clean |

"Clean" teardown means each log ends with the full
`Closing user mode device drivers` → `Closing devices in cluster` → `Closing devices in cluster
completed.` → `Cluster destructor started.` → `Cluster destructor completed.` sequence, followed by
its `exit=` line. Verified present exactly once in all five logs.

Two qualifications, stated precisely because the predecessor `HANDOFF.md` overstated them:

- The `In custom teardown, open device ids: {0, 1, ..., 31}` line — which is what actually
  enumerates all 32 devices — appears **only in `run02`**, twice, on its two failures. pytest's
  custom teardown hook logs that line on failure, not on every test. The other four logs do not
  contain it. Their clean shutdown is evidenced by the UMD cluster-close sequence above, not by a
  32-device enumeration.
- **No `tt-smi -glx_reset` was run at any point in this job**, and none was needed. No log contains
  a reset. Confirmed by grep across all five.

Runs 03, 04 and 05 are **three separate processes**: three disjoint timestamp windows, each with its
own device open/close cycle and its own `JIT cache stats: 187/187 hits (100.0%)` line at shutdown.

## 5. Calibration

### The bounds

| Case | `max_boundary_violations` | Rationale |
| --- | --- | --- |
| `k1-p0-t1`, `k8-p1-t1`, `k32-p1-t1` | **0** | `p ∈ {0.0, 1.0}` — no nucleus threshold exists, so the eligible set is exactly the top-k. Any violation is a real defect. |
| `k32-p0.9-t1`, `k32-p0.5-t0.8` | **1** | `p ∈ (0, 1)` — headroom only. See below. |
| heterogeneous slots 16-23 (`k8-p1-t1`) | **0** | Exact top-8 containment. |
| heterogeneous slots 24-31 (`k32-p0.9-t0.8`) | **1** | `p ∈ (0, 1)` — headroom only. |
| heterogeneous slots 0-7, 8-15 | exact equality | Greedy groups compared against argmax token-for-token. |

### The bound of 1 is headroom, not an observed requirement

This is worth stating plainly because it differs from the 1D suite. **Across runs 03-05 — three
fresh processes, 23 report lines each — the observed violation count was `0` in every case, in every
invocation, including both `p ∈ (0, 1)` nucleus cases.** The maximum observed post-fix violation
count anywhere in this evidence package is **zero**.

The bf16 softmax/cumsum boundary effect that forced the 1D suite to allow 2-6 violations **did not
manifest on this geometry at all** once the temperature was correct. The bound of `1` is one above
the observed maximum: deliberate headroom for that boundary, not a measured requirement.

**A run that reports any violation on these cases is a regression, not noise.** No tolerance was
relaxed to reach green — the two nucleus bounds were *tightened* from the deliberately-unconstrained
calibration values (`32` and `8`, see the `run01` rows in §4) down to `1`.

### The pre-fix / post-fix correlation

This is the core of the evidence. The violation pattern tracks temperature exactly, and only
temperature:

| Case | `T` | run01 (pre-fix, bounds open) | run02 (pre-fix, bounds final) | runs 03/04/05 (post-fix) |
| --- | --- | --- | --- | --- |
| `k1-p0.0-t1.0` | 1.0 | 0/32, 0/32 | 0/32, 0/32 | 0/32 every invocation |
| `k8-p1.0-t1.0` | 1.0 | 0/32, 0/32 | 0/32, 0/32 | 0/32 every invocation |
| `k32-p1.0-t1.0` | 1.0 | 0/32, 0/32 | 0/32, 0/32 | 0/32 every invocation |
| `k32-p0.9-t1.0` | 1.0 | 0/32, 0/32 | 0/32, 0/32 | 0/32 every invocation |
| `padded-tail` | 1.0 | 0/32 × 8 | 0/32 × 8 | 0/32 × 8 every run |
| heterogeneous `k8-p1-t1` | 1.0 | 0/8, 0/8 | 0/8 | 0/8 every invocation |
| **`k32-p0.5-t0.8`** | **0.8** | **2/32, then 9/32** | **4/32 → FAILED** | **0/32 every invocation** |
| **heterogeneous `k32-p0.9-t0.8`** | **0.8** | **1/8, then 0/8** | **2/8 → FAILED** | **0/8 every invocation** |

Three facts hold simultaneously:

1. **Every** case at `T = 1.0` showed `0` violations, pre-fix and post-fix alike — including the
   `p = 0.9` nucleus case, which exercises exactly the same bf16 cumsum boundary as the failing
   `p = 0.5` case. So the nucleus arithmetic is not the discriminator.
2. **Only** the two cases at `T != 1.0` ever showed violations, and they did so in every pre-fix
   invocation that reported a nonzero count.
3. Those violations vanished completely — to `0`, in three independent processes — after a change
   that touches nothing but the temperature buffer.

`T = 1.0` is precisely the fixed point of `T ↦ 1/T`. A multiply-vs-divide mismatch predicts exactly
this: no error wherever `T = 1.0`, error wherever `T != 1.0`, magnitude growing with distance from
`1.0`, and complete disappearance when the reciprocal is applied. Nothing else in the pipeline is
temperature-dependent in that way. The two failing cases in `run02` are the two `T != 1.0` cases and
nothing else.

The `run01` → `run02` progression on `k32-p0.5-t0.8` (`2/32`, `9/32`, then `4/32`) also shows the
expected *stochastic* character: the sampler draws fresh randomness each invocation, so a warped
distribution produces a varying number of out-of-set draws rather than a fixed one. Post-fix it is
not merely small — it is identically zero, six times over.

## 6. Host gates

Run after the docstring corrections, on this host, in this job.

| Gate | Result | Log |
| --- | --- | --- |
| `test_sampling_1d.py` + `test_sampling_2d.py` together | **`167 passed, 50 deselected in 77.55s`** | [`logs/07_host_suites_after_docstrings.log`](logs/07_host_suites_after_docstrings.log) |
| `test_sampling_1d.py` alone | **`140 passed, 50 deselected in 68.91s`** | [`logs/07a_host_1d.log`](logs/07a_host_1d.log) |
| `test_sampling_2d.py` alone | **`27 passed in 4.32s`** | [`logs/07b_host_2d.log`](logs/07b_host_2d.log) |
| `pre-commit run --files` (5 files) | 4 files fully clean; 1 pre-existing hook failure — see below | [`logs/06_pre_commit.log`](logs/06_pre_commit.log) |

### The 1D count is 140, not the 166 the handoff predicted — reconciled, not a regression

The completion handoff expected `166 passed` for `test_sampling_1d.py` and `27 passed` for
`test_sampling_2d.py`. The observed numbers are `140` and `27`. **This is a bookkeeping error in the
predecessor's notes, not a behavioural change.** The arithmetic closes exactly:

- `test_sampling_1d.py` collects `190` items and deselects `50`, leaving `140`. The deselection is
  `pytest_collection_modifyitems` in `models/common/tests/conftest.py:95`, which drops
  cross-product cases whose `ttnn_mesh_device` fixture parametrization does not match their
  `mesh_shape` parameter. It is purely parametrization-driven and completes in `0.23s` with no
  device involved, so the count is deterministic for a given file content.
- This job's diff to `test_sampling_1d.py` is a pure helper move (`_hf_valid_token_set` → imported
  `hf_valid_token_set`) and touches no `parametrize` decorator, so it cannot have changed collection.
- `test_sampling_2d.py` gained **exactly one** test function in this job
  (`test_device_temperature_buffer_holds_reciprocal_temperature`), so it held `26` before and holds
  `27` now.
- `140 + 26 = 166` — the predecessor's figure — and `140 + 27 = 167`, which is exactly what the
  combined run reports now. The `166` was a **combined** two-file run recorded against the 1D file
  alone, taken before the new host regression test was added.

The predecessor also recorded `458.41s` for that run against `77.55s` here. That is consistent with
kernel/program-cache warmth: the 1D suite opens real `1x1`/`1x2`/`1x8` meshes, and the earlier run
was the first to do so on a cold JIT cache.

All 167 tests pass. No test was skipped, xfailed, or deselected other than by the mesh cross-product
filter described above.

### `pre-commit`: one pre-existing failure, deliberately not silenced

The four files this job created or substantially rewrote are **fully clean**:

```
trim trailing whitespace / fix end of files / check merge conflicts / black / autoflake / isort
/ prefer-expect-error / large files / metalium includes / global torch import ....... all Passed
```

`test_sampling_2d.py` fails one hook, `prefer-expect-error` ("tests: use the `expect_error` fixture
instead of `pytest.raises`"). **This failure predates this job and was not introduced by it:**

- All seven flagged `pytest.raises` occurrences exist verbatim in the committed file at
  `HEAD:models/common/tests/modules/sampling/test_sampling_2d.py` (lines 75, 82, 89, 94, 99, 245,
  258 there; 75-99 and 273, 286 now — shifted only by this job's `+30` insertion).
- `git diff` on that file adds and removes **no** `pytest.raises` line.
- The hook is a `pygrep` rule added in `31e21e4d190` (2026-06-22), and **47 test files across the
  repository** currently trip it. It is a broad pre-existing condition.

**Decision (the brief does not cover this case; the conservative option was taken).** Two fixes were
available: convert the seven blocks to the `expect_error` fixture, or append the sanctioned
`# allow-pytest.raises: <why>` override to each line. Both were rejected.

- Converting them is a behavioural refactor of seven pre-existing assertions that no part of this
  brief asks for, on test surface this job otherwise only appended to.
- Appending overrides would suppress a real repo-level lint signal on code this job did not write,
  which is the same category of act as the house rule "do not relax a tolerance or a threshold"
  forbids.

So the failure is left standing and reported here rather than hidden. This is the one item on which
the brief's finish condition ("`pre-commit` clean on all five files") is **not** met, and it is not
met for a reason that predates the job. Everything this job authored is clean.

## 7. Caveats and gaps

What this evidence does **not** establish. Each of these is a deliberate scope boundary, not an
oversight.

1. **Distributional correctness of the RNG is not tested.** Every containment assertion checks only
   that the sampled token lies in the eligible *support* — the set the HuggingFace
   `TemperatureLogitsWarper → TopKLogitsWarper → TopPLogitsWarper` pipeline leaves finite. Nothing
   here checks that tokens are drawn with the *right probabilities* within that set. A sampler that
   always returned the top-1 token, or one with a badly biased RNG, would pass every containment
   case in §4. The determinism and freshness tests constrain the RNG's *reproducibility*, not its
   distribution.

   Note that this limit does not weaken the defect finding: the reciprocal-temperature bug was
   caught precisely because a wrong temperature moves tokens *outside* the support, not merely
   within it.

2. **Device tokens are never compared against `sample_host`, by design.** The two paths draw from
   different generators seeded at different widths — `_device_seed` masks the blake2b digest to
   **31 bits** (`sampling_2d.py:625`) and feeds `ttnn.manual_seed`, while `_host_seed` masks to
   **63 bits** (`:629`) and feeds `torch.Generator` / `torch.multinomial`. Token-for-token equality
   between them is not a property of the design and is not asserted anywhere. What the host path
   *is* used for is establishing that the divide-vs-multiply convention now agrees semantically
   (§3).

3. **`top_k > 32` is out of contract and untested.** Containment is exactly valid in these tests
   *because* `k <= max_top_k = 32`: the true global top-k is then guaranteed to be a subset of the
   union of the eight per-shard top-32 sets, so the all-gather cannot drop an eligible token before
   `ttnn.sampling` runs. Above 32 that argument fails and the tests would no longer be sound. No
   case here exercises it.

4. **No trace/capture coverage.** Every case runs eager `decode_forward`. `Sampling2D` under
   `ttnn` trace capture and replay is not exercised by this file. This overlaps the separate
   batched-prefill / physical-32 trace gap (gap 3) and was not addressed here.

5. **One geometry only.** All nine cases run on `(8, 4)` with `FABRIC_1D` and
   `DispatchCoreAxis.COL` — the geometry the greedy Milestone A test already qualified. No other
   mesh shape, fabric config, or dispatch axis is covered for the stochastic path.

6. **Synthetic logits, not model logits.** Inputs are a tie-free candidate ladder
   (`_CANDIDATES = 64`, top `8.0`, step `0.25`, baseline `−20.0`), chosen because a `torch.randn`
   draw over 151936 bf16 tokens produces many exact ties at the top-k threshold, which inflates the
   HuggingFace reference set and weakens containment. This makes the eligible set exact, but it is
   not a real LM-head distribution.

7. **Two invocations per case (eight for the padded-tail and unseeded tests).** Repetition is across
   processes (three post-fix runs) rather than deep within a process. A rare intermittent fault with
   a period longer than that would not be caught.

8. **`run02` is a demonstration, not a controlled A/B.** It records the pre-fix code failing the
   final bounds. It ran on the same host, in the same session, minutes before the fix
   (`run01` 17:34, `run02` 17:39, fix applied, `run03` 17:43) and differs from `run03`-`run05` only
   by the one-line change — but the two arms were run sequentially, not interleaved, and each arm
   was a single process. The argument in §5 therefore rests on the *temperature correlation across
   all five runs* — including the fact that every `T = 1.0` case is clean in both arms — not on
   `run02` alone.

## 8. The 1D follow-up — a testable prediction, deliberately not acted on

The 1D sampling suite's calibrated tolerances are **asymmetric in exactly the direction a
multiply-vs-divide mismatch predicts**:
[`test_sampling_1d.py:832-834`](../models/common/tests/modules/sampling/test_sampling_1d.py#L832-L834)
allows **6** violations at `t=0.5` but only **2** at `t=2.0`.

The mechanism differs from the 2D defect, and that distinction matters:

- `Sampling1D` passes `temp` **straight through** to `ttnn.sampling`. So *its* `temp` argument
  already **is** the op's `1/T`. The module is not wrong.
- The mismatch is in the **test's HF reference**, which warps with the raw value: it calls
  `hf_valid_token_set(..., temp=temp)`, and `TemperatureLogitsWarper` *divides*. Reference and
  device therefore disagree by a reciprocal, and the disagreement is worst where `T` is furthest
  from `1.0` — larger at `0.5` than at `2.0`, which is the observed asymmetry.

**The prediction, stated so it can be falsified cheaply:** if
`test_sampling1d_token_in_valid_set` passed `1/temp` to `hf_valid_token_set`, the **`t=0.5` and
`t=2.0` tolerances should collapse toward the `t=1.0` baseline**, and their asymmetry should
disappear.

One qualification that keeps the prediction honest, and that the completion handoff did not note:
the third case in that block is `pytest.param(32, 0.5, 1.0, 3, id="k32-p0.5-t1")` — a `T = 1.0` case
that already allows **3** violations. At `T = 1.0` the reciprocal is a no-op, so the reference and
the device agree on temperature exactly and this hypothesis cannot explain that headroom. A genuine
bf16 nucleus effect therefore does exist in the 1D geometry, independent of temperature. So the
prediction is specifically about the *temperature-dependent excess* (6 and 2 against a baseline of
3), **not** that all three tolerances go to zero. If the fix drives `t=0.5` and `t=2.0` to
approximately 3 and leaves `t=1.0` at 3, the hypothesis is confirmed; if `t=0.5` stays near 6, it is
refuted.

Why the 2D bounds could collapse to 0 where the 1D ones cannot is itself consistent: the 2D tests
use a deliberately tie-free candidate ladder (§7.6) that makes the eligible set exact, while the 1D
tests draw over full model vocabularies where bf16 ties at the top-k threshold are common.

**This was not touched and was deliberately not touched.** It is 1D test surface, it needs its own
hardware run to confirm, and it is outside this brief's scope — which explicitly forbids modifying
the 1D sampling tolerances. It is filed as a follow-up in `tttv2_2d_modules_work_log.md` with the
prediction written down.

If the prediction holds, the 1D tolerances are currently masking a reference bug rather than a
hardware effect, and tightening them would strengthen the 1D gate. If it fails, the asymmetry has
another cause and the tolerances are doing real work. Either outcome is cheap to obtain and worth
knowing.

## 9. Proposed `MILESTONE_A_STATUS.md` replacement line

**Draft only — `MILESTONE_A_STATUS.md` was NOT edited.** Per the completion handoff, that file is
being rewritten wholesale as a separate task and concurrent edits would collide.

The current `Sampling2D` row (`models/common/modules/MILESTONE_A_STATUS.md:27`) reads:

> | Sampling2D | Included in final 1259-test host gate | Qwen forced argmax repeated with exact tokens and padded-vocabulary exclusion | Qualified for the required forced-argmax hardware case; stochastic hardware is not recorded |

Proposed replacement:

> | Sampling2D | Included in final 1259-test host gate, plus a host regression test pinning the device temperature buffer to `1/T` | Qwen forced argmax with exact tokens and padded-vocabulary exclusion; stochastic path qualified on WH Galaxy `(8, 4)` — top-k/top-p containment over 5 parametrizations, padded-vocabulary exclusion under stochastic sampling, seeded repeatability with per-slot seed stability, unseeded freshness, and per-slot heterogeneous k/p/temperature | Qualified for both the forced-argmax and the stochastic hardware cases. Stochastic qualification found and fixed a reciprocal-temperature defect in the device path (`ttnn.sampling` takes `1/T`; the module passed `T`, invisible at `T = 1.0` and on the greedy path). `9 passed` in each of three fresh post-fix processes with zero boundary violations; demonstrated failing pre-fix. Evidence: `tttv2_milestone_a_gap1_evidence/`. Not covered: RNG distributional correctness, `top_k > 32`, trace/capture |

## 10. Verdict

**Gap 1 is closed.** The `Sampling2D` stochastic path is qualified on the 6U WH Galaxy at `(8, 4)`
across nine device cases, `9 passed` in three fresh post-fix processes with zero boundary violations
on every report line. Closing it surfaced and fixed a real module defect that the pre-existing
greedy-only coverage could not have reached.

Outstanding, both disclosed above and neither a device or test result:

- `pre-commit`'s `prefer-expect-error` hook fails on `test_sampling_2d.py` for seven pre-existing
  `pytest.raises` blocks that predate this job (§6). Deliberately not silenced.
- The 1D reference-temperature follow-up (§8) is filed, not fixed.

Device left clean: `tt-smi -ls` shows all 32 Wormhole Galaxy boards present and resettable
([`logs/99_tt_smi_after.log`](logs/99_tt_smi_after.log)), `ls /dev/tenstorrent | wc -l` reports
`32`, and no reset was performed at any point in this job.
