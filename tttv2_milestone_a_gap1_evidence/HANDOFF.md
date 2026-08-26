# Handoff — Gap 1: `Sampling2D` stochastic hardware coverage

Status at handoff: **the hardware work is done and green; the write-up is not.**

All device evidence has been captured, a real module defect was found, root-caused, fixed, and
re-qualified across three fresh processes. What remains is documentation, a pre-commit pass over
the two files touched after the last pre-commit run, and one final device-clean check.

- Host: `wh-glx6u-05-special-ctr-apbernal-for-reservation-116970` (complete 6U WH Galaxy, 32 devices)
- Commit: `de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd`, branch `gongyu/tttv2_wh_glx_2d_modules`
- Tree is deliberately left dirty. Nothing was committed, pushed, stashed or reset.

---

## 1. What changed

| File | Change |
| --- | --- |
| `models/common/tests/modules/_hf_reference.py` | Added `hf_valid_token_set` under a new "Sampling reference" banner at end of file. Moved verbatim from the 1D suite; only the leading underscore was dropped and one em dash normalised. |
| `models/common/tests/modules/sampling/test_sampling_1d.py` | Removed its local `_hf_valid_token_set`; imports `hf_valid_token_set` from `_hf_reference` and updated the single call site. No 1D behaviour changed. |
| `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py` | **New.** 9 device cases. See §2. |
| `models/common/modules/sampling/sampling_2d.py` | **Module fix.** `_update_call_buffers` now writes `1.0 / T` into the device temperature buffer, plus an explanatory comment above the buffer allocation. See §3. |
| `models/common/tests/modules/sampling/test_sampling_2d.py` | Added `test_device_temperature_buffer_holds_reciprocal_temperature`. Corrected one pre-existing assertion in `test_prepare_call_refreshes_lazy_sources_by_global_slot` that had pinned the buggy value (`0.5` → `2.0` for `temperature=0.5`). |

`test_sampling_2d_wh_galaxy.py` (the recorded greedy Milestone A evidence) was **not** touched.
No `*_1d.py` module implementation file was touched.

## 2. The new test file

`models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py`, 9 collected cases
on the same qualified `(8, 4)` geometry as the greedy test (same sub-core grids, same
`ShardTensor2dMesh(dims=(3,2))`, same `FABRIC_1D` / `DispatchCoreAxis.COL`).

| Node ID suffix | Covers |
| --- | --- |
| `token_in_valid_set[...-k1-p0-t1-...]` | stochastic `k=1` degenerating to argmax |
| `token_in_valid_set[...-k8-p1-t1-...]` | pure top-k, no nucleus |
| `token_in_valid_set[...-k32-p1-t1-...]` | full 32-wide candidate support |
| `token_in_valid_set[...-k32-p0.9-t1-...]` | nucleus, neutral temperature |
| `token_in_valid_set[...-k32-p0.5-t0.8-...]` | nucleus + non-unit temperature (**this is the case that exposed the defect**) |
| `stochastic_excludes_padded_vocab` | `+1000.0` padded tail, 8 stochastic invocations |
| `seeded_sampling_is_repeatable_and_slot_stable` | same-seed repeatability + slots 16-31 reseed perturbation |
| `unseeded_sampling_uses_fresh_randomness` | flat 32-token support, 8 invocations |
| `per_slot_heterogeneous_parameters` | 4 slot groups with different k/p/temp/forced_argmax in one call |

Design notes worth preserving if anyone edits it:

- **Logits are a tie-free ladder, not `torch.randn`.** `_candidate_logits` scatters `_CANDIDATES=64`
  distinct token ids per slot carrying `8.0, 7.75, 7.5, …` over a `-20.0` baseline. A bfloat16
  `randn` over 151936 tokens produces many exact ties at the top-k threshold; `TopKLogitsWarper`
  keeps every token tied with the k-th value, which inflates the reference set and weakens
  containment. The ladder makes the eligible set exact and the argmax unique (`ids[slot, 0]`).
- Device tokens are **never** compared against `sample_host`. Different seed width (31 vs 63 bits),
  different generator, different algorithm.
- Every case prints `[sampling2d-stochastic] <label>: violations=N/M allowed=A`, so a passing run
  records its own calibration. **This requires `-s`** — pytest only shows captured stdout on
  failure otherwise. All recorded runs used `-s`.

## 3. The defect (found, fixed, re-qualified)

**`Sampling2D` applied the reciprocal of the requested temperature on the device path.**

- `ttnn.sampling`'s `temp` argument is documented as `1/T`
  (`ttnn/cpp/ttnn/operations/reduction/sampling/sampling_nanobind.cpp:45`) and the compute kernel
  applies `values *= temp` before the softmax
  (`.../device/kernels/compute/sampling.cpp:466`, `mul_block_bcast_scalar_inplace`).
- `sampling_2d.py:_update_call_buffers` wrote the raw `T` into `temperature_buffer`, which
  `decode_forward` passes as `temp=`. So the device computed `softmax(logits · T)` while the
  module's own `sample_host` computes `softmax(logits / T)`.
- **Why no existing test caught it:** `1.0` is its own reciprocal, and the greedy path forces the
  buffer to `1.0`. The existing greedy hardware test sets both `forced_argmax=True` and
  `temperature=0.0`, so the buffer was always `1.0`.

**Confirmation was quantitative, not circumstantial.** Under multiply-semantics at `T=0.8, p=0.5`
the device nucleus is ranks 0..3 while the reference nucleus is ranks 0..2 — so rank 3 is the
*only* rank that can violate. All 9 observed violations in run 01 were rank 3. At `T=0.8, p=0.9`
the device keeps ranks 0..11 vs the reference's 0..7; the observed violation was rank 9. At
`T=1.0` both semantics agree and 0 violations were observed in every run.

**Fix:** `temperature_values[slot] = 1.0 if force_greedy else 1.0 / call.temperature[index]`.
`force_greedy` already covers `temperature == 0.0`, so there is no division by zero.

**Known residual, deliberately not addressed:** a positive but extremely small `T`
(below ~2.9e-39) makes `1/T` overflow bfloat16 to `inf`, which would give a NaN softmax.
`_validate_call` accepts any non-negative temperature. Adding a clamp is a separate change that
this gap's evidence does not justify, so it was left out. Worth a follow-up ticket.

## 4. Device evidence captured

All logs in `tttv2_milestone_a_gap1_evidence/logs/`. None were overwritten.

| Log | State | Result |
| --- | --- | --- |
| `run01_calibration.log` | pre-fix, nucleus tolerances unconstrained | `9 passed` |
| `run02_prefix_defect_demo.log` | pre-fix, tolerances at final values | `2 failed, 7 passed in 32.33s` |
| `run03_postfix.log` | post-fix | `9 passed in 29.46s` |
| `run04_postfix.log` | post-fix | `9 passed in 29.36s` |
| `run05_postfix.log` | post-fix | `9 passed in 32.61s` |

Observed boundary-violation counts, per parametrization, per invocation:

| Case | run01 (pre-fix) | run02 (pre-fix) | run03/04/05 (post-fix) |
| --- | --- | --- | --- |
| `k1-p0-t1` | 0, 0 | 0, 0 | 0, 0 (each run) |
| `k8-p1-t1` | 0, 0 | 0, 0 | 0, 0 (each run) |
| `k32-p1-t1` | 0, 0 | 0, 0 | 0, 0 (each run) |
| `k32-p0.9-t1` | 0, 0 | 0, 0 | 0, 0 (each run) |
| `k32-p0.5-t0.8` | **2, 9** | **4** → FAILED | 0, 0 (each run) |
| padded tail (8 invocations) | 0 ×8 | 0 ×8 | 0 ×8 (each run) |
| heterogeneous `k8-p1-t1` slots 16-23 | 0, 0 | 0 | 0, 0 (each run) |
| heterogeneous `k32-p0.9-t0.8` slots 24-31 | **1, 0** | **2** → FAILED | 0, 0 (each run) |
| unseeded | 8 distinct vectors / 8 calls | — | 8 distinct / 8 (each run) |

The heterogeneous case going 1/8 in run 01 and 2/8 in run 02 is the "passes in one process, fails
in another" signature the brief warned about — here it was genuine unseeded draw variance over a
systematically wrong distribution, not aliased L1. Running three times is what made it legible.

**Chosen tolerance and its justification:** `0` for `p ∈ {0.0, 1.0}` (no nucleus threshold exists,
so the eligible set is exact and any violation is a defect); `1` for the three `p ∈ (0,1)` cases,
which is one above the maximum of **0** observed across runs 03, 04 and 05 (six invocations of each
nucleus case). This is already what is in the file — the numbers do not need changing, only the
docstring prose (§5, item 4).

Teardown: every run ends with the normal `Closing user mode device drivers` →
`Closing devices in cluster completed` → `Cluster destructor completed` sequence, with all 32
device ids listed in the custom teardown. No `tt-smi -glx_reset` was needed at any point, and no
run hung or was killed.

## 5. What is left to do

1. **`pre-commit run --files`** over all five touched files. It was last run on only three of them,
   *before* the module fix and the two `test_sampling_2d.py` edits:
   ```
   pre-commit run --files \
     models/common/modules/sampling/sampling_2d.py \
     models/common/tests/modules/_hf_reference.py \
     models/common/tests/modules/sampling/test_sampling_1d.py \
     models/common/tests/modules/sampling/test_sampling_2d.py \
     models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py
   ```

2. **Write `tttv2_milestone_a_gap1_evidence/REPORT.md`.** Follow the shape of
   `tttv2_milestone_a_device_evidence/REPORT.md`. It must contain: node IDs, pass/fail, the
   observed violation table from §4, the calibration and its justification, repeat counts,
   teardown confirmation, the §3 defect narrative, and the §5.6 status-page replacement line.
   The brief requires the absolute path of `REPORT.md` to be printed last.

3. **Append the work-log section** `## Hardware checkpoint: Sampling2D stochastic hardware coverage
   2026-08-25` to `tttv2_2d_modules_work_log.md`, matching that file's terse bullet style (see the
   two 2026-08-25 checkpoints at the end for tone and level of detail).

4. **Replace the `CALIBRATION PENDING` docstring text** in the new test file. Two places. Suggested
   replacements, using only numbers actually observed:

   In `test_sampling_2d_wh_galaxy_stochastic_token_in_valid_set`, replace the
   `CALIBRATION PENDING:` paragraph with:

   > Calibration (this host, WH Galaxy `(8, 4)`, two invocations per case per run, three post-fix
   > runs in fresh processes — runs 03/04/05 in `tttv2_milestone_a_gap1_evidence/logs/`): every
   > case observed **0** violations out of 32 in every invocation, including both nucleus cases.
   > The nucleus bound is set to 1, one above that maximum, to leave headroom for a single bf16
   > boundary flip. The tie-free candidate ladder keeps consecutive nucleus probabilities far
   > enough apart that flips are rarer here than in the 1D suite's `torch.randn` logits. A count
   > materially above the bound is a correctness regression, not precision noise — pre-fix this
   > case reported 2, 4 and 9 violations out of 32.

   In `test_sampling_2d_wh_galaxy_per_slot_heterogeneous_parameters`, replace its
   `CALIBRATION PENDING:` paragraph with:

   > Calibration: the eight-slot nucleus group observed **0** violations in every invocation across
   > the three post-fix runs; the bound is one above that. Pre-fix the same group reported 1/8 and
   > 2/8.

5. **Final device-clean check.** `tt-smi -ls` showing 32 boards. Only `ls /dev/tenstorrent | wc -l`
   (= 32) was checked at the start; every log shows clean 32-device teardown, but the explicit
   `tt-smi -ls` confirmation the brief asks for has not been run.

6. **Status-page line** for `models/common/modules/MILESTONE_A_STATUS.md`, row `Sampling2D` — write
   this into `REPORT.md` as a proposal; **do not edit the status page**, it is being rewritten
   wholesale as a separate task. Proposed text:

   > Qualified on WH Galaxy `(8, 4)` for greedy, seeded stochastic and unseeded stochastic paths,
   > including top-p nucleus, padded-vocabulary exclusion under stochastic sampling, per-slot
   > seed stability and per-slot heterogeneous k/p/temperature. Stochastic qualification found and
   > fixed a reciprocal-temperature defect in the device path; `9 passed` across three fresh
   > processes post-fix.

## 6. Deviations and open items to disclose in the report

- **The prescribed host gate ran on hardware.** The brief's host-gate command includes
  `test_sampling_1d.py`, whose `ttnn_mesh_device` cases open `1x1` / `1x2` / `1x8` meshes on this
  Galaxy. It completed `166 passed in 458.41s`. This is in tension with the brief's "do not run any
  1D hardware matrix" prohibition; it was run because the brief explicitly prescribed the command
  and required the 1D suite to stay green after the helper move. Disclose it.
- **An existing host assertion was changed.** `test_prepare_call_refreshes_lazy_sources_by_global_slot`
  asserted the temperature buffer holds `0.5` for `temperature=0.5`. That assertion encoded the
  defect. It was corrected to `2.0` *after* the defect was demonstrated by a failing hardware test,
  not to make a failure disappear. Disclose it explicitly — it is the one change that could
  superficially look like relaxing a test.
- **Likely same-root issue in the 1D suite, not fixed.** `Sampling1D` passes `temp` straight through
  to `ttnn.sampling`, so its `temp` argument *is* the op's `1/T` and the module has no temperature
  semantic of its own. But `test_sampling1d_token_in_valid_set` feeds that same number to an HF
  reference that *divides* by it. That mismatch is consistent with its asymmetric calibrated
  tolerances (`t=0.5` → 6, `t=2.0` → 2 — larger where multiply-semantics widens the nucleus).
  Not investigated further and not changed: the brief prohibits touching 1D module source, and
  changing the 1D test's reference is outside this gap. Recommend a follow-up.
- **The op-level nightly reference disagrees with the op.**
  `tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py:_torch_sampling_reference`
  computes `softmax(values / temp)`, contradicting the nanobind docstring and the kernel. It is
  invisible there because that test only ever passes `temp = ones(32)` and never compares tokens
  against torch. Worth a ticket against the op's test, not this gap.
- **bfloat16 overflow for extremely small temperatures** — see §3.

## 7. Reproducing

Device must be free (`pgrep -af 'pytest|ttnn'` empty) and exactly one pytest may touch it at a time.
Never pipe pytest. The agent harness caps a foreground call at 600 s, so issue these as tracked
background processes.

```sh
LOG=tttv2_milestone_a_gap1_evidence/logs/<name>.log
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA -s --color=no -p no:cacheprovider \
  models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

Whole-file runtime post-fix is ~30 s warm. Note pytest reorders the cases to group by the
`mesh_device` fixture, so the log order is not the file order.
