# Handoff — Gap 1 completion (`Sampling2D` stochastic hardware coverage)

The gap 1 job ran on the 6U WH Galaxy on 2026-08-25 and was **interrupted by an authentication
failure, not by a test or hardware problem**. The technical work is essentially complete and the
device evidence is captured. What remains is write-up, one docstring correction, a lint pass, and a
final device check.

Original brief: [gap1_sampling2d_stochastic_hardware.md](gap1_sampling2d_stochastic_hardware.md).
Read it for context, but **do not re-run the work it describes** — read "What is already done" first.

## Headline: a real module defect was found and fixed

`ttnn.sampling`'s `temp` argument is the **reciprocal** temperature — its kernel multiplies candidate
logits by `temp` before the softmax — but `Sampling2D` was passing raw `T`. The fix is at
[sampling_2d.py:213](../models/common/modules/sampling/sampling_2d.py#L213):

```python
temperature_values[slot] = 1.0 if force_greedy else 1.0 / call.temperature[index]
```

with an explanatory comment at [:202-205](../models/common/modules/sampling/sampling_2d.py#L202-L205).

Why nobody caught it: `1.0` is its own reciprocal, so the bug is invisible at `T = 1.0`, and the
greedy path forces `temp = 1.0` unconditionally. The only pre-existing hardware test is greedy. **The
defect was structurally unreachable by the existing coverage** — which is exactly the argument for
this gap having been worth closing.

`sample_host` was already correct (it divides: `row / call.temperature[row_index]` at
[:256](../models/common/modules/sampling/sampling_2d.py#L256)), so host and device now agree
semantically. No host-path change was needed.

The defect was demonstrated with a failing test *before* the fix (run02), which is the sequence the
brief required.

## What is already done — verified against the working tree and the logs

### Code, uncommitted in the working tree

| File | State |
| --- | --- |
| `models/common/tests/modules/_hf_reference.py` | `_hf_valid_token_set` moved here (untracked, new) |
| `models/common/tests/modules/sampling/test_sampling_1d.py` | imports it; `-31` lines. 1D suite re-run green: **166 passed** |
| `models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py` | new, 9 device cases (untracked) |
| `models/common/modules/sampling/sampling_2d.py` | the reciprocal-temperature fix + comment |
| `models/common/tests/modules/sampling/test_sampling_2d.py` | `+30` lines: new `test_device_temperature_buffer_holds_reciprocal_temperature`, and one pre-existing assertion at [:168](../models/common/tests/modules/sampling/test_sampling_2d.py#L168) corrected — it had pinned the buggy raw-`T` value. **27 passed** |

The 9 device cases: containment × 5 parametrizations, padded-vocab exclusion under stochastic
sampling, seeded repeatability + slot stability, unseeded freshness, per-slot heterogeneous
parameters.

### Device evidence in `tttv2_milestone_a_gap1_evidence/logs/`

All five logs are present and their summary lines and per-case violation counts are confirmed:

| Log | Result | Observed violations |
| --- | --- | --- |
| `run01_calibration.log` | `9 passed in 30.71s` (bounds deliberately unconstrained) | **Only at `T != 1.0`**: `k32-p0.5-t0.8` 2/32 then 9/32; heterogeneous `t0.8` group 1/8. Every `T = 1.0` case 0/32. |
| `run02_prefix_defect_demo.log` | `2 failed, 7 passed in 32.33s` | The failing pair is exactly the two `T != 1.0` cases: 4/32 and 2/8. **This is the test that justifies the fix.** |
| `run03_postfix.log` | `9 passed in 29.46s` | 0 violations on all 23 report lines |
| `run04_postfix.log` | `9 passed in 29.36s` | 0 violations on all 23 report lines |
| `run05_postfix.log` | `9 passed in 32.61s` | 0 violations on all 23 report lines |

Three fresh post-fix processes, as the brief required. The violation pattern — present only where
`T != 1.0`, absent everywhere `T = 1.0`, and gone entirely after the fix — is precisely what a
multiply-vs-divide mismatch predicts. That correlation is the strongest single piece of evidence in
the package; make sure the report states it.

The tests print their own counts (`_report`, line 125), so every passing run records its calibration
in its log. Recover them with:

```sh
grep -h "sampling2d-stochastic" tttv2_milestone_a_gap1_evidence/logs/run03_postfix.log
```

## What remains

Six items. None needs the Galaxy except item 5, and that is a single short run.

### 1. Fix the two `CALIBRATION PENDING` docstrings — and state the calibration honestly

Two docstrings still carry placeholder text:

- [test_sampling_2d_wh_galaxy_stochastic.py:173-176](../models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py#L173-L176) (containment test)
- [:369-370](../models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py#L369-L370) (heterogeneous test)

The `pytest.param` tolerances are **already** set to their final values (`1` for the two `p ∈ (0,1)`
cases, `0` elsewhere) — only the prose is stale.

**Do not write "calibrated from observed boundary violations".** The observed maximum across runs
03-05 is **0**, including in both nucleus cases. The bf16-boundary effect that forced the 1D suite to
allow 2-6 violations **did not manifest here at all** once the temperature was correct. So the
truthful statement is that `1` is deliberate headroom, not an observed requirement:

> Calibrated on this host from runs 03-05 (three fresh processes, `tttv2_milestone_a_gap1_evidence/`):
> the observed violation count was **0 in every case, in every invocation**, including both
> `p ∈ (0, 1)` nucleus cases. The bound of 1 is one above that observed maximum — headroom for the
> bfloat16 softmax/cumsum boundary described above, which did not manifest on this geometry once the
> reciprocal-temperature defect was fixed. Pre-fix, the same two cases showed 4/32 and 2/8
> (`run02_prefix_defect_demo.log`). A run that reports violations here is a regression, not noise.

Adjust the wording per test, keeping the `p ∈ {0.0, 1.0}` cases at zero tolerance with their existing
"any violation is a real defect" rationale.

### 2. `pre-commit` on all touched files

Only ran before the module fix and the two host-test edits, so it has not seen the final state:

```sh
pre-commit run --files \
  models/common/modules/sampling/sampling_2d.py \
  models/common/tests/modules/_hf_reference.py \
  models/common/tests/modules/sampling/test_sampling_1d.py \
  models/common/tests/modules/sampling/test_sampling_2d.py \
  models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py
```

If it reformats the docstrings you just edited, re-read them and confirm the text survived intact.

### 3. Re-confirm the host suites after the docstring edit

```sh
python -m pytest models/common/tests/modules/sampling/test_sampling_1d.py \
                 models/common/tests/modules/sampling/test_sampling_2d.py -q
```

Expect `166 passed` and `27 passed` respectively (they can be run together; record whichever form you
use). A docstring edit should not change either, but the brief's rule is that recorded numbers come
from runs you actually performed.

### 4. Write `tttv2_milestone_a_gap1_evidence/REPORT.md`

Sections, following the house format used by `tttv2_milestone_a_device_evidence/REPORT.md`:

1. **Summary** — one paragraph. Lead with the defect found and fixed; say plainly that the gap is
   closed and that closing it surfaced a real module bug.
2. **Environment** — commit, branch, host, 32 devices, build type, Python. Note the tree is dirty by
   design (the fix and the new tests are the deliverable).
3. **The defect** — `ttnn.sampling` `temp` is `1/T`; the module passed `T`; invisible at `T = 1.0`
   and on the greedy path; the fix; why the existing greedy hardware test could not catch it.
4. **Results table** — one row per node ID per run, with the observed violation counts. Every row's
   log cell must point at a real file in `logs/`.
5. **Calibration** — the table from "What is already done" above, plus the pre-fix/post-fix
   correlation argument.
6. **Caveats and gaps** — at minimum: distributional correctness of the RNG itself is *not* tested
   (only support containment and determinism); device tokens are deliberately never compared to
   `sample_host` (31-bit vs 63-bit seed derivation, different generators); `top_k > 32` is out of
   contract and untested; no trace/capture coverage.
7. **The 1D follow-up** — see item 6 below.

### 5. Final device-clean confirmation

Only `/dev/tenstorrent | wc -l` was checked at the start. Teardown lines look normal in every log,
but the brief's finish condition wants an explicit final state:

```sh
tt-smi -ls > tttv2_milestone_a_gap1_evidence/logs/99_tt_smi_after.log 2>&1
ls /dev/tenstorrent | wc -l
```

Confirm 32 boards present. No reset was needed during this job and none should be needed now — if
`tt-smi -ls` looks wrong, record that rather than resetting reflexively.

### 6. Append the work-log section, and file the 1D follow-up

Append to `tttv2_2d_modules_work_log.md`, matching its terse bullet style:

```
## Hardware checkpoint: Sampling2D stochastic hardware coverage 2026-08-25
```

Cover: what was added, the defect and its root cause, the pre-fix/post-fix evidence chain, the three
fresh post-fix processes, the host gates, and the 1D follow-up.

**The 1D follow-up is worth stating precisely, because it is a testable prediction.** The 1D suite's
calibrated tolerances are asymmetric in exactly the direction a multiply-vs-divide mismatch
predicts — `t=0.5 → 6` violations allowed, `t=2.0 → 2`
([test_sampling_1d.py:859-861](../models/common/tests/modules/sampling/test_sampling_1d.py#L859-L861)).
`Sampling1D` passes `temp` straight through to the op, so **its** `temp` argument already *is* the
op's `1/T`; the mismatch there is in the test's HF reference, which warps with the raw value. The
prediction: if `test_sampling1d_token_in_valid_set` passed `1/temp` to `_hf_valid_token_set`, those
tolerances should collapse toward 0, the way the 2D ones did.

This was **not** touched and should not be touched in this job — it is 1D test surface, it needs its
own hardware run, and it is out of this brief's scope. File it as a follow-up in the work log with
the prediction written down so whoever picks it up can falsify it cheaply.

### 7. Draft the `MILESTONE_A_STATUS.md` replacement line

Draft, in the report, replacement text for the `Sampling2D` row. The current row reads:

> | Sampling2D | Included in final 1259-test host gate | Qwen forced argmax repeated with exact tokens and padded-vocabulary exclusion | Qualified for the required forced-argmax hardware case; stochastic hardware is not recorded |

**Put the proposed text in `REPORT.md`. Do not edit `MILESTONE_A_STATUS.md`** — it is being rewritten
wholesale as a separate task and three parallel edits will collide.

## House rules that still apply

- Do not `git commit`, `push`, `checkout`, `stash`, or `reset`. Leave the tree dirty for review.
- Do not edit `models/common/modules/MILESTONE_A_STATUS.md`, `models/common/modules/README.md`, or
  `tttv2_2d_modules_plan.md`.
- Do not modify any `models/common/modules/**/*_1d.py` implementation file, and do not touch the 1D
  sampling tolerances (item 6).
- Do not relax a tolerance or a threshold. The bounds are settled; if a run now reports violations,
  that is a regression to report.
- Do not re-run the device matrix "to be safe" — five runs are captured and sufficient. Item 5 is a
  `tt-smi` call, not a pytest run.
- One pytest process on the device at a time; never pipe pytest.

## Finish condition

Both docstrings corrected and honest about the observed zero; `pre-commit` clean on all five files;
host suites re-confirmed; `tttv2_milestone_a_gap1_evidence/REPORT.md` written with every claim
pointing at a log on disk; `99_tt_smi_after.log` captured showing 32 boards; the work-log checkpoint
appended including the 1D follow-up prediction; the `MILESTONE_A_STATUS.md` replacement line drafted
in the report but not applied. Print the absolute path of `REPORT.md` as the last line.

Nothing else. If you finish early, stop.
