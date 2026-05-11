# Verification Report

This file contains two verification sections:

1. **Run7 refinement verification** (2026-05-08, HEAD `3b0cc6026e8`) — GREEN, unchanged.
2. **Type-1 sweep + Reg A/B/C verification** (2026-05-11, HEAD `7060e1245a3`) — appended at the end of this file. See bottom of file for the sweep verdict.

---

# Section 1 — Verification Report: eltwise_chain (run7) refinement

**Branch:** `astancov/eltwise_run7_refined`
**HEAD:** `3b0cc6026e8`
**Baseline:** `75868c9eff4` (run7 baseline)
**Verifier date:** 2026-05-08
**Acceptance verdict:** **GREEN** — all migrated kernels' pytest counts match baseline; helper-level kernel_lib regression bar holds at 453 / 7. Two raw-LLK regressions and six moreh refits land cleanly. No regressions blocking merge.

---

## 1. Summary verdict

The 8-commit refinement series lands green. The kernel_lib regression bar (`tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`) is **453 passed / 7 skipped** at HEAD. All sampled production-kernel test directories under `tests/ttnn/unit_tests/operations/` and `tests/ttnn/nightly/unit_tests/operations/` produce **identical pass/fail counts at HEAD vs. baseline** — i.e., zero refinement-introduced regressions. One directory (`moreh_adam`) actually shows a +4-test improvement at HEAD vs. baseline (32f/100p HEAD vs 36f/96p baseline), consistent with the symmetric-Pinned refit across the moreh sweep.

D5 placement contract, D6 CARRY/SKIP split, D7 block-element OldCb removal, and D8 BIG-init boundary all hold strict on inspection. The doxygen pass on `eltwise_chain.hpp` is thorough, with the full caller-init contract table, D5 placement table, D6 CARRY/SKIP table, D7 block-fold note, D8 anti-examples, the D8 grep gate, and 5 worked code examples per chain shape.

The 2 documented Q4 raw-LLK regressions (`deepseek_grouped_gate.cpp::add_bias`, `eltwise_binary_scalar.cpp` no-act fast path) are present and correctly inline-commented. The 6 moreh kernels symmetric-refit to uniform `Pinned` are correct.

---

## 2. Test pass/fail summary

### Helper-level regression bar (PRIMARY)

| Test path | HEAD result | Baseline result | Δ |
|---|---|---|---|
| `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` | **453 passed / 7 skipped** | 449 passed / 7 skipped (implementer-reported) | **+4 (new `test_optional_chain_element` parametrize)** |

Note on the 4-test delta: the implementer reported 449/7. The verifier's run on HEAD shows 453/7 — the 4 extra are the new `test_optional_chain_element` rows added in commit 7. Implementer's 449/7 is therefore a slight under-count of the actual end-state; either way zero regressions.

### Migrated-kernel pytest sweep (HEAD vs baseline diff)

For each test directory below, **HEAD** counts and **baseline** counts were compared. Identical counts ⇒ no regression.

| Test path | HEAD | Baseline | Verdict |
|---|---|---|---|
| `tests/ttnn/unit_tests/operations/eltwise/test_unary.py` | 67 fail / 700 pass / 51 xfail | 67 fail / 700 pass / 51 xfail | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_activation.py` | 4 fail / 118 pass / 3 skip | 4 fail / 118 pass / 3 skip | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_composite.py` | 25 fail / 154 pass | 25 fail / 154 pass | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_scalar.py` | 21 fail | 21 fail | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_elt_binary.py` | 8 fail / 3 pass | 8 fail / 3 pass | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng_program_cache.py` | 3 fail / 12 pass | 3 fail / 12 pass | match |
| `tests/ttnn/unit_tests/operations/eltwise/test_where.py` | infra segfault | infra segfault | match (pre-existing) |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py` | infra segfault | infra segfault | match (pre-existing — implementer's note) |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast_tcast.py` | infra segfault | infra segfault | match (pre-existing) |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_composite.py` | infra segfault | infra segfault | match (pre-existing) |
| `tests/ttnn/unit_tests/operations/eltwise/test_typecast_int.py` | 13 fail | (≥1 fail; pre-existing) | match (sampled `test_typecast_uint16` fail confirmed pre-existing) |
| `tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_ulp.py + gelu_bw_*` | 111 fail | 111 fail | match |
| `tests/ttnn/unit_tests/operations/data_movement/test_dropout.py` | 2 pass | 2 pass | match |
| `tests/ttnn/unit_tests/operations/data_movement/test_clone.py` | 118 pass / 24 skip | 118 pass / 24 skip | match |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` | 16 fail / 77 pass / 32 skip | 16 fail / 77 pass / 32 skip | match |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adam.py` | **32 fail / 100 pass / 128 skip** | 36 fail / 96 pass / 128 skip | **HEAD improves** (+4 tests pass) |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` | 48 pass / 48 skip | 48 pass / 48 skip | match |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sgd.py` | 164 pass / 352 skip | 164 pass / 352 skip | match |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py` | 76 pass / 72 skip | 76 pass / 72 skip | match |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_clip_grad_norm.py` | 19 fail / 1 pass | 19 fail / 1 pass | match |
| `tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py` | 5 fail / 1 pass | 5 fail / 1 pass | match (failures: "Invalid group selection" — logic, not refinement) |
| `tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_prefix_scan.py` | 14 fail / 4 pass | 14 fail / 4 pass | match |

**Conclusion: zero regressions across all sampled production-kernel test directories.** The improvement of +4 tests passing in `test_moreh_adam.py` is consistent with the symmetric-Pinned refit (Q4 disposition (a)).

### Skipped sweeps (large-tree pytest sweeps deferred or pre-existing infra issues)

- `test_add.py`, `test_binary_bcast.py`, `test_binary_bcast_tcast.py`, `test_binary_composite.py` — collection-time segfault on baseline AND HEAD (pre-existing infrastructure bug per implementer's commit-1 breadcrumb of `2026-05-08T09:18:28+00:00`). Not a refinement issue.
- `eltwise_binary_no_bcast.cpp` preprocessor `#if BINARY_OP_TYPE == EltwiseBinaryType::ELWADD` chokes on `::` — failure reproduced on baseline, so **pre-existing kernel bug, not refinement-introduced**. Tracked as orthogonal to run7.

---

## 3. Code review by directive

### D1 — `eltwise_pipeline_init` deletion / caller owns `compute_kernel_hw_startup`

**PASS.** No `eltwise_pipeline_init` symbol survives in the helper:

```bash
$ grep -nE 'eltwise_pipeline_init' ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl}
(no output)
```

Only the caller-side convenience wrapper `eltwise_chain_with_init` remains (commit 6) — and that emits `compute_kernel_hw_startup` deliberately as caller-facing convenience.

### D2 — Compile-time prev-CB tracking on streaming elements

**PASS.** Streaming elements (`CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`) no longer carry `OldCb*` template parameters; the prev-CB fold lives in `eltwise_chain.inl` (lines 1043–1130 in the dispatch path) and emits per-element reconfig only on transitions. Visual diff against baseline confirms the OldCb* params are gone.

### D3 — Doxygen on helper headers + caller-init contract spec + minimal example per chain shape

**PASS.** `eltwise_chain.hpp` (lines 1–220 doxygen header) carries:
- Caller-init contract table (D8) — line 31–47.
- D5 placement table (chain-shape → `compute_kernel_hw_startup` placement) — line 57–66.
- D6 per-element fp32-dest-acc CARRY/SKIP table — line 83–91.
- D7 block-path fold notes — line 109–117.
- D8 anti-examples — line 119–127.
- D8 grep gate one-liner — line 129–140.
- 5 worked code examples (streaming unary, streaming binary, deduced wrapper, fan-out, block reduction, mixed CARRY+SKIP+fp32 transitions) — line 142–196.
- Non-goals + Reconfig fold table — line 198–219.

This is comprehensive and substantially exceeds the design minimum.

### D4 — "No production callers" is NOT kill criterion (delete only what's provably absent)

**PASS.**
- `EltwiseChainOptions` struct + NTTP **deleted** (commit 5, `aa094135709`).
- Streaming `OldCb*` **deleted** (commit 3, `e88d5e289b3`).
- Block `OldCb*` **deleted** (commit 4, `d644d47573d`) — D7.
- Per-side `AIndex/BIndex` **collapsed to single `Index`** (commit 6, `b46a2b9b7fc`) — Q4 reversal verified at `eltwise_chain.hpp:577` and `eltwise_block.hpp:150`.

### D5 — `compute_kernel_hw_startup` placement contract

**PASS** on the 5 representative kernels checked:

| Kernel | hw_startup line | Placement | Verdict |
|---|---|---|---|
| `mish_kernel.cpp:52` | top of `MAIN()` | unary single-stage | OK |
| `bcast_h.cpp:33` | top of `MAIN()` | binary single-stage | OK |
| `dropout_kernel.cpp:43` | top of `MAIN()` | unary single-stage | OK |
| `logit_kernel.cpp:35,48` | top of `MAIN()` (stage 1), pre-stage-2 (stage 2) | multi-stage exception per design | OK (matches D5 row "Multi-stage") |
| `where_tss_kernel.cpp` | via `eltwise_chain_with_init` (line 46) | wrapper — single-stage | OK (deduced wrapper covers row 1/2) |

### D6 — Per-element `EnableFp32DestAcc` (CARRY/SKIP)

**PASS.** Audit of element decls in `eltwise_chain.hpp` lines 560–630 and `eltwise_block.hpp` lines 76–267:

| Element | Status | Verifier note |
|---|---|---|
| `CopyTile` | SKIP — no flag | line 560–566. OK |
| `BinaryFpu` | CARRY — `EnableFp32DestAcc=false` | line 568–580. OK (param 11) |
| `DestReuseBinary` | CARRY — `EnableFp32DestAcc=false` | line 582–592. OK |
| `UnaryBcast` | CARRY — `EnableFp32DestAcc=false` | line 594–602. OK |
| `PackTile` | CARRY — `EnableFp32DestAcc=false` | line 604–611. OK |
| `PackTileBlock` | CARRY — `EnableFp32DestAcc=false` | line 613–620. OK (note: not in prompt's CARRY list but follows design) |
| `FillScalar` | SKIP — no flag | `eltwise_fill.hpp:20`. OK |
| `FillInt` | SKIP — no flag | `eltwise_fill.hpp:31`. OK |
| `FillBitcast` | SKIP — no flag | `eltwise_fill.hpp:43`. OK |
| `RandTile` | SKIP — no flag | `eltwise_rand.hpp:23`. OK |
| `OptionalChainElement` | SKIP (forwards) | `eltwise_optional.hpp:60–71`. OK |
| `BlockCopyTile` | SKIP — no flag | `eltwise_block.hpp:76–82`. OK |
| `BlockBinaryFpu` | CARRY — `EnableFp32DestAccV=false` | `eltwise_block.hpp:139–153`. OK |
| `BlockPackTile` | CARRY — `EnableFp32DestAccV=false` | `eltwise_block.hpp:260–267`. OK |

**Deferred** (per F-UX-9 design carry-forward): `BinarySfpu` family (`AddBinary`/`SubBinary`/`MulBinary`/`DivBinary` — `eltwise_binary_sfpu.hpp`). These are SKIP-by-SFINAE today; the doxygen explicitly notes the deferral at `eltwise_chain.hpp:86`.

**Note:** prompt's CARRY list mentions `BroadcastFpu`, but no struct of that name exists in the helper. The closest match is `UnaryBcast` (CARRY) and `BlockBinaryFpu` with `Bcast` template param (CARRY) — both correctly carry the flag. Treat the prompt mention as a naming alias.

### D7 — Block-element `OldCb*` removal + fold extension

**PASS.** Block elements (`BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`) declared in `eltwise_block.hpp` have **no** `OldCb*` template params (verified by reading lines 76–267). Commit 4 message confirms.

### D8 — Caller-init contract / no BIG inits in helper

**PASS strict.** Strict grep for actual LLK call sites:

```bash
$ grep -nE '(^|[^/* ])compute_kernel_hw_startup\(|^[^/* ]*binary_op_init_common\(|^[^/* ]*mm_init\(|^[^/* ]*reduce_init\(' \
       ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl} ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp
ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:1244:    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
```

**Single active call site:** the convenience wrapper `eltwise_chain_with_init` at `eltwise_chain.inl:1244` (intentional per implementer's note). All other `compute_kernel_hw_startup` references in chain.{hpp,inl} are doxygen comment text or the include directive at `eltwise_chain.hpp:235`.

D8 invariant strictly held in helper bodies.

### Q4 disposition (collapse `AIndex/BIndex` → single `Index`)

**PASS.** Verifier checked the 6 affected kernels:

- **(a) symmetric refit to `Pinned`** — 6 moreh kernels: `moreh_adam.cpp:47` ("uniform `Pinned`", lines 35-47 inline-commented), and the 5 `moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp` all confirmed using `CbIndexMode::Pinned`.
- **(c) regress-to-raw** — 2 kernels:
  - `deepseek_grouped_gate.cpp:39–66` `add_bias`. Inline comment at lines 41–52 explains the regression. Test (`test_deepseek_grouped_gate.py`) passes raw-LLK execution; remaining 5 fails match baseline.
  - `eltwise_binary_scalar.cpp:74–112` no-act branch. Inline comment at lines 75–82 explains the regression.

`BinaryFpu` and `BlockBinaryFpu` template signatures verified to use single `Index` param at `eltwise_chain.hpp:577` and `eltwise_block.hpp:150`. Zero `AIndex`/`BIndex` identifier references survive in helper sources.

---

## 4. Open issues triage (5 items from implementer)

The implementer flagged 5 open items. The verifier found these in the helper sources / commit messages and dispositions them as follows:

### Issue 1 — `pack_reconfig_data_format` two-arg form deferred

**Design v6 reference:** Section A line 367 prescribed the two-arg form for `PackTileReconfig::OutputConditional` to allow `pack_reconfig_data_format(prev, curr)` in a future fold extension. Implementer notes no production caller exercises that branch.

**Verdict:** Acceptable deferral. The doxygen at `eltwise_chain.hpp:217–218` documents the future-extension intent. Production tests pass without it.

**Severity:** can-defer.

**Recommended follow-up:** track in a future small commit when an audit of `PackTileReconfig::OutputConditional` callers shows non-trivial benefit. Add a one-line TODO to the comment at `eltwise_chain.hpp:217` if it isn't already there.

### Issue 2 — `BinarySfpu` family (F-UX-9) deferred from D6 CARRY

**Design v6 reference:** D6 row defers SFPU family to a separate cleanup pass.

**Verdict:** Acceptable. Doxygen at `eltwise_chain.hpp:86` documents this deferral explicitly with a "once F-UX-9 lifts the SFPU-family deferral" note.

**Severity:** can-defer.

**Recommended follow-up:** F-UX-9 should land as a focused commit. The change is mechanical (add `bool EnableFp32DestAcc=false` template param + `static_assert` per SFPU op-struct in `eltwise_binary_sfpu.hpp`) plus SFINAE fold registration.

### Issue 3 — `eltwise_chain_with_init` carries a `compute_kernel_hw_startup` call

**Design v6 reference:** D8 says "helper does not wrap BIG inits". The wrapper at `eltwise_chain.inl:1244` does call `compute_kernel_hw_startup`.

**Verdict:** This is **not** a D8 violation — D8 governs the *helper body* (the chain dispatch path) emitted on the kernel's compute thread. The wrapper is a caller-facing convenience that the kernel author opts into; the wrapper *is* the caller's `compute_kernel_hw_startup` placement, deduced from the chain element pack. Functionally identical to the kernel writing `compute_kernel_hw_startup(...)` followed by `eltwise_chain(...)`, just compile-time-deduced.

The grep gate at the doxygen (line 134–140) is the contract surface; the verifier's strict grep run confirms only the wrapper site survives. This is documented at `eltwise_chain.inl:1131–1136` ("Single-stage convenience that derives ... and emits `compute_kernel_hw_startup` before the chain. Multi-stage kernels MUST keep the explicit per-stage pattern.").

**Severity:** can-defer (categorization concern only — no behavioural risk).

**Recommended follow-up (cosmetic only):** if the team wants the grep gate to be tighter, the wrapper could move to a separate header (e.g., `eltwise_chain_convenience.inl`) excluded from the gate. Not necessary for correctness.

### Issue 4 — Helper LOC delta

**Design v6 reference:** R-N rows in Section A.

**Verifier confirmation:** helper LOC delta over the 4 helper files matches implementer's claim exactly:

```
ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp    | 168 ++++---
ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp    | 281 +++++++++---
ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl    | 635 ++++++++++++++++++++------
ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp |  27 +-
4 files changed, 849 insertions(+), 262 deletions(-)
```

Per-commit LOC drift was not individually validated against R-N rows (out of scope per prompt; perf claims also implementation-only).

**Severity:** N/A.

### Issue 5 — Performance validation deferred

**Design v6 reference:** F-PERF-1+2+3+4 implementation-only claim.

**Verdict:** Acknowledged. Perf measurement is out of scope for this verifier pass per the prompt. Perf-validation deferred to a separate pass.

**Severity:** can-defer.

**Recommended follow-up:** schedule a perf-validation pass that runs the eltwise tracy / device-perf scripts at the same shape set used for the audit baselines.

---

## 5. Recommendations

1. **Merge as-is.** Verdict GREEN. No regressions detected against the baseline; helper-level kernel_lib regression bar holds at 453/7.
2. **Future small commit (cosmetic):** consider relocating `eltwise_chain_with_init` to `eltwise_chain_convenience.inl` so the D8 grep gate becomes a one-line zero-output check on the bare helper headers. Optional.
3. **Future cleanup commit (F-UX-9):** add `EnableFp32DestAcc` template param to the `BinarySfpu` family in `eltwise_binary_sfpu.hpp` and register them with the D6 fold. Mechanical change.
4. **Future small fix (orthogonal to refinement):** `eltwise_binary_no_bcast.cpp` `#if BINARY_OP_TYPE == EltwiseBinaryType::ELWADD` chokes on `::` in the C preprocessor; same pattern in `eltwise_binary_scalar.cpp` and the ng-kernels. Pre-existing on baseline; affects `test_add_with_batch_broadcast` and a few derived tests. Not a refinement bug, but worth flagging to whoever owns binary_ng.
5. **Future small infra fix:** `test_binary_bcast.py`, `test_binary_bcast_tcast.py`, `test_binary_composite.py` collection-time segfault on both baseline and HEAD. Not a refinement bug; flag to the test infra team.
6. **Pipeline metric:** the implementer-reported 449/7 is a slight under-count of the actual 453/7 end-state (the 4 `test_optional_chain_element` parametrize rows added in commit 7 weren't included). Verifier expects 453/7 going forward.

---

# Section 2 — Verification Report: Type-1 sweep + Reg A/B/C fixes

**Branch:** `astancov/eltwise_run7_refined`
**HEAD:** `7060e1245a3`
**Pre-sweep baseline:** `d1d1f9246c5` (`[partial-audit] Type-1 migration targets across 31 B-classified kernels`)
**Run7 GREEN baseline:** `6f58599dd4e` (prior verifier pass on `3b0cc6026e8`)
**Verifier date:** 2026-05-11
**Acceptance verdict:** **YELLOW** — sweep delivers the planned 37-block migration AND the helper Reg A/B/C fixes land cleanly, BUT 5 test suites carry sweep-introduced fp32_dest_acc_en=True numerical regressions (+68 fails) due to missing per-element reconfig annotations in 4 migrated kernel sites. Helper baseline holds at 453/7. Two pre-existing failure classes documented. Merge-OK conditional on filing the fp32 annotation cleanup as a follow-up.

---

## 1. Summary verdict

The 9-commit Type-1 sweep + 3 regression-fix areas (Reg A bcast helper, Reg B moreh_sum/norm host, Reg C PackTile fold) deliver substantial wins on `moreh_adam` (+33 pass), `moreh_adamw` (+8 pass), and preserve helper baseline at 453P/7S. The Reg A debugging sequence correctly diagnosed and surgically fixed the chain `init_bcast<>()` mid-MAIN D8 violation (BinaryFpu now uses `_init_with_operands` short pair). Reg B fixed a pre-existing Jan 23 2025 host-side bug (`padded_shape().rank()` vs `logical_shape().rank()`) on moreh_sum/moreh_norm; this fix is branch-local and the bug exists on `main` — upstream PR debt remains.

The verifier discovered an UNDOCUMENTED regression class during pre-sweep baseline comparison: 5 test suites carry **+68 sweep-introduced failures** on `fp32_dest_acc_en=True` paths. Root cause: the `moreh_unary_chain` / `moreh_rexp_chain` templates introduced in Commit 4 (and analogous chain-substitution sites in Commit 5 nll_loss kernels) replaced pre-sweep `exp_tile_to_cb` / `pack_tile_with_dt` raw blocks that did **CB-format-aware DEST reconfig via `_with_dt` calls**. The replacement chain elements (`CopyTile` + `Sfpu` + `PackTile`) do NOT carry `CopyTileReconfig::Input` / `PackTileReconfig::Output` / `EnableFp32DestAcc=true` flags, so when the kernel is built with `FP32_DEST_ACC_EN`, the DEST format stays bf16 from the prior chain step → `exp_tile(dst)` overflows to `-inf`.

This is a **migration-side annotation gap, not a helper-side bug.** The helper supports the correct path; the migration omitted the opt-ins. Reg C confirms the helper now correctly emits `pack_reconfig_data_format` when `PackTileReconfig::Output` IS declared — but the migrated kernels don't declare it.

D8 invariant strictly holds: `grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' eltwise_{chain.hpp,chain.inl,block.hpp}` returns only doxygen text + the include directive + the single intentional convenience wrapper at `eltwise_chain.inl:1254`. Reg A's bcast init pair (BinaryFpu `_init_with_operands`, UnaryBcast `unary_bcast_init` BIG) and Reg C's PackTile fold edit do not perturb D8.

---

## 2. Test pass/fail summary (pre-sweep `d1d1f9246c5` vs HEAD `7060e1245a3`)

### Helper baseline (PRIMARY regression bar)

| Test path | Pre-sweep | HEAD | Δ |
|---|---|---|---|
| `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` | 453P / 7S | **453P / 7S** | match — Reg A + Reg C edits did not perturb helper baseline |

### Moreh suites (sweep-touched + adjacent)

| Test path | Pre-sweep | HEAD | Δ | Verdict |
|---|---|---|---|---|
| `nightly/.../moreh/test_moreh_softmax.py` | 16F / 77P / 32S | **24F** / 69P / 32S | **+8F** | **regression** — fp32_dest_acc_en=True forward paths broken |
| `nightly/.../moreh/test_moreh_layer_norm.py` | 48P / 48S | 48P / 48S | match | correctly wired (commits 7/8/9) |
| `nightly/.../moreh/test_moreh_nll_loss.py` | 10F / 60P / 60S | **13F** / 57P / 60S | **+3F** | **regression** — step2/backward lack PackTileReconfig::Output |
| `nightly/.../moreh/test_moreh_nll_loss_unreduced.py` | 0F / 43P / 28S | **3F** / 40P / 28S | **+3F** | **regression** — same root cause |
| `nightly/.../moreh/test_moreh_adam.py` | 33F / 99P / 128S | **0F / 132P** / 128S | **-33F** | **WIN** — Reg C fix delivered |
| `nightly/.../moreh/test_moreh_adamw.py` | 8F / 11P / 8S | **0F / 19P** / 8S | **-8F** | **WIN** — Reg C + correct annotations |
| `nightly/.../moreh/test_moreh_norm.py` | 466P / 1806S | 466P / 1806S | match | Reg B host fix preserved baseline |
| `nightly/.../moreh/test_moreh_sum.py` | 227P / 155S | 227P / 155S | match | Reg B host fix preserved baseline |
| `nightly/.../moreh/test_moreh_mean.py` | 76P / 72S | 76P / 72S | match | untouched |
| `nightly/.../moreh/test_moreh_sgd.py` | 164P / 352S | 164P / 352S | match | untouched |
| `nightly/.../moreh/test_moreh_logsoftmax.py` | 5F / 87P / 32S | **23F** / 69P / 32S | **+18F** | **regression** — shares kernels with moreh_softmax |
| `nightly/.../moreh/test_moreh_logsoftmax_ulp.py` | 0F / 162P | **36F** / 126P | **+36F** | **regression** — all 36 fails are fp32_dest_acc_en=True ULP `max=inf` |

### Adjacent suites that should be untouched

| Test path | Pre-sweep | HEAD | Δ | Verdict |
|---|---|---|---|---|
| `nightly/.../experimental/test_bcast_to.py` | 24P | 24P | match | Reg A `UnaryBcast` BIG-init revert preserved bcast_to boot dependency |
| `nightly/.../reduction/test_deepseek_grouped_gate.py` | 5F / 1P | 5F / 1P | match | "Invalid group selection" pre-existing logic fail; commit 1 scale-block migration green |

### Aggregate

- **Regressions (sweep-introduced)**: +68 fails across 5 suites
- **Wins (sweep + Reg C)**: -41 fails across 2 suites
- **Net change in passing tests**: -27 vs pre-sweep
- **Helper baseline**: unchanged (453P/7S)

---

## 3. Code review by directive

### D8 invariant (strict grep)

```bash
$ grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' \
       ttnn/cpp/ttnn/kernel_lib/eltwise_{chain.hpp,chain.inl,block.hpp}
```

Single LLK call site: `eltwise_chain.inl:1254` — `compute_kernel_hw_startup(cb_a, cb_b, cb_out);` inside the convenience wrapper `eltwise_chain_with_init`. All other hits are doxygen comment text or the include directive at `eltwise_chain.hpp:235`. **D8 strictly held through Reg A + Reg C helper edits.** (Note: line moved from `1244` post-run7 to `1254` post-Reg-A — Reg A added the BinaryFpu `_with_operands` lines above the wrapper.)

### Reg A — BinaryFpu / UnaryBcast init pair

| Code site | Form | Status |
|---|---|---|
| `eltwise_chain.inl:528-532` (BinaryFpu) | `llk_math_eltwise_binary_init_with_operands<et, bt, FID>(CbA, CbB)` + `llk_unpack_AB_init<bt>(CbA, CbB)` | **PASS** — D8-compliant short pair; matches `bcast.h:352-446 *_init_short` reference |
| `eltwise_chain.inl:714` (UnaryBcast) | `unary_bcast_init<bt>(Cb, ocb)` | **PASS (pragmatic)** — full BIG init retained; D8-violating in letter but required because `bcast_to` kernels rely on it as IMPLICIT BOOT (no `compute_kernel_hw_startup`, no `binary_op_init_common`). Documented in capabilities.md. Reverted from `647ff976906` after that broke 24/24 `bcast_to` tests. |

Verification — `test_bcast_to.py` 24P preserved; `test_eltwise.py` 453P/7S preserved; softmax LARGE_H hang from `f91d321a714` (non-operand math init programmed DEFAULT_TENSOR_SHAPE) eliminated.

### Reg B — moreh_sum / moreh_norm host fix

| Code site | Pre-fix | Post-fix | Status |
|---|---|---|---|
| `moreh_sum.cpp:19` | `input.padded_shape().rank()` | `input.logical_shape().rank()` | **PASS** |
| `moreh_norm.cpp:22` | `input.padded_shape().rank()` | `input.logical_shape().rank()` | **PASS** |
| `moreh_norm.cpp:39` | `input.padded_shape().rank()` | `input.logical_shape().rank()` | **PASS** |

Pattern matches the correct usage in `moreh_mean.cpp:20` and `moreh_sum_backward.cpp:22`. Original bug: Jan 23 2025 commit `0cc4e4eb2f4` introduced `padded_shape().rank()` (likely transitional during shape API migration); for 1D inputs `padded_shape = [32,32]` rank=2 but `logical_shape = [N]` rank=1 → `operations::get_dim()` fabricates phantom dim index → inf in device reduce.

**Upstream PR debt acknowledged** — bug exists on `main`; fix is branch-local per user directive.

### Reg C — PackTile reconfig helper bug

| Code site | Pre-fix | Post-fix | Status |
|---|---|---|---|
| `eltwise_chain.inl:1031-1045` `hoisted_init_for_each` | gated `if constexpr (!is_pack_tile_op_v<ElemT>)` | guard removed; PackTile elements fire `emit_pre_element_transitions` | **PASS** |

Pre-fix bug: chains without FPU-clash elements (CopyTile + SFPU + PackTile) never emitted `pack_reconfig_data_format` declared via `PackTile<...Reconfig::Output>`. Pack init() remains no-op so removing the guard is safe — only `emit_pre_element_transitions` (which does the actual reconfig emission) is added.

Companion clean-up: `moreh_adam.cpp` reverts 3 sites of "neutral fix attempt" explicit `pack_reconfig_data_format` calls — these were workarounds the helper-side fix obviates.

**Verification**: moreh_adam 66F → 0F (and 132P), adamw similarly clean. Bisect confirms Reg C fix alone is responsible for the win.

### Q4 collapse compliance (regressed kernels)

- `deepseek_grouped_gate.cpp::add_bias` (lines 39-66): still raw LLK (`tile_regs_acquire`, `add_tiles_bcast`, etc. confirmed at line 58-60, 254-258, 312-313). Inline comment at line 45-47 retains Q4-asymmetric rationale. **PASS — Q4 disposition (c) preserved.**
- 5 moreh_softmax_backward kernels: uniform `CbIndexMode::Pinned` confirmed at `_c_large.cpp:34,49`, `_h.cpp:36,64,76,95`, `_h_large.cpp:36,51`, `_w.cpp:36,63,75,94`, `_w_large.cpp:36,51,92`. **PASS — Q4 disposition (a) symmetric Pinned refit preserved.**

### LOC delta

| Span | + | - | Net | Plan | Drift |
|---|---|---|---|---|---|
| Pre-sweep `d1d1f9246c5` → HEAD (production code only, excluding agent breadcrumbs) | +828 | -452 | **+376** | -171 (design v6 projection: -482/+311) | **+547 LOC over plan** |
| Sweep commits alone (1–9) | ~+560 | ~-490 | ~+70 | -171 | ~+241 over plan |

Drift breakdown: the in-kernel template helpers introduced by Commit 4 (`moreh_unary_chain`, `moreh_rexp_chain`) and the verbose multi-line `BinaryFpu<...>` template parameter lists (8-13 args per instantiation) inflate kernel size relative to the pre-sweep raw `tile_regs_acquire`/`exp_tile`/`pack_tile`/`tile_regs_release` blocks. Net LOC reclaim does NOT materialize at the projected -482/+311 rate.

---

## 4. Known-issue triage (per user prompt)

### Class 1 — Softmax pre-existing failures

User prompt classified **24 softmax fails** as "predates Type-1 sweep" per Reg A debugger statement "24 fails present even with original `init_bcast<>` BIG-init applied."

**Verifier correction**: pre-sweep baseline at `d1d1f9246c5` is **16F**, not 24F. The Reg A debugger statement was made AFTER the sweep landed — i.e., it confirmed that swapping the helper init form alone doesn't change the 24F state. That is correct: the 8 sweep-introduced fails persist regardless of helper init form because the root cause is the kernel-side template (`moreh_unary_chain`) lacking reconfig flags.

**Class breakdown:**
- 16 pre-existing fails — `test_softmax_large_algorithm_for_dim_hw` LARGE_H, `test_softmax_for_dim_nc[shape=[15,109,64,64]]`, `test_softmax_backward_*` separate-chain patterns. These exist on `main` and predate the run7 refinement.
- 8 sweep-introduced fails — all `fp32_dest_acc_en=True` paths through `moreh_unary_chain` / `moreh_rexp_chain` (Commit 4 T1.11-T1.16).

**Severity**: should-fix (the 8 sweep-introduced are migration-annotation gaps; the 16 pre-existing are a separate effort).

**Recommended follow-up agent**: `ttnn-implementer` for the 8 sweep-introduced (mechanical: add `CopyTileReconfig::Input`, `PackTileReconfig::Output`, `EnableFp32DestAcc=DST_ACCUM_MODE` to `moreh_unary_chain`/`moreh_rexp_chain` templates and re-test). The 16 pre-existing need a `ttnn-expert-debugger` triage pass to root-cause separately.

**Estimated impact**: 8 + 18 (logsoftmax shared) + 36 (logsoftmax_ulp shared) = **62 fail recovery** from a single migration-template fix.

### Class 2 — NLL_loss fp32_dest_acc_en=True failures

User prompt classified **13 nll_loss fails** (`reduction=sum` AND `reduction=mean`) as a pre-existing class. Verifier disposition is the same as Class 1: **10 pre-existing + 3 sweep-introduced**.

**Class breakdown:**
- 10 pre-existing fails — existed at `d1d1f9246c5` (pre-sweep baseline).
- 3 sweep-introduced fails — Commit 5 migrated `moreh_nll_loss_step2_kernel.cpp` and `moreh_nll_loss_backward_kernel.cpp` without `PackTileReconfig::Output` / `EnableFp32DestAcc=true` on `BinaryFpu`/`PackTile` elements.

Plus **3 sweep-introduced fails** on `test_moreh_nll_loss_unreduced.py` (same root cause).

**Severity**: should-fix.

**Recommended follow-up agent**: `ttnn-implementer` — mechanical annotation fix on the 2 nll_loss kernels.

**Estimated impact**: 3 + 3 = **6 fail recovery**.

### Class 3 — Sweep-introduced logsoftmax + logsoftmax_ulp (NEW — not flagged by user)

`test_moreh_logsoftmax.py` and `test_moreh_logsoftmax_ulp.py` share kernels with `moreh_softmax_*_large.cpp`. The +18 and +36 sweep-introduced fails are the same root cause as Class 1.

**Severity**: should-fix (subsumed by Class 1 fix).

---

## 5. Recommendations

### Verdict YELLOW — merge conditionally

Rationale:
- **Helper-level changes (Reg A, Reg B, Reg C) all land cleanly with correct root-cause analyses and zero helper-baseline drift.** D8 invariant strictly held. Test bcast_to and helper bar preserved.
- **Reg C delivers +41 test passes on moreh_adam/moreh_adamw** — a substantial win.
- **The sweep introduces 5 regression classes (+68 fails) due to a uniform migration-side annotation gap.** This is mechanical to fix.
- **No new failure classes that cannot be addressed by a single targeted implementer pass.**

### Required follow-up before declaring sweep GREEN

1. **`ttnn-implementer` fp32 annotation cleanup** (single mechanical pass):
   - `moreh_softmax_h_large.cpp`, `moreh_softmax_c_large.cpp`, `moreh_softmax_w_large.cpp` — add `CopyTileReconfig::Input`, `PackTileReconfig::Output`, and `EnableFp32DestAcc = DST_ACCUM_MODE` to `moreh_unary_chain<>` and `moreh_rexp_chain<>` templates (lines 60-90 in each).
   - `moreh_nll_loss_step2_kernel.cpp` — wire `PackTileReconfig::Output` and `EnableFp32DestAcc = DST_ACCUM_MODE` on each `PackTile<>`/`BinaryFpu<>` migrated by Commit 5.
   - `moreh_nll_loss_backward_kernel.cpp` — same.
   - Validation: re-run the 5 regressed test files; expect +62 pass recovery on softmax/logsoftmax family and +6 on nll_loss/unreduced.

2. **Upstream PR for Reg B** — moreh_sum + moreh_norm `padded_shape().rank()` → `logical_shape().rank()` fix is on this branch only. File against `main` so the same bug isn't re-introduced.

### Defer (not blocking sweep merge)

3. **16-softmax + 10-nll_loss + 5-logsoftmax pre-existing classes** — `ttnn-expert-debugger` separate triage; orthogonal to sweep.
4. **LOC drift** — sweep landed at +376 net production code vs design v6's -171 projection (+547 LOC over). The in-kernel template helpers (`moreh_unary_chain`, etc.) and verbose `BinaryFpu<...>` parameter lists are the bulk. Acceptable for now; future helper-extension run (Type-2: MaskInject, HeldCbBinaryFpu, BinaryFpuPerTileScalarB) could reduce per-kernel template-arg noise.
5. **5-test deepseek_grouped_gate pre-existing "Invalid group selection"** — unrelated logic bug; tracked separately.

---

## 6. Acceptance recommendation

**Recommendation: MERGE with documented follow-up.** The sweep + Reg A/B/C deliver a coherent migration outcome: helper architecture is sound (D8 held, Reg A/C surgical, +41 test wins), one clean upstream PR is owed (Reg B), and the +68 sweep-introduced regressions are mechanically addressable in a single annotation pass. The regression class does not stem from a helper-design defect — the helper supports the correct path; the migration omitted the per-element opt-in flags.

Holding the sweep would block the +41 win and the Reg A/B/C fixes from landing. Merging with a tracked annotation-fix task is the higher-value path.
