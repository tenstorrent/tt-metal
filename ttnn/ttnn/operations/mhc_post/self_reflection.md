# mhc_post — self-reflection (post-blind, advisory)

## Summary
- **Blind:** 482 passed / 1 failed / 348 skipped (INVALID) of 831. The blind set has no translated cells: it is golden `test_op` + `test_op_loose` (456 pass) plus `test_regression.py` (26/27). `verifier_report.json`: `supported_pass` 468, `supported_fail` 0, `xpass_drift` 0. SUPPORTED equals TARGET on every axis, and EXCLUSIONS is empty.
- **The one failure is a golden-suite defect, not an op bug.** `test_hc_scalars_are_runtime` sets `hc_eps` above the fixed `HC_COLSUM` gate, and the float64 reference fails that gate too. It was diagnosed in Refinement 2 and then failed unchanged in all 8 later measurements (R2–R5, Perf 1–2, blind). That points at a framework gap: there is no channel for "golden defect", so a known defect stays counted as a failure.
- The perf phase ran 2 rounds: focus collapse T160 C5120 went from 107.5 to ~66–70 µs. All 7 helper-bypass rows were checked against the helper sources. One is probably `undocumented` (the DEST-transform row). One overstates the gap (the runtime-count reduction row). The rest are real `missing` capabilities, and they cluster on two helpers.
- Everything else looks op-level clean. The framework findings are a misleading `unary_bcast` comment, a missing FPU-accumulation rule in the numerics reference, a wrong precision model in golden `PROJ_REL`, and process gaps around golden defects and stale kernel-head comments.

## 1. Golden coverage (`eval/golden_tests/mhc_post/`)

**1.1 `HC_COLSUM` gate is tighter than its own reference (the only blind failure)**
- **What:** The last Sinkhorn step is a column normalisation `m / (colsum + hc_eps)`, so the comb column sums are 1 − O(hc_eps) by construction. `test_hc_scalars_are_runtime` uses `hc_eps = 1e-6·seed` = 1.1e-5 / 1.2e-5. The fixed gate is 1e-5, so the float64 `deepseek_hc` reference itself fails it.
- **Evidence:**
  - `test_regression.py:211`: `case.hc["hc_eps"] = 1e-6 * seed`
  - `helpers.py:330`: `HC_COLSUM = 1e-5`
  - `helpers.py:113`: `m = m / (m.sum(dim=-2, keepdim=True) + hc_eps)`
  - Blind failure: `hc_comb_column_sums: max |sum - 1| 1.11e-05 > 1e-05`. The changelog (`changelog.md:191-192`) shows the reference at 1.10e-5 and the device at 1.116e-5: they differ by about 1.6e-7.
- **Recommendation:** Propose gating the device colsum against the reference colsum instead of against 1: `|colsum_dev − colsum_ref| ≤ 1e-6`. Minimal alternative: `HC_COLSUM + hc["hc_eps"]` in `check_collapse` (`helpers.py:420-422`).
- **Confidence:** high. The arithmetic is checkable from `helpers.py:112-116` alone.

**1.2 `PROJ_REL[HiFi3/HiFi4] = 1e-5` rests on a precision model the run disproved**
- **What:** The golden comment says that at HiFi3/4 "every product is exact and only the fp32 accumulation remains". The implementer's device probe found that the FPU rounds each 16-term face partial sum to about tf32 at every fidelity. Meeting 1e-5 then took a 16-pass masked projection (`proj_passes = 16`) at HiFi3/4. That is a large perf cost at HiFi4: collapse T160 is 151 µs versus about 66 µs at HiFi2 (`changelog.md` Perf 2 guard set).
- **Evidence:**
  - `helpers.py:299-303`: "at HiFi3 / HiFi4 … every product is exact and only the fp32 accumulation remains … NOTE: … not yet calibrated on device".
  - Commit `d870fff7104`: "FPU dot product rounds the per-face (16-term) partial sum to ~tf32 at every fidelity", HiFi4 rel 3e-4 > 1e-5.
- **Recommendation:** Propose that a human decide whether the contract really requires an exact projection at HiFi3/4. If it does, fix the comment to state the face-rounding fact. If it does not, calibrate `PROJ_REL[HiFi3/4]` to the measured single-pass value (~3e-4) so the op can drop the 16-pass path.
- **Confidence:** med. This is a contract decision, not a bug.

**1.3 The deployed fidelity (HiFi2) selects code paths that no axis captures (axis-blind; preventive)**
- **What:** `math_fidelity` is not a TARGET axis. Golden `test_op` runs stats and collapse at the default HiFi4 (`helpers.py:478`). Fidelity nevertheless selects two code paths:
  - `proj_passes`: 1 vs 16 (`mhc_post_program_descriptor.py:481`);
  - the Perf 2 carve-out `not (collapse and proj_passes == 1)` (`mhc_post_program_descriptor.py:121-133`).

  The model's own HiFi2 path is exercised only at C=5120 with tile-aligned T in `LOOSE_CASES`, plus regression shapes T160/T150 C5120 and T64 C640. No failure was observed, so this is preventive. The carve-out's single-tile path (`mix_tile`) at HiFi2 has never run at an odd per-core c-extent or at multiple rows per core.
- **Evidence:** `feature_spec.py` `_perf_case` sets `math_fidelity` only for the DS4.1 sweep (T ∈ {160, 320, 640, 1280}, C = 5120).
- **Recommendation:** Propose two `LOOSE_CASES` entries with `extras={"math_fidelity": HiFi2}`, both bf16/bf16:
  - `_case((1, 1, 333, 4*6144))`, collapse, separate;
  - `_case((1, 1, 1000, 4*1792))`, stats, packed.

  Consider promoting `math_fidelity ∈ {HiFi2, HiFi4}` (fused modes only) to a TARGET axis.
- **Confidence:** med on the gap, low on its urgency.

## 2. SUPPORTED honesty
No evidence-backed finding. `supported_fail` = 0, `xpass_drift` = 0, `invalid_unexpected` = 0. SUPPORTED equals TARGET, and the one failure is a test defect (§1.1), not an over-claim. The 15 `no_axes_found` cells are regression tests without axes, which is expected.

## 3. Helper / reference docs

**3.1 `unary_bcast` says fp32 uses unpack-to-dest, but only when the CB is tagged (misleading)**
- **What:** The comment says 32-bit formats use unpack-to-dest. The code actually keys on `get_operand_dst_format(icb)`, which is TF32 for an untagged Float32 CB, so an fp32 Col broadcast silently truncates (about 8e-4 relative error). The planner copied the claim into the design: "`UnaryBcast` auto-selects unpack-to-dest for Float32".
- **Evidence:**
  - `tt_metal/hw/inc/api/compute/bcast.h:46-50`: "32bit formats are implemented using unpack to dest".
  - `op_design.md:241,397`.
  - Implementer friction breadcrumb (cost med) and commit `8f4b3122cee`.
- **Recommendation:** Propose a one-line addition at `bcast.h:46` and on `ckl::UnaryBcast` (`kernel_lib/eltwise/broadcast/bcast.hpp`): "exact fp32 only if the source CB is tagged `UnpackToDestFp32`; untagged Float32 → TF32." Also add an fp32-broadcast line to `numerical_stability_analysis_reference.md` §2.7.
- **Confidence:** high.

**3.2 Numerics reference has no rule for FPU dot-product accumulation (absence)**
- **What:** `numerical_stability_analysis_reference.md` covers the TF32 truncation of srcA/srcB operands. It never states that the FPU's in-face accumulation (16 terms) is rounded to about tf32 before it reaches fp32 DEST, at every fidelity, so HiFi4 does not give an exact fp32 dot product. The run lost one implement–fail–probe cycle to this, and the golden author had the same wrong model (§1.2).
- **Evidence:** Commit `d870fff7104`; implementer `device_print_observation`: "FPU matmul rounds each 16-term face partial to ~tf32 at all fidelities … DEST accumulation across K tiles exact"; `numerical_stability_analysis_reference.md:40-51` covers operand truncation only.
- **Recommendation:** Propose a §2.1 bullet: "matmul/reduce FPU: each 16-term face partial is rounded to ~tf32 regardless of fidelity; only the cross-K-tile DEST accumulation is fp32. An exact fp32 dot product needs one product per face (masked passes) or SFPU."
- **Confidence:** med (one probe, but consistent with the measured 3e-4).

**3.3 `verify_supported` CLI needs `python_env` (minor)**
- See §4.3.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| eltwise_chain: DEST transform around an SFPU body (`hc_row`, Perf 1) | capability | **undocumented** (likely) | The `UnaryOp` extension point takes arbitrary DEST-only `init()`/`exec_impl()` (`core/chain.inl:620-649`), and this op already wraps `HcProgram` and `SfpuRowSum` as elements (`mhc_post_compute.cpp:170`). The raw sequence at `mhc_post_compute.cpp:524-537` is exactly CopyTile → [transpose_dest, program, transpose_dest] → PackTile. The 107.6 → 96.9 µs came from moving the permute off the writer RISC, **not** from bypassing the chain, so this was effectively an ergonomics bypass with ~0 ns attributable to the API. | Document in `chain.inl:620` that a custom `UnaryOp` may run any DEST-in-place math sequence (e.g. `transpose_dest`) as well as SFPU calls. confidence: low–med (`transpose_dest_init` interplay with the chain's hoisting is unverified; a one-element one-tile chain re-inits anyway). |
| eltwise_chain: init hoisting for non-uniform chains (`mix_block_fused`, Perf 1) | capability | **missing** | Hoisting is all-or-nothing per category and requires uniform MOP + SFPU init (`api/chain.hpp:175-188`). There is no "re-init on operand-class switch". Measured gain about 9.6 µs (≈107.5k → 97.9k). | Add a per-element "init class" so the chain re-emits an init only when the class changes (sketch A). |
| eltwise_chain: two live accumulators + shared coefficient slot (`mix_pair`, Perf 2) | capability | **missing** | The element→slot map is fixed per iteration (`UnaryOp::exec(i, slot_offset)`, `chain.inl:648`). No way to keep D0/D1 live across a term sequence while D3 cycles A_j → B_j → ss. 31.0 → 25.4 µs `compute_mix`. Same helper and same site as the row above, so treat both as one finding: the fused mix outgrew the chain's per-tile model. | Consider a "window" iteration shape (k outputs per acquire, shared operand slots); low priority until a second op needs it. confidence: low. |
| eltwise_chain: runtime-count reduction over a gather CB (`dest_sum`, Perf 1) | capability | **missing (narrower than claimed)** | The row says "a chain acquires DEST per iteration". `DestAccumulation::WholeShape` keeps DEST acquired across a runtime `IterationShape::tiles(n)` (`api/chain.hpp:303-311`), but only for FPU binaries (`api/chain.hpp:535`), which truncate fp32 to tf32. So the real gap is **SFPU (exact fp32) DEST accumulation over a runtime count with per-slot waits**. Gain 107.5 → 105.7 µs (1.7 %). | Extend `DestAccumulation::WholeShape` to SFPU binary elements (`AddBinary`), and document that it is FPU-only today. |
| SFPU multi-term MAD / binary add via SFPLOADMACRO (`mhc_mix_lm.hpp`, Perf 1; `add_binary_tile` in `dest_sum`, Perf 2) | capability | **missing** | No SFPLOADMACRO path in `kernel_lib` (grep: none). LLK has load-macro precedents only for unary ops (`ckernel_sfpu_recip.h:100,158`, `_calculate_reciprocal_fast_24b_5c_`). Gains: 96.4 → 87.9 µs (mix) and 1,347 → 942 ns (combine add). The two rows are one cluster. | Propose a kernel_lib load-macro streaming element for `D_out = Σ a_k·D_k` / `D0 += D1` (sketch B). |
| `sfpu_reciprocal_iter<2>` (Sinkhorn, Perf 1) | capability | **missing** | `ckernel_sfpu_recip.h:22-53`: the template has only `max_iter` and `normalized`. The NaN/sign `v_if` guards (`:45`, `:53`) are unconditional. 5.48 → 3.49 µs per hc row. | Add a `positive_finite` template flag that drops the guards. |

No unrecorded bypass found. Every raw compute call in `mhc_post_compute.cpp` (lines 338-379, 396-444, 524-537, 669-694) and the sfpi/LLK programs in `mhc_post_hc_sfpu.hpp` / `mhc_mix_lm.hpp` maps to a row above. The W/coefficient multicast uses `kernel_lib` `McastArgs` (`mhc_post_reader.cpp:77-81`).

```cpp
// Sketch A — confidence: low (from mhc_post_compute.cpp:310-317's raw loop)
// element declares an init class; chain emits init only on class change
using CopyA = ckl::CopyTile<coef_in, ckl::Dst::D3, ckl::InitClass<0>>;  // fp32 unpack-to-dest
using CopyX = ckl::CopyTile<x_in,   ckl::Dst::D2, ckl::InitClass<1>>;   // bf16 srcA
ckl::eltwise_chain<ckl::InitHoist::OnClassSwitch>(shape, CopyA{...}, CopyX{...}, MixPass{...}, PackOut{...});

// Sketch B — confidence: low (from mhc_mix_lm.hpp add_d1_impl / mix passes)
ckl::eltwise_chain(ckl::IterationShape::tiles(count),
    ckl::CopyTile<gather_in, ckl::Dst::D1>{},
    ckl::LoadMacroMad<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{1.0f},   // D0 += 1.0*D1, macro-scheduled
    ckl::PackTile<out_spec, ckl::Dst::D0>{},
    ckl::DestAccumulation::WholeShape);
```

## 4. Agent prompts

**4.1 No structured channel for "suspected golden defect"**
- **What:** The implementer correctly diagnosed the `HC_COLSUM` defect in Refinement 2. It was then re-reported in prose 7 times across the changelog and op_requirements, never reached the golden owner as a structured signal, and still counts as the blind's only failure.
- **Evidence:**
  - Implementer breadcrumb: "reference itself violates HC_COLSUM for hc_eps=1.1e-5".
  - `op_requirements.md:146-148`: "This needs a human edit to `helpers.py`".
  - `changelog.md:188-195, 246, 301, 373, 487, 660`.
- **Recommendation:** Propose a `golden_defect` breadcrumb event (nodeid, gate, evidence that the reference itself fails) in `ttnn-implementer.md` / `incremental-verifier.md`. The dashboard can then surface it and the blind summary can tag that nodeid separately from op failures.
- **Confidence:** high.

**4.2 Perf graduations left the kernel file-head comment stale**
- **What:** The compute kernel's head still says all compute goes through helpers ("Every phase is an eltwise_chain or matmul_block helper call", "Helper gaps (recorded, not bypassed)") and that "the writer permutes the stats row into hc lanes". After Perf 1–2, the mix, hc_row (Blackhole) and dest_sum are raw loops, and on Blackhole the permute runs in compute. The per-site comments were updated; the file header was not.
- **Evidence:** `mhc_post_compute.cpp:4-6, 33-37, 39`, against the raw sites at `:310-337` and `:650-668`.
- **Recommendation:** Propose a line in `perf-coordinator.md`'s graduation checklist: "update each touched kernel's file-head summary so it matches the graduated paths and lists the bypasses."
- **Confidence:** med.

**4.3 Verifier CLI command fails without the env**
- **What:** The verifier prompt's `verify_supported` command fails with `ModuleNotFoundError` unless `python_env` is active, which looks like an op packaging bug.
- **Evidence:** Verifier friction breadcrumb (commit `8c523c72fc5`); `incremental-verifier.md:205`.
- **Recommendation:** Prefix the command with `source python_env/bin/activate &&` at `incremental-verifier.md:205`.
- **Confidence:** high, low cost.
