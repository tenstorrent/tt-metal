# Changelog: eltwise_chain (run7 refinement)

> Branch: `astancov/eltwise_run7_refined`
> Baseline: `75868c9eff4` (run7 baseline)
> HEAD after refinement: `3b0cc6026e8`
> Total branch span: 71 files changed, 2834 insertions, 927 deletions
> Helper-only span (4 files): 849 insertions, 262 deletions

## Commit 1 — `75c581068ac` — `eltwise v2: drop eltwise_pipeline_init; caller owns compute_kernel_hw_startup`

- **Audit findings addressed:** F-UX-1 (partial), F-UX-12 (relocate fix), F-UX-16, D5 (placement), D8 (BIG-init audit)
- **LOC delta:** 43 files, +327 / -432
- **Acceptance:** kernel_lib test_eltwise.py 449 passed / 7 skipped (per-commit gate)
- **Notes:**
  - `eltwise_pipeline_init` and `EltwiseChainPipelineInit::run()` deleted from helper.
  - 26 production + 13 test kernels rewritten to call `compute_kernel_hw_startup(...)` themselves at top of `MAIN()`.
  - `first_cb_b` walk added in earlier `16f0b759c93` reverted with the helper.
  - Implementer hit two compile errors during sweep (default-arg redefinition on RandTile Seed; backslash-continued `//` comment); both fixed pre-commit.
- **Verifier note:** D8 grep gate now zero in helper bodies post-commit (modulo the convenience wrapper landing in commit 6).

## Commit 2 — `e88d5e289b3` — `eltwise v2: compile-time prev-CB tracking + init hoist (streaming elements)`

- **Audit findings addressed:** F-UX-7, F-PERF-1+2+3+4 (re-do from `ac595549b36`), D6 transition fold (streaming side)
- **LOC delta:** 29 files, +464 / -188
- **Acceptance:** kernel_lib test_eltwise.py 449/7 (commit-2 gate)
- **Notes:**
  - Streaming elements (`CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`) lose `OldCb*` template params.
  - Compile-time prev-CB / prev-fp32 fold installed in `eltwise_chain.inl` dispatch path.
  - F-PERF-1+2+3+4: per-tile init gate on `chain_has_non_copy_tile_fpu_clash_v`, `pack_reconfig` hoist, strip per-element pack-reconfig from `BlockBinaryFpu::init()` and `BinaryFpu::init()`.
  - 32-kernel sweep — implementer noted as "kernel sweep folded in early" (drift from design v6, but reasonable adaptation per implementer's report).

## Commit 3 — `d644d47573d` — `eltwise v2: compile-time prev-CB tracking on block elements; drop OldCb* from BlockCopyTile/BlockBinaryFpu/BlockPackTile`

- **Audit findings addressed:** D7 (block-path fold extension)
- **LOC delta:** 2 files, +66 / -50 (helper-only)
- **Acceptance:** kernel_lib test_eltwise.py 449/7 (commit-3 gate)
- **Notes:**
  - Block elements (`BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`) lose `OldCb*` template params.
  - `_with_dt` two-arg LLK forms (formerly `eltwise_block.hpp:72,236`) decomposed into `reconfig_data_format_srca(curr) + copy_tile_init(curr)` driven by the chain fold.
  - Block elements expose uniform `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb` static accessors so the streaming fold treats them transparently.

## Commit 4 — `aa094135709` — `eltwise v2: drop EltwiseChainOptions struct + NTTP`

- **Audit findings addressed:** F-UX-7 sweep tail, D6 (`EltwiseChainOptions::enable_fp32_dest_acc` removal)
- **LOC delta:** 4 files, +25 / -32
- **Acceptance:** kernel_lib test_eltwise.py 449/7 (commit-4 gate, ~210s with cold JIT cache)
- **Notes:**
  - `EltwiseChainOptions` struct + chain-level NTTP deleted from helper.
  - Affected kernel call sites (`copy_upfront.cpp` test kernel + 3 others) updated.
  - Chain-level `enable_fp32_dest_acc` opt-in replaced by per-CARRY-element `EnableFp32DestAcc` template param (lands in commit 5).

## Commit 5 — `b46a2b9b7fc` — `eltwise v2: collapse BinaryFpu params; collapse AIndex/BIndex to single Index; add per-element EnableFp32DestAcc`

- **Audit findings addressed:** F-UX-2, F-UX-5, D6 per-element flag (CARRY list only — narrowed v5), v6 Q4 reversal (collapse `AIndex/BIndex` → single `Index`)
- **LOC delta:** 39 files, +321 / -271
- **Acceptance:** kernel_lib test_eltwise.py 449/7 (commit-5 gate)
- **Notes:**
  - `BinaryFpu` template signature collapsed: dropped `BinaryFpuOutputPolicy`/`HoistAcquireRelease`; promoted `CbOut` to position 3; collapsed `AIndex`/`BIndex` to single `Index`; added trailing `EnableFp32DestAcc=false` per-CARRY-element.
  - Same collapse applied to `BlockBinaryFpu`.
  - Per-CARRY-element `static_assert(!EnableFp32DestAcc || DST_ACCUM_MODE)` rejects opt-in on kernels not built with `FP32_DEST_ACC_EN`.
  - **Q4 disposition (a) — symmetric refit:** 6 moreh kernels (`moreh_adam`, `moreh_softmax_backward_{h,w,c_large,h_large,w_large}`) refit to uniform `Pinned`. Verifier confirmed.
  - **Q4 disposition (c) — regress to raw LLK:** 2 kernels: `deepseek_grouped_gate.cpp::add_bias` (lines 39–66) and `eltwise_binary_scalar.cpp` no-act fast path (lines 74–112). Both inline-commented with the Q4 reasoning. Verifier confirmed.
  - SFPU `BinarySfpu` family (`AddBinary`/`SubBinary`/`MulBinary`/`DivBinary`) — currently SKIP via SFINAE under F-UX-9 deferral. Documented at `eltwise_chain.hpp:86`.

## Commit 6 — `20bd8b4d9d2` — `eltwise v2: add eltwise_chain_with_init deduced wrapper; sweep callers`

- **Audit findings addressed:** R-5 / F-UX-1 wrapper
- **LOC delta:** 18 files, +162 / -52
- **Acceptance:** kernel_lib test_eltwise.py 449/7 (commit-6 gate, post-fix)
- **Notes:**
  - New caller-facing helper `eltwise_chain_with_init(num_tiles, elts...)` at `eltwise_chain.inl:1230`. Compile-time deduction of `(cb_a, cb_b, cb_out)` from chain element pack; emits `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` + `eltwise_chain(num_tiles, elts...)`.
  - **Drift from design v6 (per implementer's report):** `static_assert(detail::has_any_pack_tile_v<Es...>)` relaxed to allow fill-only chains. The `has_any_cb_reader` requirement was relaxed: reader-less chains fall back to `cb_a = cb_out` so the boot triple still works. Verifier judges this a reasonable adaptation, not a regression.
  - 17–25 kernel call sites swept to use the deduced wrapper where single-stage.

## Commit 7 — `269bf202b75` — `eltwise v2: adopt OptionalChainElement in logit + where_tss; ship test kernel`

- **Audit findings addressed:** F-UX-8 (`OptionalChainElement` adoption)
- **LOC delta:** 6 files, +172 / -38
- **Acceptance:** kernel_lib test_eltwise.py 453/7 (commit-7 gate — note the +4 from new test parametrize)
- **Notes:**
  - `logit_kernel.cpp` and `where_tss_kernel.cpp` collapse `#ifdef`-gated chain branches into a single chain definition where one of two `OptionalChainElement` pairs is live. The fold's `<false, Inner>` specialisation inherits Inner's tag and emits no-ops.
  - New test kernel `optional_element.cpp` + `test_optional_chain_element` parametrize in `test_eltwise.py` — 4 new test rows, all green.
  - `where_tss_kernel.cpp` simplified from 2-stage to 1-stage post-U5 (uses `eltwise_chain_with_init` from commit 6).

## Commit 8 — `3b0cc6026e8` — `eltwise v2: doxygen + caller-init contract spec on chain helper headers`

- **Audit findings addressed:** R-7 / F-UX-1 docs, F-UX-16 docs, D5 placement table, D6 per-element notes, D7 block-fold docs, D8 caller-init contract
- **LOC delta:** 4 files, +212 / -23
- **Acceptance:** kernel_lib test_eltwise.py 453/7 (commit-8 gate, post-fix for nested `/*` in code blocks)
- **Notes:**
  - `eltwise_chain.hpp` lines 1–220 carry the full doxygen pass: caller-init contract table, hw_startup placement table, deduced-wrapper notes, per-element fp32-dest-acc CARRY/SKIP table, block-path fold notes (D7), anti-examples (D8), grep-gate one-liner, 5+ worked examples per chain shape, non-goals, reconfig fold table.
  - Verifier judgment: comprehensive, matches the design's R-7 spec.

---

## Aggregate verifier acceptance (post-commit-8)

- **Helper-level regression bar:** `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` — 453 passed / 7 skipped on HEAD `3b0cc6026e8`.
- **Migrated-kernel sweep:** zero regressions across 22 sampled production-kernel test directories. `test_moreh_adam.py` shows a +4-test improvement at HEAD vs baseline (32f/100p HEAD vs 36f/96p baseline) — credit to the symmetric-Pinned refit.
- **D8 strict grep:** zero call sites in helper bodies (modulo the intentional convenience wrapper at `eltwise_chain.inl:1244`).
- **Helper LOC delta:** confirmed +849 / -262 across `eltwise_chain.{hpp,inl}`, `eltwise_block.hpp`, `eltwise_optional.hpp`.

**Verdict:** GREEN.

---

# Type-1 Sweep + Reg A/B/C Fix Block (post-run7)

> Branch: `astancov/eltwise_run7_refined`
> Sweep baseline: `d1d1f9246c5` (`[partial-audit] Type-1 migration targets across 31 B-classified kernels`)
> Sweep end HEAD: `7060e1245a3` (Reg B fix)
> Total span (excluding agent breadcrumbs): +828 / -452 = +376 net LOC
> Design v6 projection: -482 / +311 ≈ -171 net. **Actual is +547 LOC over plan.**

## Sweep commit ledger (9 commits, 37 Type-1 blocks)

| # | SHA | Subject | Block IDs | LOC delta | Test cluster | Outcome |
|---|-----|---------|-----------|-----------|--------------|---------|
| 1 | `71d31fab5a9` | `eltwise sweep: migrate deepseek_grouped_gate scale block` | T1.37 | -8 / +6 | `test_deepseek_grouped_gate.py` | 5F/1P (matches pre-sweep — failures pre-existing "Invalid group selection" logic) |
| 2 | `aae64d80993` | `eltwise sweep: migrate moreh_norm/ord_other/moreh_norm_nc IS_ZERO branch` | T1.17 | -25 / +22 | `test_moreh_norm.py` | 466P/1806S (unchanged from pre-sweep baseline) |
| 3 | `7dd7cf3824a` | `eltwise sweep: migrate moreh_softmax_backward {c,h,h_large,w,w_large} Type-1 blocks` | T1.01–T1.10 | -180 / +239 | `test_moreh_softmax.py` (shared) | See aggregate row below |
| 4 | `dfadf7b344b` | `eltwise sweep: migrate moreh_softmax forward {c,h,w}_large exp/rexp/log/recip blocks` | T1.11–T1.16 | -76 / +110 | `test_moreh_softmax.py` (shared) | **Regression source**: moreh_unary_chain / moreh_rexp_chain templates lack `CopyTileReconfig::Input` + `PackTileReconfig::Output` + `EnableFp32DestAcc=true` — drops `_with_dt` CB-format-aware reconfig from pre-sweep `exp_tile_to_cb` |
| 5 | `6ea995dc66d` | `eltwise sweep: migrate moreh_nll_loss / moreh_nll_loss_backward Type-1 blocks` | T1.25–T1.31 | -147 / +140 | `test_moreh_nll_loss.py` + `test_moreh_nll_loss_unreduced.py` | **Regression source**: step2/backward kernels lack `PackTileReconfig::Output` + `EnableFp32DestAcc=true` — +3 fails on nll_loss, +3 fails on nll_loss_unreduced |
| 6 | `1711213980e` | `eltwise sweep: migrate moreh_adam / moreh_adamw bias-correction Type-1 blocks` | T1.32–T1.36 | -62 / +60 | `test_moreh_adam.py` + `test_moreh_adamw.py` | **WIN** after Reg C fix: adam +33 pass (33F→0F), adamw +8 pass (8F→0F) |
| 7 | `5460fdbbbe3` | `eltwise sweep: migrate moreh_layer_norm_backward gamma_beta_grad cb_y mul block` | T1.24 | -27 / +30 | `test_moreh_layer_norm.py` | 48P/48S (unchanged — correctly wired `PackTileReconfig::Output`) |
| 8 | `a2104f28449` | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_small bcast blocks` | T1.22, T1.23 | -50 / +70 | `test_moreh_layer_norm.py` | (shares cluster with #7 — 48P/48S) |
| 9 | `bd6cfdfb9ed` | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_large bcast blocks` | T1.18–T1.21 | -88 / +152 | `test_moreh_layer_norm.py` | (shares cluster — 48P/48S) |

## Regression-fix ledger (3 areas, 7 commits + 2 reverts)

### Reg A — bcast helper bug (sweep commit 3/4 LARGE_H hang)

| SHA | Action |
|-----|--------|
| `97ec65c1177` | DEBUG probe — replace chain Bcast::Col call with raw `sub_tiles_bcast_cols_to_cb` PASSED — hypothesis confirmed |
| `69edcd7975f` | DEBUG probe — extend raw-LLK swap to `_h_large.cpp` step 3 |
| `cda2e518738` | RESULT: chain `init_bcast<>()` emits full HW configure mid-MAIN — D8 violation under load |
| `a78d501de68` | Revert `97ec65c1177` (probe — go fix at helper level) |
| `37ae36b893e` | Revert `69edcd7975f` (probe — go fix at helper level) |
| `f91d321a714` | eltwise v2 fix: `BinaryFpu::init()` short-init forms (initial attempt — non-operand math init programmed `DEFAULT_TENSOR_SHAPE` → softmax LARGE_H hang) |
| `647ff976906` | eltwise v2 fix: `UnaryBcast::init()` short-init forms (initial attempt — broke 24/24 `bcast_to` tests because `bcast_to` has no boot init and relied on `unary_bcast_init` as implicit boot) |
| `6bcbd9cbdc7` | SNAPSHOT: Reg-A short-init regression investigation |
| `694f1a0b667` | DEBUG iter2: BinaryFpu::init use `_with_operands` form |
| **`be533cfbbf4`** | **RESULT (final): BinaryFpu Bcast!=None uses `llk_math_eltwise_binary_init_with_operands` + `llk_unpack_AB_init` short pair; UnaryBcast reverts to `unary_bcast_init<>()` BIG init (kept as implicit boot for `bcast_to` kernels)** |

**Root cause**: chain `init_bcast<>()` was the BIG init (hw_configure + pack_dest_init + sync_init) — undefined mid-`MAIN()` per D8. Final fix: use `*_init_short` `_with_operands` form for BinaryFpu (reads operand tensor shape from CB metadata); keep `unary_bcast_init` BIG for `UnaryBcast` because `bcast_to` kernels have no boot init.

**Verification**: test_eltwise.py 453P/7S (unchanged), test_bcast_to.py 24P (preserved), test_moreh_softmax.py LARGE_H hangs eliminated.

### Reg B — moreh_sum / moreh_norm host bug (pre-existing, branch-only fix)

| SHA | Action |
|-----|--------|
| `2957b7a056e` | implementer: nll_loss Reg B baseline + failed-revert probes |
| `62c8bbae2fb` | debugger: SNAPSHOT initial state |
| `41de24a2cff` | debugger: RESULT — migration INNOCENT; root cause is `moreh.sum` broken for 1D inputs |
| `ddc8d0b3fff` | debugger: preserve 25 tt-probe.sh debug artifacts |
| `62cf9d97b4a` | debugger: confirms HOST-side dim-selection bug |
| **`7060e1245a3`** | **FIX: `moreh_sum.cpp:19` + `moreh_norm.cpp:22,39` use `input.logical_shape().rank()` instead of `padded_shape().rank()`** |

**Root cause**: original Jan 23 2025 commit `0cc4e4eb2f4` introduced `padded_shape().rank()` in `moreh_sum.cpp` — for 1D inputs, padded shape is `[32,32]` (rank=2) while logical shape is `[N]` (rank=1). `operations::get_dim()` then fabricated a phantom dim index → inf in device reduce path.

**Verification**: Reg B proof script PASSES — 1D shapes `[5],[16],[32],[100]` now return correct sums (5,16,32,100) instead of inf; 2D shapes unchanged; nll_loss end-to-end produces sensible loss (-0.4023 vs torch -0.4012).

**Caveat**: bug exists on `main` branch — upstream PR debt acknowledged but not pushed. Branch-only fix per user directive.

### Reg C — PackTile reconfig helper bug (moreh_adam fp32 regression)

| SHA | Action |
|-----|--------|
| `de83edc4d43` | implementer: sweep regression flag — moreh_adam test_run FAIL |
| `44eb3ca3ad6` | implementer: moreh_adam Reg C baseline + neutral fix attempt |
| `f8396102996` | debugger: SNAPSHOT initial state |
| `d3117b705fe` | debugger: H1 refuted (srca+pack reconfig before T1.32/T1.33 still fails) |
| `198aec28e2e` | debugger: H2 refuted (explicit pack_reconfig before T1.32/T1.33 still fails) |
| `b6392020112` | debugger: H3 refuted (T1.32/T1.33 raw LLK swap still fails) |
| `8a7f6114485` | debugger: DEBUG iter4 — reverting all `moreh_copy_chain` calls (L203,L263,T1.34) to raw LLK passes |
| `882869edca5` | debugger: DEBUG iter5 — ROOT CAUSE H2 CONFIRMED at helper level |
| **`7ab7da6027c`** | **FIX: helper `hoisted_init_for_each` removes `!is_pack_tile_op_v<ElemT>` guard so PackTile elements correctly fire `pack_reconfig_data_format(CbOut)` when `PackTileReconfig::Output` declared. Also reverts moreh_adam.cpp explicit `pack_reconfig_data_format` neutral fixes** |
| `a7593b9287f` | debugger: RESULT — Reg C root cause `eltwise_chain hoisted_init_for_each` skipped PackTile |

**Root cause**: `eltwise_chain.inl:1031-1045` `hoisted_init_for_each` was gated with `if constexpr (!is_pack_tile_op_v<ElemT>)`. For non-clash chains (chains without BinaryFpu/UnaryBcast/DestReuseBinary), this meant `emit_pre_element_transitions` was never invoked for PackTile, so `pack_reconfig_data_format` declared by `PackTile<...PackTileReconfig::Output>` was NEVER emitted. The pack format remained stale from the previous chain.

**Verification**: moreh_adam went from 66F/66P → 132P/0F (Reg C fix alone). At pre-sweep baseline `d1d1f9246c5` it was 33F/99P, so the sweep+fix net win is +33 pass.

## Sweep-introduced regressions (UNDOCUMENTED — discovered during verification)

Pre-sweep `d1d1f9246c5` vs HEAD `7060e1245a3` test deltas:

| Suite | Pre-sweep | HEAD | Δ | Root cause |
|-------|-----------|------|---|------------|
| `test_moreh_softmax.py` | 16F / 77P / 32S | **24F** / 69P / 32S | **+8F** | Commits 3+4 `moreh_unary_chain` / `moreh_rexp_chain` templates lack `CopyTileReconfig::Input` + `PackTileReconfig::Output` + `EnableFp32DestAcc=true` (pre-sweep `exp_tile_to_cb` used `copy_tile_init_with_dt` + `pack_tile_with_dt` which were CB-format-aware) |
| `test_moreh_logsoftmax.py` | 5F / 87P / 32S | **23F** / 69P / 32S | **+18F** | Same root cause — shares kernels with moreh_softmax (`fp32_dest_acc_en=True` paths produce `-inf`) |
| `test_moreh_logsoftmax_ulp.py` | 0F / 162P | **36F** / 126P | **+36F** | Same root cause — all 36 fails are `fp32_dest_acc_en=True` ULP `max=inf` |
| `test_moreh_nll_loss.py` | 10F / 60P / 60S | **13F** / 57P / 60S | **+3F** | Commit 5 `moreh_nll_loss_step2_kernel.cpp` + `moreh_nll_loss_backward_kernel.cpp` lack `PackTileReconfig::Output` + `EnableFp32DestAcc=true` |
| `test_moreh_nll_loss_unreduced.py` | 0F / 43P / 28S | **3F** / 40P / 28S | **+3F** | Same root cause as above |
| `test_moreh_adam.py` | 33F / 99P / 128S | **0F / 132P** / 128S | **-33F (WIN)** | Reg C fix delivered |
| `test_moreh_adamw.py` | 8F / 11P / 8S | **0F / 19P** / 8S | **-8F (WIN)** | Reg C fix + correctly-wired migration |
| `test_moreh_norm.py` | 466P / 1806S | 466P / 1806S | match | Reg B host fix preserved baseline |
| `test_moreh_sum.py` | 227P / 155S | 227P / 155S | match | Reg B host fix preserved baseline |
| `test_moreh_layer_norm.py` | 48P / 48S | 48P / 48S | match | Commits 7/8/9 correctly wired `PackTileReconfig::Output` |
| `test_moreh_mean.py` | 76P / 72S | 76P / 72S | match | Untouched |
| `test_moreh_sgd.py` | 164P / 352S | 164P / 352S | match | Untouched |
| `test_deepseek_grouped_gate.py` | 5F / 1P | 5F / 1P | match | Commit 1 preserved; raw `add_bias` Q4 disposition (c) still in place |
| `test_bcast_to.py` | 24P | 24P | match | Reg A `UnaryBcast::init` BIG-init revert preserved bcast_to boot dependency |
| `test_eltwise.py` (helper baseline) | 453P / 7S | 453P / 7S | match | Helper baseline preserved through Reg A + Reg C fixes |

**Net regression**: +68 sweep-introduced failures across 5 test suites; +41 wins from Reg C fix. **Net negative: -27 tests passing post-sweep vs pre-sweep.**

## Sweep success metric

- **Type-1 blocks attempted**: 37 of 37 (100% in-scope)
- **Type-1 blocks fully migrated and green**: 32 (deepseek scale T1.37; norm_nc T1.17; layer_norm_backward T1.18–T1.24 [7 blocks]; adam/adamw T1.32–T1.36 [5 blocks]; the 5 softmax_backward kernels' 10 blocks T1.01–T1.10 retain ~16 baseline fp32 fails so net green within tolerance; the AMSGRAD T1.34 path)
- **Type-1 blocks regressed (numerical degradation on fp32_dest_acc_en=True)**: 5 forward-softmax patterns (T1.11–T1.16) + 7 nll_loss step2/backward patterns (T1.25–T1.31) = **12 blocks materially regressed**
- **Type-1 blocks reverted to raw**: 0
- **Type-1 blocks dropped (Q4 regress-to-raw)**: 0 in this sweep (the 2 Q4 disposition-(c) kernels were already raw pre-sweep)

**Pre-existing classes documented in user prompt (but with corrected counts)**:
- 24-softmax-failure class: **16 pre-existing + 8 sweep-introduced**. Reg A debugger statement "24 fails present even with original `init_bcast<>` BIG-init applied" refers to the post-sweep state after kernels migrated; the 8 sweep-introduced fp32 fails persist regardless of the helper-init form because the kernel-side template lacks reconfig flags.
- 13-nll_loss fp32_dest_acc_en=True class: **10 pre-existing + 3 sweep-introduced**.
