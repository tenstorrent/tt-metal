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
