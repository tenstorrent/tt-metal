# `eltwise_chain` (run7) — Refinement Design (v5)

## Section A — Scope

**Branch:** `astancov/eltwise_run7_refined` — hard-reset to `75868c9eff4` (run7 baseline). Two off-script implementation commits and four prior design commits (v1, v2, v3, v4) sit on top. v5 narrows **D6** scope per user clarification — the per-element `EnableFp32DestAcc` flag is added **only to elements that emit DEST-format-sensitive LLK**, not to every chain element. v4 (995-line design at commit `19fd0e8f3f2`) earlier built on v3 (915-line at blob `6160335fd85^`) by amending it with **D7** (drop `OldCb*` from Block elements; extend the prev-CB / prev-fp32 fold to the block path) and **D8** (helper does not wrap "BIG inits" — generalise D1's caller-owned-`compute_kernel_hw_startup` into a full caller-init contract).

**v5 surgical update (this revision):** D6 narrowed to a CARRY/SKIP element split. Sections updated: A directive table (D6 row), A LOC delta (R-4), B group P1+U1-OldCb-streaming (Step 3b transition fold + SFINAE probe), B group P1-block (block-element fp32 fold), B group U3 (Step 6 — explicit per-element CARRY/SKIP table), B group U6 (doxygen — only CARRY elements expose `@tparam EnableFp32DestAcc`), F Q6 (note on SKIP-element default moot-ness), F Q10 (`static_assert` only fires on CARRY), F Q13 (uniform fold mechanism), F Q16 (NEW — SFINAE detection for SKIP elements). All other sections untouched.

**Already landed (out of this design's scope as fixes — but the *shape* of one of them is being reworked):**
- `16f0b759c93` — F-UX-12 fix-in-place (`first_cb_b` walk inside `EltwiseChainPipelineInit::run()`) and F-UX-11 (`RandTile` static-init via NTTP seed).
- `ac595549b36` — F-PERF-1+2+3+4 (per-tile init gate on `chain_has_non_copy_tile_fpu_clash_v`, `pack_reconfig` hoist, strip pack-reconfig from `BlockBinaryFpu::init()` and `BinaryFpu::init()`).

These commits are not on `astancov/eltwise_run7_refined` HEAD (which is `75868c9eff4`); the design carries them forward as functionality the implementer must reproduce when landing the U2 / P1 commits below. Numbering kept for trace continuity with v2.

**Directive 1 reverses the F-UX-12 fix-in-place direction:** the helper does not own the boot anymore. `eltwise_pipeline_init` is deleted entirely; callers issue `compute_kernel_hw_startup` directly. The `first_cb_b` walk added in `16f0b759c93` is removed with it. F-UX-11's RandTile NTTP-seed fix stays as-is.

**Inputs:** `agents/refinement_audit.md` (984 lines), `eltwise_helper_run7_vs_run8.md`, `eltwise_chain.{hpp,inl}`, `eltwise_block.hpp`, `with_dt_tree.md`, `eltwise_optional.hpp`.

**Scope axes:** UX/API + perf only. Functionality gaps and test gaps remain out of scope.

**Regression bar:** the 401-test validation suite at `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` PLUS every migrated-kernel pytest covering the touched-kernel cluster — every commit, every step. The verifier runs both.

### Active directive stack (D1–D8)

| ID | Summary | Lands in commit |
|----|---------|------------------|
| **D1** | Delete `eltwise_pipeline_init`. Caller writes `compute_kernel_hw_startup(...)` themselves in `MAIN()`. Per-element bootstrap (`add_tiles_init`, `*_tile_init`, `init_bcast`, etc.) stays inside the chain. | U2 |
| **D2** | Pipeline-internal compile-time prev-CB tracking. Drop `OldCb*` from streaming elements; `prev_cb_for_idx` fold over the chain element pack at compile time; per-element reconfig becomes `if constexpr (current_cb != prev_cb)`. | P1+U1-OldCb (streaming) |
| **D3** | Doxygen on helper headers + caller-init contract spec + minimal example per chain shape. Header doxygen carries: caller's full init contract (D8), lifecycle expectations, table mapping chain shape → required pre-chain inits, 2-3 example kernel snippets per chain shape. | U6 |
| **D4** | "No production callers" is NOT a kill criterion — grep raw-LLK pattern in unmigrated kernels. Delete only what is provably absent codebase-wide (`EltwiseChainOptions`; streaming `OldCb*`; **D7** Block `OldCb*`). Keep `BinaryFpuOutputPolicy::HoistAcquireRelease`, `PackTileIndexMode::{BlockIter,Pinned,Absolute}`, `WaitUpfrontPopAtEnd`/`UpfrontReservePushAtEnd`, `OutputConditional`, `FillBitcast`, `FillInt`, `BWaitTiles`, per-side `AIndex/BIndex`. | U1, U3 |
| **D5** | `compute_kernel_hw_startup` placement contract: caller calls it as the **first statement of `MAIN()`** if the chain shape needs it; otherwise omits it (calling it elsewhere is undefined). Doxygen encodes a chain-shape → boot-needed table; the U2 sweep enforces top-of-`MAIN()` placement and removes spurious mid-`MAIN()` calls. | U2 (sweep), U6 (doc) |
| **D6** | `enable_fp32_dest_acc` is **per-element**, not chain-level. The flag is added **only to elements that emit DEST-format-sensitive LLK** (the **CARRY list**: `BinaryFpu`, `BlockBinaryFpu`, the `BinarySfpu` op-struct family in `eltwise_binary_sfpu.hpp` — `AddBinary`/`SubBinary`/`MulBinary`/`DivBinary`, `DestReuseBinary`, `BroadcastFpu`, `UnaryBcast`, `PackTile`, `BlockPackTile`). Elements that do NOT have dest-mode-dependent LLK behavior (the **SKIP list**: `CopyTile`, `BlockCopyTile` — input-side, no dest-mode-sensitive LLK selection; `FillScalar`, `RandTile`, `FillInt`, `FillBitcast` — constant fill, dest-mode-irrelevant; `OptionalChainElement<COND, Inner>` — forwards to `Inner`, which carries the flag if applicable) do NOT carry the template parameter. Pipeline machinery composes per-element fp32-dest-acc transitions into the prev-CB fold; the fold treats SKIP-list elements as transparent (passes the running prev-fp32 value through unchanged — see Q16). `EltwiseChainOptions::enable_fp32_dest_acc` (chain-level) is deleted; the field's intent is replaced by per-CARRY-element control via `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` LLK toggles (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:96-120`). | U1 (delete `EltwiseChainOptions`), U3 (per-element flag added to CARRY list only), P1 (transition logic in fold + SFINAE detection for SKIP elements), U6 (doc) |
| **D7** *(new)* | Drop `OldCb*` template params from `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`. Generalise the **D2** prev-CB fold and the **D6** prev-fp32-dest-acc fold to the block path so that the `_with_dt` two-arg LLK forms at `eltwise_block.hpp:72,236` and the explicit `srca/srcb` reconfig pair at `eltwise_block.hpp:142-143` consume chain-derived prev-CB info instead of template-passed `OldCb*`. Reverses Q12's defer in v3. | P1-block (NEW commit 3) |
| **D8** *(new)* | Helper does not wrap "BIG inits". Caller owns engine-wide setup (`compute_kernel_hw_startup`, `binary_op_init_common`, `mm_init`, `reduce_init`); chain owns per-element setup only (`add_tiles_init`, `*_tile_init`, `init_bcast`, `copy_tile_to_dst_init_short`, `reconfig_data_format_*`, `tile_regs_*` lifecycle). U6 doxygen documents the boundary; U2 audits the helper for any other BIG init. | U2 (audit + sweep), U6 (doc) |

### Refinement → audit-finding → commit-group → LOC delta

| Ref-ID | Audit IDs                            | Commit group              | Files touched                                  | LOC delta (≈)        |
|--------|--------------------------------------|---------------------------|------------------------------------------------|----------------------|
| R-1    | F-UX-1, F-UX-12 (re-do), F-UX-16, **D5**, **D8** audit | U2 (delete pipeline_init + hw_startup placement sweep + BIG-init audit) | `eltwise_chain.{hpp,inl}`, 26 prod + 13 test kernels | +60 / -200 |
| R-2    | F-UX-7, F-PERF-1 follow-up, **D6 transition fold** | P1+U1-OldCb-streaming (commit 2) | `eltwise_chain.{hpp,inl}` | +95 / -90 |
| R-2b   | **D7 block extension** | P1-block (commit 3) | `eltwise_chain.inl` (fold generalisation), `eltwise_block.hpp` (drop `OldCb*`, route via fold) | +50 / -35 |
| R-3    | F-UX-7 sweep tail, **D6 `EltwiseChainOptions` delete**, **D7 block-kernel sweep** | U1 (genuinely-dead-only) | `eltwise_chain.hpp` (decls), 17+ binary kernels (incl. block-element call sites) | +0 / -100 |
| R-4    | F-UX-2, F-UX-5, **D6 per-element flag (CARRY list only — narrowed in v5)** | U3 (`BinaryFpu` params)   | `eltwise_chain.{hpp,inl}`, `eltwise_block.hpp`, `eltwise_binary_sfpu.hpp`, ~17 kernels | +30 / -100 |
| R-5    | F-UX-1 wrapper                       | U4 (`eltwise_chain_with_init`) | `eltwise_chain.hpp`, ≤25 kernels (sweep set)   | +30 / -150 |
| R-6    | F-UX-8                               | U5 (`OptionalChainElement` adoption) | `logit_kernel.cpp`, `where_tss_kernel.cpp`, new test kernel + py | +60 / -40 |
| R-7    | F-UX-1 docs, F-UX-16 docs, **D5 placement table**, **D6 per-element notes**, **D7 block-fold docs**, **D8 caller-init contract** | U6 (Doxygen + spec) | `eltwise_chain.hpp`, `eltwise_block.hpp`, key element headers | +250 / 0 |

Net LOC reclaim: ~715 removed, ~580 added. **D7 net** is approximately +15 LOC: +20 LOC in the fold generalisation (block-path entries in the per-element traits walk), ~10 LOC removed from `eltwise_block.hpp` (`OldCb` / `OldCbA` / `OldCbB` / `OldCbOut` template lines), and ~5 LOC removed from kernel call sites (block-element trailing zeros). **D8 net** is small to negative: the only BIG init currently in the helper is `compute_kernel_hw_startup` (already removed by D1 in commit 1). No additional helper-side LOC moves; D8 ships as ~30 LOC of doxygen plus a one-line CI-grep gate documented in U6.

---

## Section B — Refinement plan

### Group U2 — Delete `eltwise_pipeline_init`; caller owns `compute_kernel_hw_startup`; **D8 BIG-init audit** (commit 1) — D1 + **D5** + **D8**

#### Commit subject
`eltwise v2: drop eltwise_pipeline_init; caller owns compute_kernel_hw_startup`

#### Audit findings addressed
- **F-UX-1** (chain typed twice) — partially addressed; the deduced wrapper in U4 finishes it.
- **F-UX-12** (binary CB pair) — addressed by relocation: the boot is now in caller hands and caller writes the correct three-CB call, so the silent-miscompile site no longer exists in the helper. The patch in `16f0b759c93` (the `first_cb_b` walk) is reverted as part of this commit. RandTile fix from the same earlier commit stays.
- **F-UX-16** (only 26/78 use pipeline_init) — addressed: there is now a single boot pattern, every kernel writes `compute_kernel_hw_startup(...)` directly, and the inconsistency disappears.
- **D5** (placement contract) — addressed: the sweep enforces top-of-`MAIN()` placement per the chain-shape table; doxygen in U6 records the rule.
- **D8** (BIG-init boundary audit) — addressed: the commit body documents the single BIG init present in the helper today (`compute_kernel_hw_startup` in `eltwise_chain.inl:793-795`) and confirms by grep that nothing else is wrapped (no `binary_op_init_common`, no `mm_init`, no `reduce_init` in `eltwise_chain.{hpp,inl}` or `eltwise_block.hpp`). The grep is included as a CI-comment marker in `eltwise_chain.hpp` and re-run in U6 doxygen.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — delete `eltwise_pipeline_init`/`eltwise_pipeline_init_for` declarations at lines 560-570.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — delete `EltwiseChainPipelineInit::run()` at lines 785-797 (`compute_kernel_hw_startup` calls at lines 793-795), `EltwisePipelineInitDispatch` at lines 804-810, `eltwise_pipeline_init` definition at lines 812-815, the `first_cb_a`/`first_cb_b`/`first_pack_cb` finders at lines 760-778 (no other consumer), and `is_reader_pred`/`is_writer_pred`/`is_binary_pred` at lines 780-781 (no other consumer).
- 26 production kernels listed below (each rewrites the `using Chain = ...; eltwise_pipeline_init<Chain>(); eltwise_chain(...);` block to `compute_kernel_hw_startup(cb_in_a, cb_in_b, cb_out); eltwise_chain(...);` placed at the top of `MAIN()`).
- 13 test kernels at `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/` — same rewrite.

Production callers (26): `data_movement/bcast/.../{bcast_h,hw,w}.cpp`, `experimental/dropout/.../dropout_kernel.cpp`, `experimental/unary_backward/gelu_backward/.../eltwise_bw_gelu_poly.cpp`, `copy/typecast/.../eltwise_typecast.cpp`, `examples/.../eltwise_sfpu.cpp`, `eltwise/binary_ng/.../{eltwise_binary_no_bcast,eltwise_binary_scalar}.cpp`, `eltwise/binary/.../{bcast_h,hw,w}.cpp`, `eltwise/unary/.../{tanhshrink,logit,eltwise_sfpu,hardswish,mish,where_tss,logsigmoid}_kernel.cpp`, `eltwise/unary_backward/{tanh_bw,gelu_bw}/.../eltwise_bw_*.cpp`, `eltwise/ternary/.../{ternary_sfpu_no_bcast_ttt,ternary_addc_ops_fpu,ternary_addc_ops_sfpu,ternary_sfpu_no_bcast_tts_tst,ternary_addcmul_int_sfpu}.cpp`.

Test kernels (13): `tests/eltwise/kernels/{binary_block,multi_chain,binary_fpu,inplace_accumulate,fanout,copy_sfpu_pack,binary_fpu_same_cb,copy_upfront,binary_fpu_bcast,fill_scalar,pack_lifecycle,copy_exp_pack,dest_reuse}.cpp`.

#### Concrete changes

**The new contract per chain shape (D5 placement table — copied verbatim into U6 doxygen):**

| Chain shape (first reader → … → trailing pack)    | Caller pre-chain init                                         | Placement |
|---------------------------------------------------|---------------------------------------------------------------|-----------|
| Unary `CopyTile<cbA, …>{} … PackTile<cbOut, …>{}` | `compute_kernel_hw_startup(cbA, cbA, cbOut);`                 | First statement of `MAIN()`. |
| Binary `BinaryFpu<cbA, cbB, …>{} … PackTile<cbOut>{}` | `compute_kernel_hw_startup(cbA, cbB, cbOut);`             | First statement of `MAIN()`. |
| `DestReuseBinary<cb, …>{}` only                   | `compute_kernel_hw_startup(cb, cb, cbOut);`                   | First statement of `MAIN()`. |
| Multi-stage (e.g. `logit_kernel.cpp`)             | One `compute_kernel_hw_startup(cbA, cbB, cbOut)` per stage. Stage 1 boots at top of `MAIN()`; subsequent stages re-boot only when the next stage's CB triple differs. | Stage 1 at top of `MAIN()`; stages 2+ immediately before that stage's chain call (NEVER inside a per-tile loop). |
| Mid-loop chain calls (moreh inner-loop pattern)   | Caller's outer `binary_op_init_common(...)` already covers the chain — no extra boot needed. | Omit; calling `compute_kernel_hw_startup` mid-`MAIN()` here is **undefined per D5**. |
| Copy-only chain whose CB formats already match defaults | Omit. | N/A (omitted). |

**D5 placement enforcement.** The U2 sweep MUST:

1. **Insert** `compute_kernel_hw_startup(...)` at the **first statement of `MAIN()`** in every kernel from rows 1–4 of the table.
2. **NOT insert** `compute_kernel_hw_startup` in kernels matching row 5 (mid-loop / outer `binary_op_init_common` already covers).
3. **Verify** no kernel ends up with a mid-`MAIN()` `compute_kernel_hw_startup` call after the sweep — this would violate D5 and the `compute_kernel_hw_startup.h` LLK contract (line 26-30 of `compute_kernel_hw_startup.h` documents "MMIO writes ... unsafe to call this function in the middle of a kernel execution").
4. **For each kernel where placement is non-trivial** (multi-stage, conditional boot, kernel-side custom prologue) the implementer flags it in the commit message body. Known non-trivial cases:
   - `logit_kernel.cpp` — multi-stage; explicit per-stage boot is correct (stage 1 at top, stage 2 mid-`MAIN()` is the documented exception). Document in U6.
   - `where_tss_kernel.cpp` — single-stage post-U5; row-1/2 of the table.
   - All 7 moreh kernels (`moreh_softmax_backward_*`, `moreh_adam`, `moreh_norm/ord_other/*`) which currently use `binary_op_init_common(...)` in their outer loop — row 5 of the table; the sweep does NOT insert a chain-helper boot for them.

**D8 BIG-init audit.** The commit body documents the grep evidence (run pre-commit on `astancov/eltwise_run7_refined` HEAD `75868c9eff4`):

```bash
grep -nE '_init_common|compute_kernel_hw_startup|mm_init|reduce_init' \
     ttnn/cpp/ttnn/kernel_lib/eltwise_{chain.hpp,chain.inl,block.hpp}
```

Pre-commit matches:
- `eltwise_chain.hpp:92` — `#include compute_kernel_hw_startup.h` (header include, not a call site).
- `eltwise_chain.hpp:560` — doxygen comment on `eltwise_pipeline_init` (deleted by D1).
- `eltwise_chain.inl:793-795` — three `compute_kernel_hw_startup(...)` call sites inside `EltwiseChainPipelineInit::run()` (deleted by D1).

Result: the **only** BIG init the helper currently wraps is `compute_kernel_hw_startup`. There is no `binary_op_init_common`, no `mm_init`, no `reduce_init`. After U2 lands, helper-side BIG-init call-site count is zero — D8's invariant is satisfied trivially.

Per-element inits that STAY in the chain (D8 boundary kept):
- `init_bcast<et, bt>(CbA, CbB, ocb)` at `eltwise_chain.inl:358` and `eltwise_block.hpp:159` — broadcast-aware analogue of `add_tiles_init`, programs one specific op variant.
- `copy_tile_to_dst_init_short` at `eltwise_chain.inl:114`; `copy_tile_to_dst_init_short_with_dt` at `eltwise_block.hpp:72`.
- `pack_reconfig_data_format` at `eltwise_block.hpp:147,234,236`; `reconfig_data_format_srca/srcb` at `eltwise_chain.inl:338-339` and `eltwise_block.hpp:142-143`.
- `add_tiles_init / sub_tiles_init / mul_tiles_init` at `eltwise_chain.inl:348-350` and `eltwise_block.hpp:150-152`.

The U2 commit body records this evidence verbatim. U6 doxygen records the boundary table (Section B group U6 below). No CI machinery is added — the grep is a one-liner the reviewer / future contributor runs ad-hoc.

Sample rewrites: `mish_kernel.cpp` collapses `using Chain = …; eltwise_pipeline_init<Chain>(); eltwise_chain(num_tiles, …);` to `compute_kernel_hw_startup(cb_input, cb_input, cb_output); eltwise_chain(num_tiles, …);` at top of `MAIN()`. `logit_kernel.cpp` (multi-stage) keeps an explicit `compute_kernel_hw_startup` per stage (stage 1 at top of `MAIN()` for input→cb_tmp0; stage 2 immediately before its chain call for cb_tmp0→cb_output — D5 row 4 exception).

`eltwise_binary_no_bcast.cpp` (D8 + D5 row 5 coexistence): the activations branch already calls `binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out)` at line 45 (raw, no chain). The block-mode chain branch adds `compute_kernel_hw_startup(cb_post_lhs, cb_post_rhs, cb_out)` BEFORE the `#if HAS_ACTIVATIONS` split — both branches need engine boot. The activations branch keeps its `binary_op_init_common`. Both calls are caller-side per D8 and coexist legally (`binary_op_init_common` runs after `compute_kernel_hw_startup` per LLK requirement).

#### Why this group?
This is the biggest semantic shift in the run. Every other group lands cleanly only after the boot path is final — U3's `BinaryFpu` reorder, U4's deduced wrapper, U5's `OptionalChainElement` adoption all reference call-site boot patterns. Land it first.

The 39-kernel sweep (26 prod + 13 test) is mechanical but large; bisecting against this commit alone is the friendliest shape.

#### Risk assessment

| Failure mode | Detector |
|---|---|
| Multi-stage kernel (logit) misses the second `compute_kernel_hw_startup` between stages | `test_unary.py::test_logit` covers Stage 2 dtype. Visible as Stage 2 producing wrong-format outputs. |
| Sweep inserts `compute_kernel_hw_startup` mid-`MAIN()` (after some other init or after a per-tile loop) — violates D5 | Reviewer catches; per `compute_kernel_hw_startup.h:26-30`, runtime symptoms are "race conditions and undefined behavior which can be hard to debug". The 401-suite covers the canonical shapes; obscure miscompiles are caught by the migrated-kernel pytest sweep. |
| Sweep inserts hw_startup in a moreh kernel that already uses `binary_op_init_common` (row 5) — D5 says omit | Two boots run; second one MMIO-clobbers state programmed by `binary_op_init_common`. Detected by `test_moreh_adam.py`, `test_moreh_softmax_backward.py`. The implementer must consult the row-5 list explicitly. |
| Sweep MISSES inserting hw_startup in a row-1/2/3 kernel (forgets) | Compile fails (chain helper assumes hw_startup ran) — actually, wait: chain helper does NOT verify hw_startup ran. Symptom is silent miscompile. Detected by 401-suite + migrated pytests. The implementer's checklist (commit body lists every kernel + its row classification) is the primary defense. |
| Caller swaps `cbA` / `cbB` order between boot and chain | Compiles; runtime miscompiles srcA/srcB. Detected by 401-suite mixed-dtype binary tests + `test_binary_bcast.py`. |
| **D8** kernel uses raw binary primitives outside the chain (e.g. `eltwise_binary_no_bcast.cpp` activations branch) and forgets to re-call `binary_op_init_common` after a chain call (chain may have re-programmed binary state via per-element `add_tiles_init`) | Detected by `test_binary_ng.py` activations-on rows. Doxygen in U6 explicitly notes that per-element `*_tiles_init` mutates math/unpack state and the kernel must re-init if it returns to raw mode. |

#### Acceptance criteria
1. 401-test suite green: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`.
2. Migrated-kernel cluster green: `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `data_movement/test_bcast.py`, `test_moreh_adam.py`, `test_moreh_softmax_backward.py`.
3. Build clean — no `eltwise_pipeline_init` symbol remains anywhere (`grep -rl eltwise_pipeline_init ttnn/` returns empty).
4. **D5 audit:** in every modified kernel, `grep -A5 'void MAIN'` shows either `compute_kernel_hw_startup(...)` as the first non-decl statement OR no `compute_kernel_hw_startup` at all. No mid-`MAIN()` placement allowed.
5. **D8 audit:** `grep -E '_init_common|mm_init|reduce_init|compute_kernel_hw_startup' ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl} ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` returns ZERO matches (only the CI-gate doxygen comment).

---

### Group P1+U1-OldCb-streaming — Pipeline-internal compile-time prev-CB tracking on streaming elements (commit 2) — D2 + **D6 transition fold**

#### Commit subject (single)
`eltwise v2: compile-time prev-CB tracking + init hoist (streaming elements)`

#### Audit findings addressed
- **F-UX-7** — `OldCb*` template parameters dead in streaming chain elements (decls + impls).
- **F-PERF-1 follow-up** — the FPU-clash gate (commit `ac595549b36`) hoists to boot-time emission; this commit closes the loop by deduceing the prev-CB-for-srca/-srcb/-pack at each element's compile-time position so the boot-time emissions don't redundantly reprogram state.
- **D6 transition fold** — the same `prev_cb_for_idx` walk also tracks per-element `EnableFp32DestAcc`. When element I has `EnableFp32DestAcc != prev`, the fold emits `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:96-120`) before that element's `init()` fires. Same compile-time-elision discipline as the prev-CB tracker.

This is the missing piece run8 has and run7 doesn't (run7-vs-run8 §4 line 133): `prev_cb_for_idx` fold over the element pack. Adding it makes the per-tile gate (already landed) emit the right minimal sequence at boot. **Note re D6:** run8 has *no* fp32_dest_acc handling in its chain header (`git show astancov/eltwise_run8:ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` finds zero matches for `fp32` / `dest_acc`). Run7's per-element D6 design is therefore **ahead** of run8 on this axis — a deliberate design choice run7 makes alone.

#### Bisect ergonomics — single commit or split?

The directive allows splitting into `(a) tracking infrastructure` and `(b) hoist gate`. **Recommendation: single commit for streaming.** Rationale:
1. The infrastructure has no other consumer than the hoist gate. Split (a) would land dead code that gets exercised only when (b) lands — a bisect failure between (a) and (b) reaches the same place.
2. The hoist gate cannot exist without the infrastructure, so commit (b) alone won't compile.
3. Both are within `eltwise_chain.inl`, single-file diff, no kernel sweep — the size is bounded.
4. **D6's transition fold** lives in the same body of code — splitting into (a)/(b) would create a third dimension of split.

The block-path generalisation (D7) is split off into commit 3 so streaming-vs-block bisects cleanly.

If the implementer finds the diff unwieldy in review, splitting after-the-fact is cheap. **Default: single commit for streaming.**

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — delete `OldCb` / `OldCbA` / `OldCbB` / `OldCbOut` template parameters from streaming elements at lines 439, 454-456, 469, 479-480, 489, 498. Public API loses these params.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — delete `OldCb*` from `CopyTile` (line 78), `PackTile` (167), `PackTileBlock` (243), `BinaryFpu` (306-308), `DestReuseBinary` (429), `UnaryBcast` (503-504). None of the implementations reference them — verified by reading the lines above.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — **untouched in this commit.** Block-element `OldCb*` migrates in commit 3 (D7).
- Add the prev-CB fold infrastructure as a `detail::` namespace block in `eltwise_chain.inl` near the existing trait-predicate impls (`eltwise_chain.inl:583-657`).

#### Concrete changes

**Step 1 — Per-element CB sentinels.** Each chain element exposes `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb` constexpr aliases (or `NO_PREV_CB = 0xFFFFFFFFu` if the element doesn't touch that side):
- `CopyTile<Cb, …>` → srca = Cb, srcb/pack = `NO_PREV_CB`.
- `BinaryFpu<CbA, CbB, …>` → srca = CbA, srcb = CbB, pack = CbOut when != 0 else `NO_PREV_CB`.
- `DestReuseBinary<Cb, …, ReuseType>` → srcb = Cb (when DEST → srca) or srca = Cb (when DEST → srcb).
- `UnaryBcast<Dim, Cb, …>` → srca = Cb.
- `PackTile<Cb, …>` → pack = Cb.
- `FillScalar` / `RandTile` / SFPU op-structs (DestOnly) → all `NO_PREV_CB`.

**Step 2 — Compile-time `prev_cb_for_idx` fold.** Add a `detail::` block:

```cpp
inline constexpr uint32_t NO_PREV_CB = 0xFFFFFFFFu;
enum class Side : uint8_t { SrcA, SrcB, Pack };

// Walk Es[0..I-1] backwards; most recent non-NO_PREV_CB on Side.
template <Side S, std::size_t I, class... Es>
constexpr uint32_t prev_cb_for_idx() { /* index_sequence fold over Es */ }
```

**Step 3 — Per-element compile-time elision.** Inside the per-element init dispatch wrap each reconfig in `if constexpr (current_cb != prev_cb) reconfig_data_format_*(current_cb);` — zero runtime cost, all elision is compile-time. Both boot-time and (under `clash_gate == true`) per-tile init paths use the same elision — the gate now decides **"emit min-set at boot only"** vs **"emit min-set per-tile too"**; the min-set itself is compile-time-computed.

**Step 3b — D6 transition fold (v5: SKIP-element-aware).** Augment the per-element walk with a parallel scan over each element's `EnableFp32DestAcc` template flag — but ONLY for elements on the CARRY list (those with DEST-format-sensitive LLK). SKIP-list elements (`CopyTile`, `BlockCopyTile`, `FillScalar`, `FillInt`, `FillBitcast`, `RandTile`, `OptionalChainElement<COND, Inner>` when `Inner` is itself a SKIP element) have NO `EnableFp32DestAcc` member; the fold treats them as **transparent** — the running prev-fp32 value passes through unchanged when the walk encounters them.

```cpp
// Symmetric to prev_cb_for_idx, but tracking the boolean fp32-dest-acc flag.
// Walks Es[0..I-1] backwards; returns the most recent EnableFp32DestAcc value
// from a CARRY-list element, or the chain's default (false) if no prior CARRY
// element exists. SKIP-list elements (no EnableFp32DestAcc member) are skipped
// transparently via the SFINAE probe below — see Q16.
template <std::size_t I, class... Es>
constexpr bool prev_fp32_dest_acc_for_idx() { /* index_sequence fold over Es */ }

// SFINAE probe — Q16's chosen mechanism. Returns E::EnableFp32DestAcc when
// the member exists; returns the supplied default otherwise (used by the fold
// to pass `prev` through unchanged for SKIP-list elements).
template <class E, bool Default, class = void>
struct fp32_or_default { static constexpr bool value = Default; };
template <class E, bool Default>
struct fp32_or_default<E, Default, std::void_t<decltype(E::EnableFp32DestAcc)>> {
    static constexpr bool value = E::EnableFp32DestAcc;
};
```

Per-element dispatch:

```cpp
template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    constexpr bool prev_fp32 = detail::prev_fp32_dest_acc_for_idx<I, Es...>();
    // For CARRY elements, curr_fp32 = E::EnableFp32DestAcc.
    // For SKIP elements (no member), curr_fp32 = prev_fp32 (transparent).
    constexpr bool curr_fp32 = detail::fp32_or_default<E, prev_fp32>::value;
    if constexpr (curr_fp32 != prev_fp32) {
        if constexpr (curr_fp32) { enable_fp32_dest_acc(); }
        else                     { disable_fp32_dest_acc(); }
    }
    // ... then the prev-CB reconfig elision per Step 3 ...
}
```

**SKIP-element handling at fold position J.** When the fold walks `Es[0..I-1]` backwards and hits a SKIP element at position J, the SFINAE probe returns `fp32_or_default<E_J, prev>::value == prev` — the running `prev` is NOT updated by element J; the walk continues backwards. SKIP elements do not "anchor" fp32 state; only CARRY elements do. (Worked walk in Q13.)

`enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` are documented as "lightweight, standalone reconfiguration that is safe to call mid-kernel without re-running compute_kernel_hw_startup" (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:84-91`). They are the right primitive for mid-chain transitions.

The fold's start-of-chain "prev" is the chain default (`false` per Q6 below). That means: if the first element has `EnableFp32DestAcc=true`, the fold emits one `enable_fp32_dest_acc()` at chain entry. If the first element has `EnableFp32DestAcc=false`, no transition emits (the kernel either had hw_startup running with bf16 default or had previously set fp32 mode itself — but the latter is not the helper's contract). This matches caller expectations: `compute_kernel_hw_startup` does NOT touch fp32-dest-acc mode, so the chain inherits whatever the kernel set up before chain entry.

**Step 4 — Drop `OldCb*` from streaming elements' template params** (the dead bookkeeping). The four trailing literal zeros (`OldCbA, OldCbB, OldCbOut`) at ~17 kernel call sites fall off in the U1 sweep. `CbOut` stays — real semantics; U3 reorders it. **Block-element `OldCb*` removal is deferred to commit 3 (D7).**

#### Why this group?
F-UX-7 (kill `OldCb*`) is bookkeeping cleanup; the bookkeeping was deleted in commit `381e193d18f` and the params have been dead since. The clean reclaim is mechanical. Bundling with the prev-CB tracking is the right call because:
1. The prev-CB infrastructure replaces what `OldCb*` was meant to thread (run-7-original-design carry-over of last-CB-on-srca per element).
2. Both touch the same per-element init bodies.
3. The D6 transition fold is structurally identical (same walk, different per-element axis); landing all three together keeps the reconfig-emission code in one logical block.
4. Bisect surface stays single (a kernel that breaks here, breaks against the new compile-time tracker — and that's where the tests want to find the regression).

The streaming-only scoping (vs full streaming+block) is the bisect-ergonomics decision in commit-3 below.

#### Risk assessment

| Failure mode | Detector |
|---|---|
| Per-element `reconfig_*_cb` sentinel set wrong on some element kind | 401-suite hits each element kind. Mismatch → wrong dtype reconfig emitted at boot → 401's `test_3_*` `fp32_dest_acc=True` fails. |
| `prev_cb_for_idx` walks the wrong direction | Fold is unit-testable in isolation; trivial to verify with a static_assert in the test kernel `multi_chain.cpp`. |
| Block-mode regresses because Block elements still use `OldCb*` while streaming elements no longer have it | Block elements untouched in this commit; their `OldCb` template params remain. No interference. Commit 3 (D7) generalises the fold. |
| Compile-time elision elides a needed reconfig (e.g. fp32_dest_acc-gated `_with_dt` form vs single-arg form) | Streaming chain uses single-arg `reconfig_data_format_srca(Cb)` only — there is no two-arg path to elide differently. The `_with_dt` form is block-only and lands in commit 3. |
| `clash_gate == true` per-tile path emits more reconfigs than before | After Step 3 elision, per-tile emissions can only *decrease* — the math always emits ≤ the unconditional pre-change shape. |
| **D6** `prev_fp32_dest_acc_for_idx` returns wrong value at element 0 (no prior element exists) | Convention: returns chain default (`false`). Q6 below records this. The fold's `if constexpr (false != element0.EnableFp32DestAcc)` correctly fires only when element 0 opts in. |
| **D6** transition fold emits `enable_fp32_dest_acc()` and `disable_fp32_dest_acc()` redundantly | Compile-time elision in Step 3b's `if constexpr (curr_fp32 != prev_fp32)` blocks redundant emission. Verified by reading the resulting MOP listing for canonical shapes (advisory, not gating). |
| **D6** kernel runs with fp32_dest_acc=true at boot (kernel-side enable_fp32_dest_acc() before chain) but chain element has `EnableFp32DestAcc=false` (default), causing fold to silently downgrade | The fold compares against chain default (`false`), not kernel state. If the kernel pre-enables fp32 and uses default chain elements, the fold emits a `disable_fp32_dest_acc()` *before* the first element — silently downgrading. **Mitigation:** Q6 below addresses this with the recommendation that `EnableFp32DestAcc` defaults derive from the `FP32_DEST_ACC_EN` macro at element instantiation site (transparent migration). |

#### Acceptance criteria
1. 401-test suite green.
2. Binary cluster green: `test_binary_ng.py`, `test_binary_bcast.py`.
3. Unary cluster green: `test_unary.py` (mish, hardswish, identity, tanhshrink — these are clash-free chains where the elision matters most).
4. Moreh cluster green: `test_moreh_adam.py`, `test_moreh_norm.py`, `test_moreh_softmax_backward.py`, `test_moreh_layer_norm.py` — these kernels currently use `#ifdef FP32_DEST_ACC_EN` for DST capacity / scalar packing; D6 must not interfere.
5. (Advisory) MOP count on `reconfig_data_format_srca` / `_srcb` / `pack_reconfig_data_format` drops at boot for clash-free chains; stays minimal but ≥1 for clash-gated chains.
6. (Advisory) **D6** MOP count on `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` is exactly the count of `EnableFp32DestAcc` transitions in the chain (verifiable for `multi_chain.cpp` test kernel built with mixed per-element flags).

---

### Group P1-block — Extend prev-CB / prev-fp32 fold to block path; drop block-element `OldCb*` (commit 3 — NEW) — D7

#### Commit subject
`eltwise v2: compile-time prev-CB tracking on block elements; drop OldCb* from BlockCopyTile/BlockBinaryFpu/BlockPackTile`

#### Audit findings addressed
- **D7 (new)** — generalises Directive 2's compile-time prev-CB fold to block-mode chain elements. Reverses Q12's defer in v3.
- **F-UX-7 tail** — block-element `OldCb*` template parameters become unreachable / re-routed via the chain-derived prev-CB info; deletion is now safe.

This commit is the bisect-friendly counterpart to commit 2: every block-element shape change is isolated. A pytest regression in `test_binary_ng.py` block-mode rows after commit 2 lands but before commit 3 lands signals a streaming-vs-block boundary issue. After commit 3 lands, the fold is uniform and the surface is consistent.

#### Pre-conditions
- Commit 2 (P1+U1-OldCb-streaming) shipped: `prev_cb_for_idx` and `prev_fp32_dest_acc_for_idx` folds exist in `eltwise_chain.inl` and operate over streaming elements via `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb` static aliases on each element.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — drop `OldCb` / `OldCbA` / `OldCbB` / `OldCbOut` template parameters from `BlockCopyTile` (line 54), `BlockBinaryFpu` (lines 119-122), `BlockPackTile` (line 221). The `_with_dt` calls at lines 72, 236 are re-routed to consume chain-derived prev-CB info instead of template-passed `OldCb*`. The single-arg reconfig at lines 142-143 (`reconfig_data_format_srca(CbA); reconfig_data_format_srcb(CbB);`) is replaced by the chain's compile-time-elided pre-element reconfig (Step 3 of commit 2).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — the per-element `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb` traits walk (introduced in commit 2 for streaming elements) is extended to recognise block elements via the same uniform accessors. Block-element `init()` bodies are augmented with the chain-derived prev-CB info — implementation choice below.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — block-element doxygen updated (provisional — final pass in U6).
- ~12 block-element call sites in production kernels — drop the trailing zeros from `BlockCopyTile<…, OldCb=0>{}`, `BlockBinaryFpu<…, OldCbA=0, OldCbB=0, OldCbOut=0, CbOut=0>{}`, `BlockPackTile<…, OldCb=0>{}`. The `CbOut` argument on `BlockBinaryFpu` stays (real semantics — used by `pack_reconfig_data_format(CbOut)` at line 147 and the `init_bcast` ocb at line 158).

Block-element call sites (pre-grep):
- `eltwise/binary_ng/.../eltwise_binary_no_bcast.cpp:74-75` — `BlockBinaryFpu<…> + BlockPackTile<…>`.
- `eltwise/binary_ng/.../eltwise_binary_scalar.cpp` (similar shape).
- ~10 other `*_bcast`/scalar variants in `eltwise/binary_ng/.../kernels_ng/`.
- Test kernels at `tests/eltwise/kernels/{binary_block,inplace_accumulate,copy_upfront}.cpp`.

#### Concrete changes

**Step 1 — Block-element trait accessors.** Each block element gains the same static accessors the streaming elements already expose post-commit-2:

```cpp
// eltwise_block.hpp — BlockCopyTile (post-commit-3)
template <uint32_t Cb,
          uint32_t BlockSize,
          Dst BaseDst                = Dst::D0,
          CopyTilePolicy Policy      = CopyTilePolicy::WaitAndPop,
          CopyTileReconfig Reconfig  = CopyTileReconfig::None>
                                                  // v5: BlockCopyTile is SKIP — no EnableFp32DestAcc.
struct BlockCopyTile : CopyTileTag {
    // ...
    static constexpr uint32_t reconfig_srca_cb = Cb;
    static constexpr uint32_t reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb = NO_PREV_CB;
    // ...
};

// BlockBinaryFpu — same pattern, exposes srca = CbA, srcb = CbB, pack = CbOut.
// BlockPackTile — exposes pack = Cb.
```

**Step 2 — Re-route `_with_dt` calls to chain-derived prev-CB.** The chain's per-element init dispatch (added in commit 2) is extended to handle block elements. For `BlockCopyTile` the dispatch produces:

```cpp
// Replaces eltwise_block.hpp:70-76 (current) — moved into the chain's per-element
// init dispatch in eltwise_chain.inl.
constexpr uint32_t prev_a  = detail::prev_cb_for_idx<Side::SrcA, I, Es...>();
if constexpr (Reconfig == CopyTileReconfig::Input && prev_a != NO_PREV_CB) {
    copy_tile_to_dst_init_short_with_dt(prev_a, Cb, /*transpose=*/0);
} else if constexpr (Reconfig == CopyTileReconfig::Input) {
    // No prior CB — first element. Use single-arg form (kernel must have set
    // pre-chain srca format via compute_kernel_hw_startup or prior raw call).
    copy_tile_to_dst_init_short(Cb);
} else {
    copy_tile_init(Cb);
}
```

Note: the `_with_dt` variant only fires when the chain's prior CB is *known* (a streaming or block reader before this `BlockCopyTile`). Otherwise we fall back to single-arg, which assumes the caller boot programmed srca correctly — D5 placement contract. This matches the existing semantics: `with_dt_tree.md:23-32` documents the two-arg form as taking explicit old/new pair when crossing a fp32-dest-acc boundary.

**Step 3 — Re-route `BlockBinaryFpu::init()`.** The current block-binary init at `eltwise_block.hpp:138-161`:

```cpp
// Current (run7) — single-arg srca/srcb reconfig.
if constexpr (DF == BinaryDataFormatReconfig::Input ||
              DF == BinaryDataFormatReconfig::InputAndOutput) {
    reconfig_data_format_srca(CbA);
    reconfig_data_format_srcb(CbB);
}
```

becomes (post-commit-3) the same compile-time-elided pre-element reconfig that streaming elements already consume — the chain's `emit_pre_element_transitions<E, I, Es...>()` emits the D6 fp32 transition, srca/srcb/pack reconfig (if `prev != cur`), then per-element `init()` (which no longer re-emits reconfig — that's hoisted to the pre-element fold).

The `pack_reconfig_data_format(CbOut)` at `eltwise_block.hpp:147` (inside `BlockBinaryFpu::init()`) and the two-arg `pack_reconfig_data_format(OldCb, Cb)` form at `eltwise_block.hpp:236` (`BlockPackTile::init()` under `PackTileReconfig::OutputConditional`) are similarly hoisted into the chain-driven fold. Post-commit-3, the `init()` body is left empty for `BlockPackTile` (or contains only the per-op LLK shape for the pack op itself, which is a no-op for a plain `pack_tile`).

The two-arg form (`pack_reconfig_data_format(prev_pack, Cb)`) is selected by the chain when `Reconfig == PackTileReconfig::OutputConditional` AND `prev_pack != NO_PREV_CB` AND `prev_pack != Cb`. This mirrors `with_dt_tree.md:23-32`: two-arg form when crossing a known boundary; single-arg for first-element / no-prior cases.

**Step 4 — Drop block-element `OldCb*` template params.** `BlockCopyTile`'s `OldCb` (line 54), `BlockBinaryFpu`'s `OldCbA, OldCbB, OldCbOut` (lines 119-121), `BlockPackTile`'s `OldCb` (line 221). The `CbOut` parameter on `BlockBinaryFpu` (line 122) STAYS — it's the pack target, threaded into `init_bcast<…>` and the chain's pack reconfig fold.

**Step 5 — Block-element call-site sweep.** Trailing zeros for `OldCb` / `OldCbA` / `OldCbB` / `OldCbOut` are dropped at every `Block*` instantiation. The `CbOut` argument on `BlockBinaryFpu` stays. In `eltwise_binary_no_bcast.cpp` the `BlockBinaryFpu<..., /*BWaitTiles=*/0, /*OldCbA=*/0, /*OldCbB=*/0, /*OldCbOut=*/0, /*CbOut=*/cb_out>` becomes `BlockBinaryFpu<..., /*BWaitTiles=*/0, /*CbOut=*/cb_out>`. Same shape across all 12 block call sites.

#### Why this group (sequencing decision)?

**Locked: ships as a SEPARATE commit AFTER commit 2 (streaming compile-time prev-CB).** Bisect ergonomics drives the split:

1. **Streaming-only bisect.** A regression in `test_unary.py` (which exercises streaming elements only) is binary-attributable to commit 2; block-mode tests in `test_binary_ng.py` cannot regress until commit 3 lands.
2. **Block-only bisect.** A regression in `test_binary_ng.py` block-mode rows is binary-attributable to commit 3; streaming-only kernels remain unchanged.
3. **Surface size.** Commit 2 is single-file (eltwise_chain.inl + eltwise_chain.hpp). Commit 3 is two-file (adds eltwise_block.hpp) plus ~12 kernel call-site rewrites. Splitting keeps each diff comprehensible.
4. **The `_with_dt` two-arg semantics are non-trivial** (`with_dt_tree.md` §Tile-copy lines 49-63 and §Pack-reconfig lines 23-32) — the implementer benefits from validating the streaming fold first, then generalising.

#### Risk assessment

| Failure mode | Detector |
|---|---|
| Block-element `init()` body still references deleted `OldCb*` after rewrite | Compile fail — caught immediately. |
| Chain's pre-element fold doesn't dispatch correctly for block elements (e.g. confuses `block_size=N` with `block_size=1` streaming) | Block-element `block_size` is independent of the fold's per-element reconfig — fold operates on `reconfig_*_cb` accessors which are uniform across streaming and block. Detected by `test_binary_ng.py` block-mode rows. |
| `_with_dt` two-arg form fires with `prev_a == NO_PREV_CB` (first element of chain is a `BlockCopyTile`) — old code unconditionally called the two-arg form with `OldCb=0` (which is invalid CB ID) | Falls back to single-arg form per Step 2 — correct behaviour and matches the v3 streaming pattern. Detected by 401-suite `test_2_*` rows that start with a `BlockCopyTile`. |
| `pack_reconfig_data_format(prev_pack, Cb)` two-arg form selected when `prev_pack` is some unrelated upstream pack (e.g. multi-stage where pack CB changed). | The fold only knows the chain's element pack — multi-stage kernels emit one `compute_kernel_hw_startup` per stage (D5 row 4) which resets pack programming. The chain's prev-pack tracking restarts at each chain call. Documented in U6. |
| Mixed streaming + block-element chain (e.g. `CopyTile + BlockBinaryFpu + BlockPackTile`) — the fold must walk both kinds | The fold reads `reconfig_*_cb` uniformly across streaming and block elements; for `EnableFp32DestAcc` the SFINAE probe (Q16) handles SKIP elements (`CopyTile`, `BlockCopyTile`) transparently — only `BlockBinaryFpu` and `BlockPackTile` participate in the block-side fp32 fold; `BlockCopyTile` is treated as transparent (passes prev value through). Mixed shapes already exist in test kernel `multi_chain.cpp`. Add a row to that test that mixes streaming and block elements explicitly. See Q13 below. |
| **D6** in block path — `BlockBinaryFpu` with `EnableFp32DestAcc=true` at element I, `BlockBinaryFpu` with `=false` at element I+1 | Same compile-time-elided transition emit as streaming. The block element's per-iter loop runs entirely under one fp32 mode (no per-iter mode toggle inside `exec()`). Block-element CARRY list: `BlockBinaryFpu`, `BlockPackTile`. Block-element SKIP list: `BlockCopyTile` (input-side, no dest-mode-sensitive LLK selection — the `_with_dt` form at `eltwise_block.hpp:72` selects on data-format pair, not dest mode). Detected by a new 401-suite row exercising mixed-mode block chain. |
| Block-element `init()` is now empty for some shapes (all reconfig hoisted to fold) → harmless but easy to write a stub method that confuses readers | Doxygen in U6 documents the shift: `init()` is now per-op LLK programming only (`add_tiles_init`, `init_bcast`, etc.); reconfig is fold-driven. |

#### Acceptance criteria
1. 401-test suite green.
2. Block-mode kernels green: `test_binary_ng.py` (no_bcast, scalar, all `*_bcast` variants), `test_ternary.py` block-mode rows, `test_eltwise.py::test_binary_block`.
3. **D7 mixed-fold gate:** add a `multi_chain.cpp` row mixing `CopyTile + BlockBinaryFpu + BlockPackTile` to verify the fold dispatches correctly across shapes.
4. (Advisory) Block-mode `_with_dt` MOP listing matches expectation: two-arg form fires only when `prev_a != NO_PREV_CB` and `prev_a != Cb`.

---

### Group U1 — Genuinely-dead surface only (commit 4) — D4 + **D6 chain options delete** + **D7 block-kernel sweep**

#### Commit subject
`eltwise v2: drop OldCb padding zeros + EltwiseChainOptions struct (~17 kernels)`

#### Audit findings addressed
- **F-UX-7** — finishes the streaming `OldCb*` removal at call sites (the `eltwise_chain.{hpp,inl}` decl removal is in commit 2; this commit is the kernel-side sweep alone).
- **F-UX-3** — deletes `EltwiseChainOptions` struct entirely (zero callers per the per-Appendix-B-candidate evidence below).
- **D7 follow-up** — the kernel-side block-element trailing-zero drop already happens in commit 3; this commit only finishes the streaming-side trailing-zero sweep + struct delete.

This was previously bundled with U1 in the original design. Per Directive 4, the sweep is narrowed to **only** the items provably absent codebase-wide. **D6** subsumes the v2 design's "delete `EltwiseChainOptions::enable_fp32_dest_acc`" line: per-element `EnableFp32DestAcc` on each element type replaces the chain-wide flag.

**Reframed for v4:** since commits 2 and 3 already purge `OldCb*` from the helper headers (streaming and block), this commit is the kernel-side residue: trailing-zero drops at streaming call sites and the `EltwiseChainOptions` struct deletion.

#### Directive 4 — Per-Appendix-B-candidate evidence

The original design's Appendix B / U1 deletion list is re-evaluated against the criterion: **"can a raw-LLK pattern matching the construct's intent be found in any unmigrated kernel under `ttnn/cpp/ttnn/operations/`?"** If yes → kept (un-migrated callers exist). If no → safe to delete.

| Construct | Appendix B verdict | New verdict (Directive 4) | Evidence |
|---|---|---|---|
| `BinaryFpuOutputPolicy::HoistAcquireRelease` | dead | **KEEP** | `eltwise/binary_ng/.../eltwise_binary_no_bcast.cpp:53-63` does `tile_regs_acquire(); for (i) {...}; tile_regs_release()` — single acquire/release wraps the loop. Future migration target. |
| `BinaryFpuOutputPolicy::PerTile` enum | dead default | **KEEP** | Pair of HoistAcquireRelease. PerTile literal becomes optional default in U3 reorder. |
| `EltwiseChainOptions::enable_fp32_dest_acc` | 0 callers | **DELETE — replaced by D6** | `grep -rn EltwiseChainOptions ttnn/cpp/` returns only helper header (`eltwise_chain.hpp:308`) and doc comment. No production caller. **D6 replaces with per-element `EnableFp32DestAcc` template flag** so the field's intent (fp32-dest-acc transitions) survives at finer granularity. |
| `EltwiseChainOptions::upfront_block_size` | 0 callers | **DELETE** | Same grep. `block_path` auto-fires on `Es::is_upfront == true` (`eltwise_chain.inl:871`). Field is redundant. |
| `EltwiseChainOptions` struct + NTTP | 0 callers | **DELETE** | Both fields go, struct goes. `eltwise_chain` signature simplifies (drops the `template <EltwiseChainOptions Opts = ...>` NTTP at `eltwise_chain.hpp:579`). |
| `PackTileIndexMode` enum (whole) | mostly dead | **KEEP** | Pattern alive in unmigrated kernels. `experimental/transformer/rotary_embedding_llama/.../rotary_embedding_llama.cpp:73,84,94,111` does `pack_tile(j, cb, j)` inside a `for (j)` loop = BlockIter. `experimental/transformer/all_reduce_create_qkv_heads/.../reduction.cpp:50` does `pack_tile(i, cb_out0, p * max_dst_tiles + i)` = Absolute. `topk.cpp:67` does `pack_tile(base_offset + 1, cb1)`. ~6 kernels. |
| `CbIndexMode::Pinned` (used in BinaryFpu / DestReuseBinary) | dead | **KEEP — migrated callers exist** | `moreh_adam.cpp:44-45` `IdxA == 0 ? CbIndexMode::FirstTile : CbIndexMode::Pinned`; `moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp:34-35,52-67` use `Pinned` as part of already-migrated kernels. |
| `WaitUpfrontPopAtEnd` / `UpfrontReservePushAtEnd` | 0 prod callers | **KEEP** | `experimental/conv3d/.../compute.cpp:86-87,94,118-119` — `cb_wait_front(in0_cb, num_tiles); ... cb_pop_front(in0_cb, dst_tiles)` upfront/end pair. Multiple variants. Future migration target. |
| `OutputConditional` (`PackTileReconfig`) | 0 callers | **KEEP** | Two-arg `pack_reconfig_data_format(old, new)` form alive in: `ssm/hc_sum_reduce/.../ssm_1d_sum_reduce.cpp:51,65`, `ssm_eltwise_mul.cpp:44,69,144`, `attn_matmul/.../transformer_attn_matmul.cpp:86`, `group_attn_matmul/.../transformer_group_attn_matmul.cpp:163`, `rotary_embedding/.../rotary_embedding.cpp:101,129`, `rotary_embedding_single_tile.cpp:76`. Two-arg form has FP32_DEST_ACC-gated semantics single-arg lacks (`with_dt_tree.md:23-32`). 8+ kernels. **Note:** post-commit-3, the chain emits the two-arg form internally when `prev_pack != Cb` — `OutputConditional` becomes the explicit caller-side opt-in (`PackTileReconfig::OutputConditional`). |
| `FillBitcast` | 0 callers | **KEEP** | `randn/.../compute_standard_normal.cpp:42` — `fill_tile_bitcast(2, two_pi)`. `eltwise/unary/common/unary_op_utils.cpp:150` — host emits `fill_tile_bitcast({}, {:#x}u);`. `reduction/accumulation/.../accumulation_program_factory.cpp:120` — `defines_kernel_args["FILL_TILE"] = "fill_tile_bitcast"`. |
| `FillInt<DF, Slot>` | 2 callers | **KEEP** | `ternary_addcmul_int_sfpu_bcast.cpp:62`, `binary_ng/.../binary_ng_program_factory.cpp:776`. Already used. |
| `OldCb*` on streaming elements | 0 callers, dead since `381e193d18f` | **DELETE — bookkeeping, replaced by Directive 2** | Per Directive 4 carve-out: `OldCb*` is bookkeeping (not a feature). The bookkeeping was deleted in `381e193d18f`. The compile-time prev-CB fold replaces it. Four trailing zeros fall off ~17 kernels' templates. |
| `OldCb*` on **Block** elements | dead at API surface; internally consumed | **DELETE — D7 drops them in commit 3** | `eltwise_block.hpp:72,236,142-143` calls re-routed via the chain-derived prev-CB fold. Block-element call-site trailing-zero sweep happens in commit 3 alongside the helper-side change (single coherent commit per D7's locked sequencing). This U1 commit only handles the streaming residue + struct delete. |
| `chain_has_non_copy_tile_fpu_clash_v` | already wired | **KEPT** | Wired by `ac595549b36` as `clash_gate`. |
| `chain_is_hoist_safe_v`, `chain_loads_share_cb_v` | computed-not-used | **KEEP** | Defensive predicates for F-PERF-5 (gate narrowing) and F-UX-21 (streaming dup-cb). |
| `BWaitTiles` (BlockBinaryFpu) | internal | **KEEP** | `eltwise_block.hpp:118` handles `BIndex == FirstTile` scalar-bcast wait shape (B = single scalar, A = block). Required for `eltwise_binary_scalar.cpp` correctness. |

**Summary of Appendix B re-evaluation:**

| Provably absent (DELETE) | Pattern alive in unmigrated kernels (KEEP — future migration targets) |
|---|---|
| `EltwiseChainOptions` struct + both fields (replaced by D6 per-element + auto-block) | `BinaryFpuOutputPolicy::HoistAcquireRelease` (`eltwise_binary_no_bcast.cpp` raw pattern) |
| `OldCb*` on streaming elements (bookkeeping) | `PackTileIndexMode::{BlockIter, Absolute, Pinned}` (rotary_embedding_llama, all_reduce_create_qkv_heads, moreh_adam) |
| `OldCb*` on Block elements (D7 — bookkeeping replaced by chain-derived fold) | `WaitUpfrontPopAtEnd` / `UpfrontReservePushAtEnd` (conv3d) |
|  | `OutputConditional` (ssm, attn_matmul, rotary_embedding) — semantics now expressed by chain-driven two-arg dispatch |
|  | `FillBitcast` (randn, unary_op_utils) |
|  | `FillInt` (ternary_addcmul_int, binary_ng_program_factory) |

#### Files to touch (U1)
~17 binary kernels: drop the four trailing zeros (`OldCbA, OldCbB, OldCbOut`) from `BinaryFpu<…>` template lists. The `CbOut` immediately following is **not** removed — U3 reorders it. Block-element call sites already swept in commit 3; this commit handles streaming-element residue.

Plus `eltwise_chain.hpp:304-312` (`EltwiseChainOptions` struct) and `eltwise_chain.hpp:579` (the NTTP on `eltwise_chain` declaration) and `eltwise_chain.inl:821,835-839` (the NTTP on the function template definition + the `Opts.upfront_block_size` references).

Sweep set (streaming residue): `data_movement/bcast/.../{bcast_h,hw,w}.cpp` (3), `eltwise/binary/.../{bcast_h,hw,w}.cpp` (3), `eltwise/binary_ng/.../{eltwise_binary_no_bcast,eltwise_binary_scalar,eltwise_binary_sfpu_no_bcast,eltwise_binary_sfpu_scalar,eltwise_where_no_bcast,eltwise_where_sfpu,eltwise_where_sfpu_scalar}.cpp` streaming branches (7), `eltwise/binary_ng/.../kernels_ng/{eltwise_binary_col_bcast,row_bcast,row_col_bcast,scalar_bcast,sfpu_row_bcast,where_sfpu_row_bcast}.cpp` streaming branches (6), `ternary/.../ternary_addc_ops_*.cpp`, `experimental/ssm/prefix_scan/.../ssm_prefix_scan.cpp`, `experimental/reduction/deepseek_grouped_gate/...`. Plus test kernels: `binary_block,binary_fpu,binary_fpu_same_cb,binary_fpu_bcast,inplace_accumulate,multi_chain`.

#### Acceptance criteria
1. 401-test suite green.
2. `test_binary_bcast.py`, `test_binary_ng.py` green.
3. `grep -rn EltwiseChainOptions ttnn/` returns empty.
4. `grep -rn 'OldCb' ttnn/cpp/ttnn/kernel_lib/' returns empty (post commits 2+3+4 — full helper-side removal verified).

---

### Group U3 — `BinaryFpu` 15→9-effective template params (commit 5) — D4 + **D6 per-element flag**

#### Commit subject
`eltwise v2: collapse BinaryFpu to 9 effective template params; reorder for callsite ergonomics`

#### Audit findings addressed
- **F-UX-2** — `BinaryFpu` 15-param surface.
- **F-UX-5** — `BinaryFpuOutputPolicy` enum dead-default → optional.
- **D6** — adds `EnableFp32DestAcc` template param ONLY to elements on the CARRY list (DEST-format-sensitive LLK). The SKIP list (input-side / fill / forwarding wrappers) gets no flag. **v5 narrows the v4 "every element" wording.**

#### Pre-conditions
- U2 shipped (callers already moved off `eltwise_pipeline_init`; D5 + D8 enforced).
- Commit 2 (P1+U1-OldCb-streaming) shipped (`OldCb*` removed from streaming `BinaryFpu` template list; D6 transition fold consumes the new flag).
- Commit 3 (P1-block, D7) shipped (`OldCb*` removed from block elements; fold extended to block path).
- U1 shipped (`EltwiseChainOptions` deleted; the `enable_fp32_dest_acc` chain-level flag is gone, so per-element is the only path).

#### Directive 4 — `AIndex/BIndex` collapse vs retain (Q4)

The original design recommended retaining the per-side split. **Directive 4 forces the search.** Result:

- `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:55-56` — passes `CbIndexMode::BlockIter` for AIndex and `CbIndexMode::FirstTile` for BIndex (the `add_bias` chain — A is pre-waited block, B is per-tile scalar).
- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp:44-45` — sets per-side mode conditionally on `IdxA == 0` and `IdxB == 0` independently.
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp:34-35` — same pattern.

**Verdict: KEEP per-side split.** Asymmetric A vs B index mode is alive in 6+ migrated kernels.

`APolicy/BPolicy` (per-side `CopyTilePolicy`) — same conclusion. Migrated kernels routinely pass mismatched policies (`A: NoWaitNoPop, B: WaitAndPop` for pre-waited A pattern). Keep split.

#### Final shape (12 named, 9 effective)

After U2, commit 2, commit 3, and U1 the surviving param list is `{CbA, CbB, Op, Bcast, OutPolicy, DfReconfig, APolicy, BPolicy, AIndex, BIndex, DstSlot, CbOut}` = 12 params.

D6 adds one more: `EnableFp32DestAcc`. The audit's "~8" target is approximate; the U3 commit lands `9 effective + 4 rarely-overridden trailing defaults` = **13 named params**.

Reductions for U3:

**Step 1 — Move `CbOut` from position 12 to position 3.** It's the most frequently overridden param (~10 binary kernels need it).

**Step 2 — `OutPolicy` becomes default-trailing.** `BinaryFpuOutputPolicy::PerTile` is the default (currently and after); `HoistAcquireRelease` stays as a value (Directive 4: pattern alive in `eltwise_binary_no_bcast.cpp`). The param moves to the end of the list with `= PerTile` default. 99% of callers stop spelling it.

**Step 3 — D6 add `EnableFp32DestAcc`.** New trailing default. Position: end of the trailing-defaults block, AFTER `OutPolicy` and `DstSlot` so prior callsite ordering is preserved up to that point. Default value: see Q6 below — recommend `false` for run7 compatibility, with a forward path for transparent migration via `FP32_DEST_ACC_EN` macro derivation.

**Step 4 — Final shape (13 named params):** `BinaryFpu<CbA(1), CbB(2), CbOut=0(3 — promoted), Op=Add(4), Bcast=None(5), DfReconfig=InputAndOutput(6), APolicy=WaitAndPop(7), BPolicy=WaitAndPop(8), AIndex=FirstTile(9), BIndex=FirstTile(10), DstSlot=D0(11 — trailing default), OutPolicy=PerTile(12 — trailing default), EnableFp32DestAcc=false(13 — D6, new trailing default)>`. 99% of callers use ≤9 positionally (everything past `BIndex` is rarely overridden — `DstSlot` only in 1 kernel, `OutPolicy` only when `HoistAcquireRelease` opts in, `EnableFp32DestAcc` only when fp32 mode folds into the chain).

The audit's "~8" target is approximate. **9 effective + 4 rarely-overridden trailing defaults** is within tolerance and preserves every Directive-4-surviving feature plus D6's per-element fp32 control.

**Step 5 — `BlockBinaryFpu`** at `eltwise_block.hpp:107-122` (post-commit-3 shape, no `OldCb*`) mirrors the same reorder. `BWaitTiles` stays (real semantics per Directive 4 evidence). `EnableFp32DestAcc` added as trailing default in the same position as `BinaryFpu`.

**Step 6 — D6 on the CARRY list only (v5 narrowing).** Only elements on the CARRY list grow their template parameter list by one to add `EnableFp32DestAcc` as a trailing default. SKIP-list elements are untouched — their template parameter lists do not change. The fold tolerates the absence via SFINAE (Q16).

| Element                     | List   | Position of `EnableFp32DestAcc` (CARRY) / Reason for skip (SKIP) |
|-----------------------------|--------|------------------------------------------------------------------|
| `BinaryFpu`                 | CARRY  | trailing (Step 4 above) |
| `BlockBinaryFpu`            | CARRY  | trailing (Step 5 above, post-D7) |
| `BinarySfpu` op-struct family (`AddBinary`, `SubBinary`, `MulBinary`, `DivBinary` in `eltwise_binary_sfpu.hpp:24,30,36,42`) | CARRY  | trailing on each struct (binary FPU/SFPU emits dest-mode-dependent LLK) |
| `DestReuseBinary`           | CARRY  | trailing (after `IndexMode`) |
| `BroadcastFpu`              | CARRY  | trailing (after `Bcast` param) — broadcast FPU is dest-format-sensitive (matches `BinaryFpu` reasoning) |
| `UnaryBcast`                | CARRY  | trailing (after `Reconfig`) |
| `PackTile`                  | CARRY  | trailing (after `Reconfig`) — pack readout from DEST is dest-mode-sensitive |
| `BlockPackTile`             | CARRY  | trailing (after current params, post-D7) |
| `CopyTile`                  | SKIP   | Input-side; copies into DEST without dest-mode-sensitive LLK selection. No template change. |
| `BlockCopyTile`             | SKIP   | Block input-side; same reasoning as `CopyTile`. The `_with_dt` two-arg form at `eltwise_block.hpp:72` selects on data-format pair, not dest mode. No template change. |
| `FillScalar`                | SKIP   | Constant fill; dest-mode-irrelevant. No template change. |
| `FillInt`                   | SKIP   | Constant fill of integer dtype; dest-mode-irrelevant. No template change. |
| `FillBitcast`               | SKIP   | Constant fill via bitcast; dest-mode-irrelevant. No template change. |
| `RandTile` (NTTP-seed fix from `16f0b759c93`) | SKIP   | Random tile fill; dest-mode-irrelevant. No template change. |
| `OptionalChainElement<COND, Inner>` | SKIP (transparent forwarder) | Forwards to `Inner`. The wrapper itself does not expose `EnableFp32DestAcc`; if `Inner` is a CARRY element, `Inner::EnableFp32DestAcc` participates in the fold via the SFINAE probe seeing through to the inner. (`eltwise_optional.hpp:55-84` — the conditional ladder selects an `Inner` or a no-op tag-only specialisation; either way the wrapper exposes no `EnableFp32DestAcc` directly.) |
| SFPU unary op-structs (CRTP base `UnaryOp`/`TernaryOp`/`QuaternaryOp` in `eltwise_chain.hpp:314-329`) | mixed (carry per struct) | DEST-format-sensitive unary SFPU ops (those reading/writing DEST under dest-mode-dependent LLK) carry the flag at the struct definition level. Pure SFPU compute (math on DEST values) is dest-mode-sensitive — see Q6 default discussion. CRTP base provides `static constexpr bool EnableFp32DestAcc = false;` default that derived structs override. |

Each CARRY element exposes `static constexpr bool EnableFp32DestAcc = …;`; the P1 fold's `prev_fp32_dest_acc_for_idx<I, Es...>()` reads it via the SFINAE probe (Q16). SKIP elements have no such member — the probe returns the supplied default (the running `prev`), passing through unchanged. Authors writing a chain composed only of SKIP elements (e.g. `CopyTile + FillScalar`) do not need to think about fp32 mode at all — the chain-default `false` flows through with no transitions emitted.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — `BinaryFpu` decl at lines 442-458 → new 13-param shape (CARRY). Other CARRY-list streaming elements: `DestReuseBinary` at 460-470, `UnaryBcast` at 472-481, `PackTile` at 483-490, `PackTileBlock` at 492-499 — each grows by one trailing default. SKIP-list streaming elements (`CopyTile` at 432-440, `FillScalar`/`FillInt`/`FillBitcast`/`RandTile` at 502-509) are NOT touched (no `EnableFp32DestAcc` member added).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — `BinaryFpu` impl at line 295-416 (template-arg-list head only; body untouched). Other CARRY elements: same surgical head-only edit. SKIP-element impls untouched. CRTP base `BinaryOp` at `eltwise_chain.hpp:314-329` gets a `static constexpr bool EnableFp32DestAcc = false;` declaration that derived CARRY structs (e.g. `BinarySfpu` op-struct family) override; `UnaryOp`/`TernaryOp`/`QuaternaryOp` bases get the same default for derived CARRY structs (harmless on SKIP-derived structs — the SFINAE probe returns the inherited member matching the chain default).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — `BlockBinaryFpu` decl at 107-122 (post-D7) and `BlockPackTile` at line 222 grow by one trailing default (CARRY). `BlockCopyTile` at line 55 is NOT touched (SKIP).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp` — `AddBinary`/`SubBinary`/`MulBinary`/`DivBinary` at lines 24/30/36/42 grow by one trailing default each (CARRY).
- 17 binary kernels (same U1 set, second pass — implementer reorders the template lists). Most kernels do NOT need to spell `EnableFp32DestAcc` because the default `false` matches their behavior (see "D6 grep findings" in the final report).

#### Risk assessment

| Failure mode | Detector |
|---|---|
| Argument-position errors | Compile-time fail at every miswritten call site. No silent runtime bug. |
| Default `EnableFp32DestAcc=false` silently downgrades a kernel that pre-enabled fp32-dest-acc (caller called `enable_fp32_dest_acc()` before chain) | The fold's `prev_fp32_dest_acc_for_idx` start-of-chain returns chain default (`false`), so element 0 with default `false` → no transition emits → fp32 mode preserved. **No downgrade actually occurs at element 0.** Downgrade can only occur if the chain has a mid-chain element with `EnableFp32DestAcc=false` *after* an element with `EnableFp32DestAcc=true` — that's intended D6 behavior. |
| `EnableFp32DestAcc=true` in a chain whose kernel was built without `FP32_DEST_ACC_EN` define → DST capacity halved at element-time but caller assumes 8-tile DST | D6 does NOT change DST capacity. `enable_fp32_dest_acc()` is a *mode* toggle for math/pack readout, not a DST resize. DST capacity is governed by `DST_ACCUM_MODE` at hw_startup time (`compute_kernel_hw_startup.h:41-50`). If a kernel enables fp32 mode mid-chain on a kernel built without `FP32_DEST_ACC_EN`, the math writes 32-bit values into a 16-bit DST → garbage. **Mitigation:** the P1 fold's transition does NOT enable fp32 unless the kernel-level `DST_ACCUM_MODE` permits it. We surface this as a `static_assert` on `BinaryFpu` (and friends): `static_assert(!EnableFp32DestAcc || DST_ACCUM_MODE, "EnableFp32DestAcc requires kernel built with FP32_DEST_ACC_EN");` — enforces D6 + DST sizing coherence at compile time. |
| Migrated moreh kernels currently use `#ifdef FP32_DEST_ACC_EN` for DST size / scalar packing (15+ kernels, see grep findings in final report) | These ifdefs are NOT replaced by D6. D6's `EnableFp32DestAcc` is the helper's mid-chain transition flag; the kernel's own DST sizing remains its own concern. Tested by running `test_moreh_*.py` with both `fp32_dest_acc=True` and `False` parametrize values. |

- 401-suite covers `BinaryFpu` exhaustively.

#### Acceptance criteria
1. 401-test suite green.
2. Every `BinaryFpu` / `BlockBinaryFpu` caller compiles.
3. Migrated-kernel pytests for the 17 kernels green.
4. **D6** sub-test row in the 401-suite (or a new test): `test_binary_fpu_per_element_fp32_dest_acc` — chain with element 0 `EnableFp32DestAcc=true`, element 1 `=false`, element 2 `=true` exercises both transitions. Expected MOP listing: 1× `enable_fp32_dest_acc`, 1× `disable_fp32_dest_acc`, 1× `enable_fp32_dest_acc` at element starts.

---

### Group U4 — `eltwise_chain_with_init` deduced wrapper (commit 6)

#### Commit subject
`eltwise v2: add eltwise_chain_with_init deduced wrapper; sweep callers`

#### Audit findings addressed
- **F-UX-1** (chain typed twice).
- **F-UX-16** (only 26/78 use pipeline_init — replaced by uniform per-call wrapper).

After U2 ships, every kernel writes `compute_kernel_hw_startup(...)` directly. F-UX-1's pain is no longer "double-typing the chain" — that's gone — but a deduced wrapper still provides ergonomic value: it deduces the boot's three CBs from the chain elements and emits the boot.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — add new function template after `eltwise_chain` declaration.
- ≤25 production kernels (the 26 from the U2 set MINUS multi-stage kernels that need explicit boot per stage — `logit_kernel.cpp` is the known multi-stage exclusion). Each kernel collapses `compute_kernel_hw_startup(cbA, cbB, cbOut); eltwise_chain(num_tiles, ...)` to `eltwise_chain_with_init(num_tiles, ...)`.

#### Concrete change

`eltwise_chain_with_init<class... Es>(uint32_t n_tiles, Es... elts)` template-deduces three CBs from the chain pack at compile time: `cb_a` ← first reader's `cb_a_id()` (`is_cb_reader_op_v`); `cb_b` ← first binary's `cb_b_id()` (`is_binary_fpu_op_v || is_dest_reuse_binary_op_v`), fallback `cb_a` for unary; `cb_out` ← first writer's `pack_cb_id()`. `static_assert(cb_a != 0 && cb_out != 0)`. Body is `compute_kernel_hw_startup(cb_a, cb_b_or_a, cb_out); eltwise_chain(n_tiles, elts...);`. The CB folds are the same walk that was inside the deleted `EltwiseChainPipelineInit::run()`. Pure header (`eltwise_chain.hpp`), inline.

Doxygen warns: "use this only for single-stage kernels" — multi-stage (different PACK output CB per stage) must call `compute_kernel_hw_startup` themselves per stage; the deduced wrapper would emit it once and stage 2's PACK would target the wrong CB.

#### Why this group?

This is the largest LOC reclaim of the run (~150 LOC across 25 kernels). Bundling with U2 would have made the boot-deletion commit also a wrapper-add commit; bisect surface doubles. Ship after U2 has settled.

`logit_kernel.cpp` stays on the explicit per-stage boot pattern — handled in U5 (which adopts `OptionalChainElement` inside the existing two-stage shape).

`where_tss_kernel.cpp` adopts `OptionalChainElement` (per Directive 1 / Q3) and migrates to `eltwise_chain_with_init` since it's a single-stage kernel after the `OptionalChainElement` rewrite.

#### Risk assessment

| Failure mode | Detector |
|---|---|
| The deduced fold finds wrong reader/writer (e.g. `DestReuseBinary` confusing the binary detector) | Same fold as old `EltwiseChainPipelineInit::run()` (verified by `16f0b759c93` test run); 401-suite green confirms. |
| Kernel author adopts wrapper but the kernel is multi-stage | Compile-time success but runtime miscompile on stage 2 PACK. Documented in doxygen + caught by `test_unary.py::test_logit` if logit_kernel.cpp is mistakenly migrated. **logit explicitly excluded.** |
| Sweep miss leaves kernel on legacy explicit boot | Both APIs coexist; legacy keeps working. Not a regression. |
| **D5** caller of `eltwise_chain_with_init` places it mid-`MAIN()` | The wrapper internally calls `compute_kernel_hw_startup` — same MMIO unsafety per `compute_kernel_hw_startup.h:26-30`. Doxygen warns; reviewer catches. |

#### Acceptance criteria
1. 401-test suite green.
2. All swept kernels' tests green: `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `test_bcast.py`.

---

### Group U5 — `OptionalChainElement` adoption (commit 7)

#### Commit subject
`eltwise v2: adopt OptionalChainElement in logit + where_tss; ship test kernel`

#### Audit findings addressed
- **F-UX-8** — `OptionalChainElement` has zero production callers.

#### Files to touch
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logit_kernel.cpp` (currently 66 LOC).
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp` (currently 69 LOC) — per Directive 1 / Q3, **migrate** by introducing a `constexpr bool` flag derived from the `INP_*` macros and using `OptionalChainElement` (no `#ifdef` around chain elements).
- New test kernel `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/optional_element.cpp` (per Q5 — yes, ship).
- `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` — add `test_optional_chain_element` parametrize.

#### Concrete changes

**`logit_kernel.cpp` rewrite:** `#ifdef CLAMP` collapses to `constexpr bool DO_CLAMP`. Stage 1 chain becomes `CopyTile + OptionalChainElement<DO_CLAMP, Clamp<Dst::D0>>{clamp_inner} + PackTile`. Stage 1 calls `compute_kernel_hw_startup(cb_input, cb_input, cb_tmp0)` at top of `MAIN()`; stage 2 (unconditional, no `OptionalChainElement`) re-boots with `compute_kernel_hw_startup(cb_tmp0, cb_tmp0, cb_output)` per D5 multi-stage exception.

**`where_tss_kernel.cpp` rewrite (per Directive 1 / Q3):** the two `#ifdef`-gated halves (`INT32/UINT32` vs `FLOAT/FLOAT32`) substituting `FillInt` for `FillScalar` collapse to a `constexpr bool USE_INT_FILL = defined(INP_INT32) || defined(INP_UINT32)` flag. The chain is a single 7-element list: `CopyTile + OptionalChainElement<USE_INT_FILL, FillInt<...>>{packed_scalar1} + OptionalChainElement<USE_INT_FILL, FillInt<...>>{packed_scalar2} + OptionalChainElement<!USE_INT_FILL, FillScalar>{*true_value} + OptionalChainElement<!USE_INT_FILL, FillScalar>{*false_value} + WhereSfpu + PackTile`. The `<false, …>` specialization is a tag-only no-op (`eltwise_optional.hpp:55-84`); dead branches cost zero at runtime. `#ifdef` is gone from the chain body.

After this rewrite, `where_tss_kernel.cpp` is single-stage and can adopt U4's `eltwise_chain_with_init` wrapper. **Land that as part of U5** (one extra line per kernel — collapses the boot + chain to a single call). No need to revisit in U4's sweep.

**`optional_element.cpp` test kernel** — exercises both `<true, Inner>` and `<false, Inner>` paths over each tag the conditional-ladder dispatches to (`PackTileTag`, `CopyTileTag`, `BinaryFpuTag`, `DestReuseBinaryTag`, `UnaryBcastTag`, `FillTileTag`, `RandTileTag`, `DestOnlyTag`). Shape: a `CopyTile + OptionalChainElement<COND, Inner> + Pack` chain run with golden vs. raw-bypass comparison.

**Test py** — `test_optional_chain_element` parametrize: `(COND ∈ {True, False}) × (Inner ∈ {Mask, Clamp, Eqz})`. Validates the `std::conditional_t` ladder at `eltwise_optional.hpp:57-65`. Ship in same commit.

#### Why this group?
F-UX-8 explicit user direction: adopt in 2 production kernels + a test. Smallest scope, lowest risk. Shipped last so nothing else depends on it.

#### Risk assessment
- `OptionalChainElement<false, FillInt<…>>` inheriting `FillTileTag` — verified at `eltwise_optional.hpp:63-65`. The chain pipeline's predicates (`is_fill_tile_op_v`) classify it correctly via inherited tag.
- A future inner-kind addition (e.g. `Mask` element) without updating the `std::conditional_t` ladder → silent miscompile. Mitigation: the test kernel covers each kind explicitly; new kinds need new test rows.

#### Acceptance criteria
1. 401-suite green including new `test_optional_chain_element`.
2. `test_unary.py::test_logit` passes (both CLAMP=on/off variants).
3. `test_unary.py::test_where` passes (all of `INP_INT32`, `INP_UINT32`, `INP_FLOAT`, `INP_FLOAT32` variants).

---

### Group U6 — Doxygen documentation pass (commit 8) — D3 + **D5** + **D6** + **D7** + **D8**

#### Commit subject
`eltwise v2: doxygen + caller-init contract spec on chain helper headers`

#### Audit findings addressed
- **D3** Doc — supports F-UX-1, F-UX-16 (caller-init contract was implicit).
- **D5** placement contract table.
- **D6** per-element fp32_dest_acc usage notes.
- **D7** block-element prev-CB fold note (block path now mirrors streaming).
- **D8** caller-init boundary table — the full caller responsibility list, not just `compute_kernel_hw_startup`.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — top-of-file doxygen block; per-element `///` blocks on `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`, `eltwise_chain`, `eltwise_chain_with_init`.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — top-of-file doxygen + per-element on `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile` (D7 block-path note).
- Key element headers — `eltwise_fill.hpp`, `eltwise_optional.hpp`, `eltwise_rand.hpp`, `eltwise_helpers.hpp` (aggregator).

#### Required content per file

**`eltwise_chain.hpp` — top-of-file doxygen block** with these sub-sections:

- `@section caller_init_contract` — **D8 boundary table.** The full caller-side responsibility:

  | Init kind | Owner | When to call | Notes |
  |---|---|---|---|
  | `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` | **caller** | First statement of `MAIN()` (D5). | Engine boot. MMIO-unsafe mid-kernel. Required for chains that read/write CBs. |
  | `binary_op_init_common(cb_a, cb_b, cb_out)` | **caller** (when applicable) | Once per `MAIN()`, before any chain or raw binary call. | Required when the kernel mixes raw binary primitives (`add_tiles`, `binary_tiles_init<...>`) with chain calls; not required for chain-only kernels. |
  | `mm_init(...)` | **caller** | N/A for eltwise chain (chain is eltwise-only). | If a kernel mixes matmul and chain, kernel author owns `mm_init` placement. |
  | `reduce_init<...>(...)` | **caller** | N/A for eltwise chain. | Same as `mm_init`. |
  | `add_tiles_init / sub_tiles_init / mul_tiles_init / init_bcast<...>(...)` | **chain** | Per-element, run before each binary element's `exec()`. | Chain owns the per-element programming. Caller does NOT call these. |
  | `copy_tile_to_dst_init_short(cb)` / `copy_tile_to_dst_init_short_with_dt(prev, cb)` | **chain** | Per-`CopyTile`/`BlockCopyTile`, fold-driven prev-CB choice. | Chain emits the right form based on `prev_cb_for_idx`. |
  | `reconfig_data_format_srca/srcb(cb)` / `pack_reconfig_data_format(cb)` / `pack_reconfig_data_format(prev, cb)` | **chain** | Per-element, fold-driven (D2 + D7). | Compile-time-elided when `prev == cur`. |
  | `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` | **chain** | Per-element fold transition (D6). | Compile-time-elided when `prev_fp32 == cur_fp32`. |
  | `tile_regs_acquire / commit / wait / release` | **chain** | Per-iteration. | Chain owns the lifecycle. |

- `@section caller_init_contract_examples` — three worked examples (chain-only single-stage `mish_kernel.cpp` shape, chain + raw binary `eltwise_binary_no_bcast.cpp` shape with `compute_kernel_hw_startup` at top followed by `binary_op_init_common`, and mid-loop chain moreh row-5 `moreh_adam.cpp` shape with kernel-side `binary_op_init_common` and NO `compute_kernel_hw_startup`).

- `@section caller_init_wrong_way` — **D8 worked anti-example.** Three failure modes:
  1. Calling `compute_kernel_hw_startup` mid-`MAIN()` (after some other init) → undefined behaviour per `compute_kernel_hw_startup.h:26-30` (MMIO write mid-kernel; race conditions under load).
  2. Chain-only kernel forgetting `compute_kernel_hw_startup` entirely → silent miscompile; default-format reconfigs may match by accident; first mismatched-dtype kernel produces garbage.
  3. `EnableFp32DestAcc=true` on a kernel built without `FP32_DEST_ACC_EN` → compile-time fail per the `static_assert` in `BinaryFpu` (Q10).

- `@section hw_startup_placement` — **D5 explicit rule** (verbatim from group U2 above): `compute_kernel_hw_startup` is the first statement of `MAIN()` if the chain shape requires it; mid-`MAIN()` is undefined per `compute_kernel_hw_startup.h:26-30`. Multi-stage exception per row 4 of the placement table. **Q15 resolution:** the same once-at-MAIN()-top rule applies symmetrically to other BIG inits (`binary_op_init_common`, `mm_init`, `reduce_init`).

- `@section per_element_fp32_dest_acc` — **D6 explicit rule (v5-narrowed):** only **CARRY-list** elements (DEST-format-sensitive LLK) expose `EnableFp32DestAcc` as a template parameter (default `false`). The CARRY list: `BinaryFpu`, `BlockBinaryFpu`, `BinarySfpu` op-struct family (`AddBinary`/`SubBinary`/`MulBinary`/`DivBinary`), `DestReuseBinary`, `BroadcastFpu`, `UnaryBcast`, `PackTile`, `BlockPackTile`. The SKIP list (`CopyTile`, `BlockCopyTile`, `FillScalar`, `FillInt`, `FillBitcast`, `RandTile`, `OptionalChainElement<COND, Inner>`) does NOT expose the template parameter. The fold (Q16) detects the absence via SFINAE and treats SKIP elements as transparent (passes prev value through). The fold emits `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` at transitions; `static_assert(!EnableFp32DestAcc || DST_ACCUM_MODE)` (Q10) fires only on CARRY elements (naturally — only they declare the flag). Doxygen warns the chain inherits NO fp32 state from the kernel — use the per-CARRY-element flag uniformly. Authors writing a chain composed only of SKIP elements (e.g. CopyTile + Fill) do not think about fp32 mode at all unless the kernel itself preset it.

- `@section deduced_wrapper` — when to use `eltwise_chain_with_init` vs. the explicit boot (single-stage only).

- `@section lifecycle` — acquire/release window shape, `init()` re-fire gating on `chain_has_non_copy_tile_fpu_clash_v`, `pack_reconfig` boot-only behaviour after commits 2+3, D6 `enable_fp32_dest_acc()` per-element fold transition. **D7 block-path note:** block-element `init()` bodies no longer emit reconfig — that's fold-driven; `init()` only programs the per-op LLK shape (`add_tiles_init`, `init_bcast`).

- `@section examples` — five canonical kernel snippets: unary (CopyTile + SFPU + Pack); binary (BinaryFpu + Pack); two-stage logit-style with re-boot; D6 mixed-fp32 chain; **D7** block + streaming mixed chain illustrating uniform fold dispatch.

**Per-element `///` blocks** on `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock` — `@tparam` for every parameter; the new `EnableFp32DestAcc` `@tparam` is present **only on CARRY-list elements** (`BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`). SKIP-list elements (`CopyTile`) get an explicit doxygen note: "this element is dest-mode-transparent — it does not affect or read the chain's fp32-dest-acc state; cross-element transitions are decided by adjacent CARRY elements". Both lists carry the explicit "the element does NOT emit `compute_kernel_hw_startup` — caller's pre-chain init must cover its CBs" reminder; cross-link `@ref caller_init_contract` and `@ref per_element_fp32_dest_acc`.

**`eltwise_block.hpp`** — top-of-file doxygen explaining "block elements use the same prev-CB fold as streaming elements (post-D7); no `OldCb*` template params, no per-element reconfig in `init()`. The `_with_dt` two-arg LLK form fires at the chain level, fold-driven." + same D6 per-element fp32 notes; per-element `///` blocks on `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`.

**Element headers** — `eltwise_fill.hpp`, `eltwise_optional.hpp`, `eltwise_rand.hpp` — short header doxygen + per-struct `@brief` + `@tparam` lines. Per v5: these are SKIP-list elements; they do NOT expose `EnableFp32DestAcc`. Doxygen explicitly notes "dest-mode-transparent: this element does not affect or read the chain's fp32-dest-acc state; constant fill / forwarding has no DEST-format-dependent LLK". `eltwise_optional.hpp` documents that `OptionalChainElement<COND, Inner>` forwards to `Inner` — when `Inner` is a CARRY element, the inner's `EnableFp32DestAcc` participates in the fold (the SFINAE probe sees through the wrapper because the wrapper's chosen specialisation IS the `Inner` struct, per the `std::conditional_t` ladder at lines 57-65).

**Aggregator** `eltwise_helpers.hpp` — list which element-family headers it pulls and what each ships.

#### Why ship as a separate commit?
- Pure documentation; cannot regress runtime.
- Bisect surface zero.
- Easier review (a single "doc commit" is grep-friendly).
- Captures the U2 boot-contract decision so future kernel authors don't re-discover it from code.
- Encodes D5 + D6 + D7 + D8 explicitly so future implementers don't need to reverse-engineer the placement, per-element-fp32, block-fold, and BIG-init-boundary invariants.

#### Risk assessment
None — pure comments / doc strings.

#### Acceptance criteria
1. 401-suite green (no functional change).
2. `git diff --stat` shows only comment additions; reviewer can confirm by inspection.
3. **D8 CI-grep gate** documented in the doxygen block: `grep -E '_init_common|mm_init|reduce_init|compute_kernel_hw_startup' eltwise_chain.{hpp,inl} eltwise_block.hpp` returns ZERO matches.

---

## Section C — Sequencing & dependencies

```
U2 ──> P1+U1-OldCb-streaming ──> P1-block (D7) ──> U1 ──> U3 ──> U4 ──> U5 ──> U6
```

### Hard sequencing rules

1. **U2 first.** Boot semantics are the foundation. Every later commit references the new caller-owns-boot contract — including U6's documentation. **D5 placement contract** is a U2 sweep concern, not a helper-side change. **D8 BIG-init audit** lands in U2 because the only BIG init currently in the helper is `compute_kernel_hw_startup` (whose removal U2 already does); the audit is a one-line grep evidence comment.
2. **Streaming prev-CB fold (commit 2) before block prev-CB fold (commit 3).** D7 explicitly locks this sequencing for bisect ergonomics — a regression in `test_unary.py` is binary-attributable to commit 2; a regression in `test_binary_ng.py` block-mode rows is binary-attributable to commit 3.
3. **Commit 3 (D7 block) before U1.** Block-element call-site sweep happens in commit 3 alongside the helper change (single coherent commit per D7 lock). U1 only handles streaming-element residue + struct delete.
4. **U1 before U3.** U3 reorders the surviving params and ADDS the per-element `EnableFp32DestAcc`. Reordering with deletions interleaved is one tangled diff; serializing produces two clean diffs. U1 also deletes `EltwiseChainOptions` (whose `enable_fp32_dest_acc` field motivates D6's per-element replacement) — landing the chain-wide field deletion before the per-element addition makes the migration story coherent: "chain-wide knob deleted, per-element flag added in its place."
5. **U3 before U4.** U4 sweeps every chain caller for the `eltwise_chain_with_init` rewrite. If U3 hasn't reordered first, U4 sweeps with the old shape; then U3 re-sweeps. Two passes over the same files is double the bisect surface.
6. **U5 after U4.** U5 migrates `where_tss` to a single-stage shape that uses U4's wrapper.
7. **U6 last.** Documentation reflects the final shape. **D5 placement table**, **D6 per-element fp32 notes**, **D7 block-path notes**, and **D8 caller-init contract** can only be authored after U2 (D5 sweep + D8 audit proven), commit 3 (D7 block path landed), and U3 (D6 per-element flag landed).

### Files-touched matrix (overlap risk)

| Commit | `eltwise_chain.hpp` | `eltwise_chain.inl` | `eltwise_block.hpp` | Migrated kernels | Test kernels |
|--------|---------------------|---------------------|---------------------|------------------|--------------|
| 1: U2     | yes (delete pipeline_init decl + D8 grep comment) | yes (delete pipeline_init impl + finders) | no | 26 | 13 |
| 2: P1+U1-OldCb-streaming | yes (decls — streaming only) | yes (impl + new fold + D6 transition fold) | no | no | no |
| 3: P1-block (D7) | minor (block doxygen pointer; final pass in U6) | yes (fold extension to block elements) | yes (drop `OldCb*`, route via fold) | ~12 (block call sites) | ~3 (binary_block, inplace_accumulate, copy_upfront) |
| 4: U1     | yes (delete `EltwiseChainOptions`) | yes (delete `Opts` NTTP) | no | ~17 (streaming literal sweep residue) | ~6 |
| 5: U3     | yes (BinaryFpu decl + per-element `EnableFp32DestAcc`) | yes (head only + CRTP base) | yes (BlockBinaryFpu + Block elements + per-element flag) | ~17 | ~6 |
| 6: U4     | yes (new wrapper) | no | no | ~25 | no |
| 7: U5     | no | no | no | 2 (logit, where_tss) + new test kernel | yes (1 new) |
| 8: U6     | yes (doxygen) | maybe (doxygen on impl-internal types) | yes (doxygen) | no | no |

Commits 1 and 6 (U2 / U4) both sweep largely-overlapping kernel sets — but at different abstraction layers (U2: explicit boot, U4: deduced wrapper). Serializing means each commit is bisectable on its own concern. Commits 2 and 3 (streaming / block fold) are non-overlapping at file-region level — commit 2 doesn't touch `eltwise_block.hpp`; commit 3 only adds entries to the existing fold infrastructure. U1 and U3 both touch `eltwise_chain.hpp` head — U1 deletes one struct (`EltwiseChainOptions`) and unbinds the NTTP; U3 modifies struct decls (add `EnableFp32DestAcc`). The two are non-overlapping at the file-region level.

**D7 commit-3 dependency on commit 2:** the fold infrastructure (`prev_cb_for_idx`, `prev_fp32_dest_acc_for_idx`) is added in commit 2 as a streaming-only path; commit 3 extends it to recognise block-element accessors. Commit 3 cannot land before commit 2 (the fold doesn't exist) and commit 2's streaming path is untouched by commit 3 (the extension is additive on the per-element trait walk).

**D8 has no file-touch overlap with other commits beyond U2 + U6.** The audit is a grep comment in U2; the doxygen contract is in U6.

---

## Section D — Acceptance test plan

### Per-commit gate

Every commit:

1. 401-test suite green: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`.
2. Migrated-kernel pytest sample for the touched-kernel cluster:

| Commit | Pytest set                                                                                          |
|--------|-----------------------------------------------------------------------------------------------------|
| 1: U2     | `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `test_bcast.py` (data_movement), `test_moreh_adam.py`, `test_moreh_softmax_backward.py` (D5: row-5 omit verification; D8: grep-evidence in commit body) |
| 2: P1+U1-OldCb-streaming | `test_binary_ng.py`, `test_binary_bcast.py`, `test_unary.py` (clash-free chains: mish, hardswish, identity, tanhshrink), **D6**: any 401-row already running with `fp32_dest_acc=True` parametrize (the existing `test_3_*` rows in the 401-suite already exercise this — verify D6 doesn't regress them) |
| 3: P1-block (D7) | `test_binary_ng.py` (no_bcast, scalar, all `*_bcast` block-mode rows), `test_ternary.py` block-mode rows, `test_eltwise.py::test_binary_block`, **D7**: new `multi_chain.cpp` row mixing streaming + block elements |
| 4: U1     | `test_binary_bcast.py`, `test_binary_ng.py` |
| 5: U3     | same as U1 + `test_moreh_softmax_backward.py`, `test_moreh_adam.py`, `test_moreh_layer_norm.py` + **D6**: new `test_binary_fpu_per_element_fp32_dest_acc` 401-suite row exercising mixed-mode chain |
| 6: U4     | `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `test_bcast.py` |
| 7: U5     | `test_unary.py::test_logit`, `test_unary.py::test_where`, `test_eltwise.py::test_optional_chain_element` |
| 8: U6     | 401-suite only (no functional change) |

### D5 placement-coverage gate (U2)

The U2 pytest gate MUST include every kernel where D5's placement reasoning is non-trivial:

| Kernel | D5 row | Required pytest |
|---|---|---|
| `logit_kernel.cpp` | row 4 (multi-stage) | `test_unary.py::test_logit` (both CLAMP variants) |
| `where_tss_kernel.cpp` (post-U5: row 1/2; pre-U5: row 5 with custom prologue) | row 5 → row 1/2 | `test_unary.py::test_where` |
| `moreh_adam.cpp` | row 5 (outer `binary_op_init_common`) | `test_moreh_adam.py` |
| `moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp` | row 5 | `test_moreh_softmax_backward.py` |
| `moreh_norm/ord_other/{h,w,nc}.cpp` | row 5 | `test_moreh_norm.py` |
| All 7 binary_ng `*_bcast.cpp` files | row 2 | `test_binary_bcast.py`, `test_binary_ng.py` |
| `bcast_h.cpp`, `bcast_hw.cpp`, `bcast_w.cpp` (data_movement and eltwise/binary) | row 2 | `test_bcast.py` |
| `dropout_kernel.cpp` | row 1 | `test_unary.py::test_dropout` |
| `eltwise_typecast.cpp` | row 1 | `test_unary.py::test_typecast` |

The implementer's commit body for U2 lists every kernel + its row classification. The reviewer cross-checks against this table.

### D6 per-element-fp32 coverage gate (U3 / commit 2)

At minimum one row in the 401-suite must exercise:
- A chain with element 0 `EnableFp32DestAcc=true`, element 1 `=false`, element 2 `=true`. Expected MOP listing (advisory, via `DPRINT` in `--dev` mode): `enable_fp32_dest_acc`, `disable_fp32_dest_acc`, `enable_fp32_dest_acc` at element starts (in addition to per-element reconfig and init calls).
- A chain run twice — once with kernel built with `FP32_DEST_ACC_EN` defined, once without. The default `EnableFp32DestAcc=false` chain produces identical numerical results in both builds (confirms D6's no-op default).

If the existing 401-suite already has a row with `fp32_dest_acc=True` parametrize (per audit hint), confirm it still passes with the per-element flag default `false` — this proves D6 has not silently downgraded any existing case.

### D7 block-mode coverage gate (commit 3)

At minimum one row in the 401-suite must exercise:
- `multi_chain.cpp` row mixing streaming + block elements (e.g. `CopyTile + BlockBinaryFpu + BlockPackTile`) — verifies the fold dispatches across element kinds uniformly.
- `binary_block.cpp` 401-row already exists (`test_eltwise.py::test_binary_block`) — verify it stays green with the deleted `OldCb*` and the chain-driven `_with_dt` form.
- A 401-row exercising `BlockPackTile<…, PackTileReconfig::OutputConditional>` after a prior pack with a different CB — expected MOP listing (advisory): two-arg `pack_reconfig_data_format(prev_pack, Cb)` fires, single-arg form fires for first-element / first-pack cases.

### D8 BIG-init-boundary gate (U2 + U6)

- **U2 commit body** includes the grep evidence (no `_init_common`, no `mm_init`, no `reduce_init` in `eltwise_chain.{hpp,inl}` or `eltwise_block.hpp`).
- **U6 doxygen** documents the boundary table + worked anti-example.
- **No CI-machinery** is added in this run; the grep is a manual one-liner the reviewer / future contributor runs ad-hoc.

### End-of-pipeline gate
Full ~74 migrated-kernel pytest set under `tests/ttnn/unit_tests/operations/eltwise/` + every moreh kernel test touching a chain-using kernel, run with both `fp32_dest_acc=True` and `False` parametrize.

---

## Section E — Out of scope (deferred); also Directive-4 outcomes

### Directive-4 outcome summary

Per the per-candidate evidence in U1 above, the run-7 helper retains far more surface than the original audit's Appendix B suggested. The only items deleted are:

1. `EltwiseChainOptions` struct + both fields (zero callers, zero raw-LLK pattern). The `enable_fp32_dest_acc` field's intent migrates to D6's per-element flag.
2. `OldCb*` template params on streaming elements (bookkeeping replaced by Directive 2's compile-time prev-CB fold).
3. **D7** `OldCb*` template params on Block elements (bookkeeping replaced by chain-derived prev-CB fold extension to block path; previously deferred in v3, now landing in commit 3).

Items kept (un-migrated callers exist or pattern is alive in raw-LLK convention):

| Construct | Future migration target |
|---|---|
| `BinaryFpuOutputPolicy::HoistAcquireRelease` | `eltwise_binary_no_bcast.cpp` raw acquire/release pattern |
| `PackTileIndexMode::{BlockIter, Pinned, Absolute}` | `rotary_embedding_llama*`, `all_reduce_create_qkv_heads/reduction.cpp`, `topk.cpp` raw `pack_tile(j, …, j)` / `pack_tile(idx, …, abs)` patterns |
| `WaitUpfrontPopAtEnd` / `UpfrontReservePushAtEnd` | `experimental/conv3d/.../compute.cpp` upfront wait/pop pattern |
| `OutputConditional` (`PackTileReconfig`) | `ssm/*`, `attn_matmul/*`, `rotary_embedding/*` two-arg `pack_reconfig_data_format(old, new)` callers — semantics now expressed by chain-driven two-arg dispatch (D7) |
| `FillBitcast` | `randn/.../compute_standard_normal.cpp`, host-emitted `fill_tile_bitcast` |
| `FillInt<DF, Slot>` | `ternary_addcmul_int_sfpu_bcast.cpp`, `binary_ng/.../FILL_LLK = fill_tile_int<...>` macro injection |

### Truly out of scope for this run

| ID            | Why deferred                                                                                                              |
|---------------|---------------------------------------------------------------------------------------------------------------------------|
| F-UX-9        | Family-split SFPU collapse — entangled with test suite. Future pass.                                                      |
| F-UX-13/18    | Default reconfig vs convenience wrapper inconsistency — effectively dead after commits 2+3 (compile-time tracker handles it). Document only via U6. |
| F-UX-14       | `DEST_TO_SRCA` SHOUTING_CASE rename — cosmetic.                                                                            |
| F-UX-15       | `BroadcastDim` constexpr switch — theoretical risk; `ckernel::BroadcastType` is stable.                                   |
| F-UX-17       | `UnaryBcast` 3-caller audit — functionality-deletion question, out of UX/perf scope.                                      |
| F-UX-19       | `BinarySfpu` enum unification — 3 kernel rewrites + test updates.                                                          |
| F-UX-20       | Local op-struct dedup — needs ~5 new helper structs each with their own test.                                              |
| F-UX-21       | Defensive streaming dup-CB predicate — no production failure pattern.                                                     |
| F-PERF-5      | `count > 1` gate narrowing — only after commits 2+3 land.                                                                 |
| F-PERF-6      | Combined `reconfig_data_format(CbA, CbB)` — folded into commits 2+3 if convenient (the prev-CB tracker can emit the combined form when both sides change at the same element). |
| F-PERF-7      | TU bloat coupled to F-UX-9.                                                                                               |
| F-PERF-8      | Mixed-upfront chain dispatch — no production evidence.                                                                    |
| Audit §3 test gaps | Closed by U5 for `OptionalChainElement`; rest out of scope per user.                                                |

**Items that were in v3's deferred list and are now in scope (this run):**
- **D7 block-mode prev-CB fold** — ships as commit 3 (this v4 directive).
- **D7 block-mode D6 fp32-dest-acc transitions** — ships as part of commit 3 (the fold extension covers the prev-fp32 axis uniformly with the prev-CB axis).

---

## Section F — Resolved and newly surfaced design questions

### Q1 — `RandTile` delete vs fix-and-keep
**Resolved (already landed).** Commit `16f0b759c93` made `init()` static via NTTP seed. User-locked decision per Directive header.

### Q2 — `logit_kernel.cpp` under U4's deduced wrapper or kept on raw 2-typedef?
**Resolved: skip.** Logit is multi-stage (different PACK output CB per stage). The deduced wrapper would emit `compute_kernel_hw_startup` once with stage-1 CBs; stage 2's PACK then targets the wrong CB. Keep `logit_kernel.cpp` on the explicit `compute_kernel_hw_startup` per stage pattern. U5 adopts `OptionalChainElement` inside the existing two-stage shape, leaving the boot pattern intact.

Rationale: the multi-stage exception is documented in U6's doxygen block; the single-stage majority gets the wrapper benefit; logit pays nothing extra (10 LOC kept explicit is cheaper than a `static bool initialised` per-stage idempotency guard).

### Q3 — `where_tss_kernel.cpp` adoption shape
**Resolved (per Directive 1): adopt with `constexpr bool` flag + `OptionalChainElement<COND, FillInt/FillScalar>`** in U5. The `#ifdef`-gated chain elements collapse to a single chain definition where one of two `OptionalChainElement` pairs is live (the other is the no-op tag-only specialization). After this rewrite where_tss is single-stage, so it ALSO migrates to U4's `eltwise_chain_with_init` — collapsed in the same U5 commit. Net LOC: ~10 saved per kernel.

### Q4 — `BinaryFpu` `AIndex/BIndex` collapse to single `Index`?
**Resolved per Directive 4 grep evidence: KEEP per-side split.**

- `deepseek_grouped_gate.cpp:55-56` — AIndex=BlockIter, BIndex=FirstTile.
- `moreh_adam.cpp:44-45` — `IdxA == 0 ? FirstTile : Pinned, IdxB == 0 ? FirstTile : Pinned` (independent per-side).
- `moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp:34-35` — same conditional independent per-side pattern.

`APolicy/BPolicy` follows the same conclusion — migrated kernels routinely pass mismatched policies (pre-waited A + per-tile B). Keep the split.

### Q5 — Ship a `OptionalChainElement` unit test in U5?
**Resolved: yes.** Ship `optional_element.cpp` test kernel + `test_optional_chain_element` parametrize. Justifications:

1. The helper has zero coverage today (`eltwise_optional.hpp` is uncovered — the conditional-ladder at lines 57-65 has 7 branches none of which are exercised by any 401-suite test).
2. The unit test grounds the abstraction. `where_tss` and `logit` exercise only 2 of the 7 inner-tag specializations.
3. Future inner-kind additions (e.g. a new chain element type) will fail loudly only if the test covers each tag explicitly.

Test kernel shape: `CopyTile<…> + OptionalChainElement<COND, Inner> + PackTile<…>` chain over `(COND ∈ {True, False}) × (Inner ∈ {Mask, Clamp, Eqz})` — three Inner kinds covering `DestOnlyTag`-via-CRTP, fill-via-`FillTileTag`, and predicate-SFPU. Reference output: golden chain without the OptionalChainElement (when COND=true) / golden chain with the element removed (when COND=false).

### Q6 (newly surfaced — D6) — What is the default for `EnableFp32DestAcc`?

**Resolved: default `false` for run7 compatibility; transparent migration via macro is a recommended *non-default* pattern.** **v5 note:** the question is moot for SKIP-list elements (`CopyTile`, `BlockCopyTile`, `FillScalar`, `RandTile`, `FillInt`, `FillBitcast`, `OptionalChainElement<COND, Inner>`) — they have no `EnableFp32DestAcc` template param to default. The fold's SFINAE probe (Q16) returns the running `prev` for them — they pass through transparently and require no chain-level default.

Rationale + alternatives considered (applies to CARRY-list elements only):

| Option | Behavior | Verdict |
|---|---|---|
| (A) **Default `false` everywhere on CARRY** (chosen) | Every existing CARRY-element call site compiles unchanged. Per-element fp32 must be opted in explicitly. | **Chosen.** Zero call-site churn. Run7-compat. |
| (B) Default derived from `FP32_DEST_ACC_EN` macro at element instantiation | `static constexpr bool EnableFp32DestAcc = bool(FP32_DEST_ACC_EN)` — every CARRY element auto-enables when the kernel is built with fp32. | Rejected as default. Kernels currently use `#ifdef FP32_DEST_ACC_EN` blocks for DST sizing / scalar packing — making elements auto-enable would mid-chain-toggle fp32 mode for kernels that have always treated fp32 as a build-time global. Risk: silent behavior change in 15+ migrated moreh kernels. |
| (C) Require explicit opt-in per element with no default | Every CARRY element must spell `EnableFp32DestAcc=false` or `=true`. | Rejected — every existing call site breaks. |

**The recommended pattern for kernels wanting "global fp32 chain":**

```cpp
constexpr bool kFp32 =
#ifdef FP32_DEST_ACC_EN
    true;
#else
    false;
#endif
// then per element: BinaryFpu<..., kFp32>{}, CopyTile<..., kFp32>{}, ...
```

This is option (B) at user-discretion. Doxygen in U6 documents this idiom.

**Element 0 transition behavior:** `prev_fp32_dest_acc_for_idx<0, Es...>()` returns the chain's "before-element-0" assumption — chosen as `false`. Caveat: if the kernel called `enable_fp32_dest_acc()` before chain entry and element 0 has default `false`, the chain emits `disable_fp32_dest_acc()` before element 0 — silently downgrading the kernel's pre-chain state. **Mitigation:** doxygen explicitly warns that the chain inherits NO fp32 state from the kernel; use the per-element flag uniformly, do not rely on pre-chain `enable_fp32_dest_acc()` to "leak" into the chain.

### Q7 (newly surfaced — D2 + D6) — Does the prev-CB fold model fp32-dest-acc transitions?

**Resolved: yes — symmetric fold, same compile-time walk, separate logical axis.** Run8 has no per-element fp32 flag (`eltwise_helper_run7_vs_run8.md:114-144`); run7's D6 is ahead on this axis. `prev_fp32_dest_acc_for_idx<I, Es...>()` walks `Es[0..I-1]` backwards, returning the most recent `EnableFp32DestAcc` value or chain default (`false` per Q6) if no prior element.

The CB-side and fp32-side folds are independent — same element pack, different axes. Per-element dispatch order:
1. fp32 transition if `prev_fp32 != curr_fp32` (D6) — emitted FIRST.
2. CB reconfig if `prev_cb != curr_cb` for srca/srcb/pack (D2 + D7).
3. Per-element `init()` (existing).
4. Per-element `exec()` (existing).

Rationale for fp32-first: `enable_fp32_dest_acc()` reprograms math/pack state; subsequent `reconfig_data_format_*` must observe the new mode (`compute_kernel_hw_startup.h:96-101` — calls `llk_math_set_fp32_dest_acc` and `llk_pack_set_fp32_dest_acc`).

### Q8 (newly surfaced) — Should `eltwise_chain_with_init` static-assert single-stage?

**No.** The wrapper's contract is "single-stage kernel"; multi-stage would compile against the same shape (one trailing PackTile). Multi-stage detection lives at the kernel-author level. Document via U6's doxygen + the `// Use this only for single-stage kernels` comment block at the wrapper definition.

### Q9 (newly surfaced — D5) — How does the U2 sweep verify "no mid-MAIN hw_startup"?

**Resolved: explicit checklist in commit body + reviewer pass.**

Verification plan:

1. The U2 commit body lists every modified kernel + its row classification (rows 1-6 of the D5 placement table). The implementer fills this in by hand during the sweep.
2. Reviewer's grep: `for f in <kernel list>; do grep -A3 'void MAIN' "$f" | head -5; done` — visual confirmation that `compute_kernel_hw_startup(...)` is the first non-decl statement OR is absent.
3. **Negative grep:** `grep -B1 'compute_kernel_hw_startup(' <kernel list>` — any kernel with a non-MAIN-prefix line preceding the call is flagged. Lines preceding the call may only be variable declarations (e.g. `uint32_t cb_in0 = get_compile_time_arg_val(0);`), namespace usings, or comments.
4. The 401-suite + migrated-kernel pytest cluster catches runtime regressions if the static checks miss anything.

This is intentionally light tooling. A grep-based checker is brittle; the human checklist + reviewer pass is the authoritative gate.

### Q10 (newly surfaced — D6 + DST sizing) — How does the chain coordinate per-element fp32 with kernel-level DST_ACCUM_MODE?

**Resolved: `static_assert` on every CARRY-list element with `EnableFp32DestAcc=true`.** v5 narrowing: SKIP-list elements have no `EnableFp32DestAcc` template parameter, so the `static_assert` mechanically cannot fire on them — that's the desired behaviour (SKIP elements are dest-mode-irrelevant; their compilation does not depend on `DST_ACCUM_MODE`).

```cpp
template <..., bool EnableFp32DestAcc = false>
struct BinaryFpu {
    static_assert(!EnableFp32DestAcc || DST_ACCUM_MODE,
                  "BinaryFpu<...EnableFp32DestAcc=true> requires kernel built with "
                  "FP32_DEST_ACC_EN (DST_ACCUM_MODE must be 1).");
    // ...
};
```

The same `static_assert` lives on every other CARRY-list element (`BlockBinaryFpu`, the `BinarySfpu` op-struct family, `DestReuseBinary`, `BroadcastFpu`, `UnaryBcast`, `PackTile`, `BlockPackTile`). It does NOT live on SKIP elements — they have no flag to assert against.

`DST_ACCUM_MODE` is a compile-time integer the kernel build defines as `1` when `FP32_DEST_ACC_EN` is set, `0` otherwise (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:43,45,46`). The `static_assert` rejects element-level `EnableFp32DestAcc=true` if the kernel did not opt into fp32 DST sizing — preventing the silent garbage-DST-readout failure mode noted in U3's risk table.

### Q11 (resolved — Bisect ergonomics of P1+U1-OldCb-streaming)
The directive allows splitting into `(a) tracking infrastructure` and `(b) hoist gate`. **Single commit for streaming.** Rationale: infrastructure has no consumer until the hoist gate lands; splitting creates a phantom-feature commit. D6's transition fold lives in the same body of code — splitting would create three identical fold infrastructures.

The streaming-vs-block split (commit 2 vs commit 3) is the orthogonal axis and IS split per D7's locked sequencing.

### Q12 (resolved — D7) — Should `OldCb*` be removed from Block elements?
**Yes — in this pass.** v3 deferred this to a follow-up commit. v4 reverses: D7 directive locks the block-element `OldCb*` removal as commit 3 (NEW), immediately after the streaming-element fold (commit 2). The `_with_dt` two-arg LLK forms at `eltwise_block.hpp:72,236` and the explicit `srca/srcb` reconfig pair at `eltwise_block.hpp:142-143` get re-routed through the chain-derived prev-CB fold (extension to commit 2's infrastructure). The `with_dt_tree.md` §Tile-copy lines 49-63 and §Pack-reconfig lines 23-32 semantics are preserved: the two-arg form fires when `prev_cb != current_cb` and is known; the single-arg form fires for first-element / no-prior cases (caller's hw_startup boot covers them).

### Q13 (newly surfaced — D7) — How does the prev-CB fold handle mixed streaming + block element packs?

**Resolved: symmetric treatment for `reconfig_*_cb`; SFINAE-detected for `EnableFp32DestAcc`.**

Every element (streaming and block) — CARRY and SKIP alike — exposes `static constexpr uint32_t reconfig_srca_cb / reconfig_srcb_cb / reconfig_pack_cb;` (using `NO_PREV_CB = 0xFFFFFFFFu` when an element does not touch a side). The `prev_cb_for_idx<Side, I, Es...>()` fold walks them uniformly. **The CB-side fold has no SKIP/CARRY distinction — every element exposes the three accessors.**

For `EnableFp32DestAcc` (D6, v5-narrowed), only CARRY elements expose the member. The `prev_fp32_dest_acc_for_idx<I, Es...>()` fold uses the SFINAE probe from Q16 (`fp32_or_default<E, prev>::value`) to read CARRY members and pass-through SKIP elements. This is uniform-by-design: the fold loop body is identical for all elements; the SFINAE probe absorbs the CARRY/SKIP difference at compile time.

Block-element-specific fields (e.g. `block_size`) are orthogonal — fold ignores them.

Worked walk for `CopyTile<cb_in> + BlockBinaryFpu<cb_in, cb_b, …, CbOut=cb_tmp, EnableFp32DestAcc=true> + BlockPackTile<cb_out, EnableFp32DestAcc=true>`:
- I=0 `CopyTile` (SKIP): srca=cb_in, srcb/pack=NO_PREV. fp32 fold: SFINAE probe returns `prev` (chain default `false`); no transition emit at I=0.
- I=1 `BlockBinaryFpu` (CARRY, fp32=true): srca=cb_in (matches prev → elide), srcb=cb_b (emit `reconfig_data_format_srcb(cb_b)`), pack=cb_tmp (emit `pack_reconfig_data_format(cb_tmp)`). fp32 fold: prev=`false` (from SKIP I=0), curr=`true` → emit `enable_fp32_dest_acc()`.
- I=2 `BlockPackTile` (CARRY, fp32=true): pack=cb_out (prev=cb_tmp differs → emit two-arg `pack_reconfig_data_format(cb_tmp, cb_out)`). fp32 fold: prev=`true` (from I=1), curr=`true` → no transition emit.

Test coverage: add a `multi_chain.cpp` row exercising this exact shape post-commit-3.

### Q14 (newly surfaced — D8) — What's the audit method for confirming "no BIG init left in helper"?

**Resolved: grep evidence in U2 commit body; grep gate documented in U6 doxygen; no CI machinery.** Audit one-liner: `grep -nE '_init_common|mm_init|reduce_init|compute_kernel_hw_startup' eltwise_{chain.hpp,chain.inl,block.hpp}`. Pre-U2 baseline yields five matches (one `#include`, one doxygen comment, three call sites in `EltwiseChainPipelineInit::run()`); all three call sites are deleted by D1 in U2. Active call-site count post-U2: zero. Grep gate documented in U6 doxygen as the "BIG-init invariant gate" — manual one-liner, no CI wiring (CI machinery for a 3-line invariant is over-engineering).

### Q15 (newly surfaced — D8) — Once-at-MAIN()-top symmetry for other BIG inits?

**Resolved: yes, symmetric. Document in U6.** D5 says `compute_kernel_hw_startup` is once at top of `MAIN()`. D8 generalises: every BIG init (`binary_op_init_common`, `mm_init`, `reduce_init`) is also called once at top of `MAIN()`, before any chain or raw call. Rationale: `binary_op_init_common` programs unpack/math binary state via MMIO writes (`compute_kernel_api/eltwise_binary.h`); mid-`MAIN()` carries the same hazard as `compute_kernel_hw_startup`. Multi-stage exception (D5 row 4) extends symmetrically.

Concrete example: in `eltwise_binary_no_bcast.cpp` activations branch, the kernel calls `compute_kernel_hw_startup(cb_post_lhs, cb_post_rhs, cb_out)` first (engine boot, D5 row 2), then `binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out)` (binary-state programming, required because the kernel also uses raw binary primitives). Both are caller-side per D8; order matches the documented LLK requirement (`binary_op_init_common` assumes hw_startup already ran).

### Q16 (newly surfaced — D6 v5 narrowing) — How does the fold handle elements without an `EnableFp32DestAcc` member?

**Resolved: SFINAE detection probe in `detail::`, returning a caller-supplied default for elements without the member.** Cleanest of three options considered.

| Option | Mechanism | Verdict |
|---|---|---|
| (A) **SFINAE probe (chosen)** | Primary `fp32_or_default<E, Default>::value = Default;` + partial specialisation `fp32_or_default<E, Default, std::void_t<decltype(E::EnableFp32DestAcc)>>::value = E::EnableFp32DestAcc;`. The fold passes its running `prev` as `Default` so SKIP elements pass through. See Step 3b code block. | **Chosen.** Element struct definitions stay clean (no placeholder member polluting SKIP elements). Fold logic is uniform — `fp32_or_default<E, prev>::value` is the only call site; behaviour differs by C++17 SFINAE at compile time. Zero runtime cost. C++17 `std::void_t` is already used in trait predicates (`eltwise_chain.inl:583-657`), confirming the project's C++ standard supports this idiom. |
| (B) Element-side placeholder member | Every SKIP element carries a placeholder `static constexpr bool EnableFp32DestAcc = false;`; fold reads uniformly. | Rejected. Pollutes SKIP structs with a member they do not semantically own; the placeholder lies (`@tparam` doxygen would mis-imply opt-in); chain default repeated everywhere → fragile if it changes. |
| (C) C++20 `requires` member-presence test | `if constexpr (requires { E::EnableFp32DestAcc; })` per iteration. | Rejected for portability. Compute-kernel build path is C++17 (`-std=c++17` per `tt_metal/jit_build/build.cpp`); C++20 `requires` unavailable. SFINAE is the C++17 idiom. |

**Element-side authoring contract:** CARRY-list elements declare `EnableFp32DestAcc` as a template parameter AND a public `static constexpr bool EnableFp32DestAcc = …;` member mirroring it (so the SFINAE probe finds it). SKIP-list elements declare neither.

**Verification:** SFINAE detection is exercised by every chain mixing CARRY and SKIP elements. The `multi_chain.cpp` row added in Q13's worked walk is sufficient — `CopyTile` (SKIP) feeding `BlockBinaryFpu` (CARRY) feeding `BlockPackTile` (CARRY). A negative test row (purely-SKIP `CopyTile + FillScalar`) confirms the chain-default `false` flows through with no transitions emitted.

---

## Final ordered commit list

| # | Subject (suggested)                                                                       | Scope                                                                                              |
|---|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| 1 | `eltwise v2: drop eltwise_pipeline_init; caller owns compute_kernel_hw_startup`           | F-UX-1 (partial), F-UX-12 (relocated), F-UX-16, **D5** placement contract, **D8** BIG-init audit (grep evidence in body) — delete helper, sweep 26 prod + 13 test kernels with top-of-`MAIN()` placement enforcement. |
| 2 | `eltwise v2: compile-time prev-CB tracking + init hoist (streaming elements)`             | F-UX-7 (streaming decls only), F-PERF-1 follow-up, **D6 transition fold** — new `prev_cb_for_idx` + `prev_fp32_dest_acc_for_idx` folds; drop `OldCb*` from streaming element decls. |
| 3 | `eltwise v2: compile-time prev-CB tracking on block elements; drop OldCb* from BlockCopyTile/BlockBinaryFpu/BlockPackTile` | **D7** — extend fold to block path; reroute `_with_dt` two-arg form via chain-derived prev-CB; drop block-element `OldCb*` template params + ~12 block-element call-site rewrites. |
| 4 | `eltwise v2: drop OldCb padding zeros (streaming) + EltwiseChainOptions struct (~17 kernels)` | F-UX-7 streaming residue, F-UX-3 (D6 chain options delete) — kernel-side sweep; only the four trailing zeros at streaming call sites + `EltwiseChainOptions` struct + NTTP. |
| 5 | `eltwise v2: collapse BinaryFpu to 9 effective template params; reorder for callsite ergonomics; add per-element EnableFp32DestAcc` | F-UX-2, F-UX-5, **D6 per-element flag** — promote `CbOut` to position 3, demote `OutPolicy` to trailing default, add `EnableFp32DestAcc` trailing default (streaming + block); ~17 kernel rewrites. |
| 6 | `eltwise v2: add eltwise_chain_with_init deduced wrapper; sweep callers`                   | F-UX-1 (final), F-UX-16 — collapse boot+chain to one call; ≤25 kernels swept.                       |
| 7 | `eltwise v2: adopt OptionalChainElement in logit + where_tss; ship test kernel`            | F-UX-8 — adopt in 2 production kernels; new test kernel + parametrize.                              |
| 8 | `eltwise v2: doxygen + caller-init contract spec on chain helper headers`                  | Directive 3 — header-level doxygen, per-element annotations, **D5 placement table**, **D6 per-element fp32 notes**, **D7 block-fold note**, **D8 caller-init contract table + worked anti-example**, examples. |

Per-commit regression bar: 401-suite + ALL ~74 migrated-kernel pytests covering the touched-kernel cluster, per Section D table.

Open after this run: F-UX-9 family-split, F-UX-19 BinarySfpu enum unification, F-UX-20 local-op dedup, F-PERF-5 gate narrowing — each tracked in Section E. **Block-mode prev-CB fold** and **block-mode D6 fp32-dest-acc transitions** were deferred in v3 and now ship in commit 3 of v4.
