# `eltwise_chain` (run7) — Refinement Design (v3, revised)

## Section A — Scope

**Branch:** `astancov/eltwise_run7_refined` — hard-reset to `75868c9eff4` (run7 baseline). Two off-script implementation commits and two prior design commits were discarded. v3 builds on v2 (683-line design at blob `4dcf7ee7b15`) by amending it with **D5** (`compute_kernel_hw_startup` placement contract) and **D6** (`enable_fp32_dest_acc` per-element, not chain-level).

**Already landed (out of this design's scope as fixes — but the *shape* of one of them is being reworked):**
- `16f0b759c93` — F-UX-12 fix-in-place (`first_cb_b` walk inside `EltwiseChainPipelineInit::run()`) and F-UX-11 (`RandTile` static-init via NTTP seed).
- `ac595549b36` — F-PERF-1+2+3+4 (per-tile init gate on `chain_has_non_copy_tile_fpu_clash_v`, `pack_reconfig` hoist, strip pack-reconfig from `BlockBinaryFpu::init()` and `BinaryFpu::init()`).

These commits are not on `astancov/eltwise_run7_refined` HEAD (which is `75868c9eff4`); the design carries them forward as functionality the implementer must reproduce when landing the U2 / P1 commits below. Numbering kept for trace continuity with v2.

**Directive 1 reverses the F-UX-12 fix-in-place direction:** the helper does not own the boot anymore. `eltwise_pipeline_init` is deleted entirely; callers issue `compute_kernel_hw_startup` directly. The `first_cb_b` walk added in `16f0b759c93` is removed with it. F-UX-11's RandTile NTTP-seed fix stays as-is.

**Inputs:** `agents/refinement_audit.md` (984 lines), `eltwise_helper_run7_vs_run8.md`, `eltwise_chain.{hpp,inl}`, `with_dt_tree.md`, `eltwise_optional.hpp`.

**Scope axes:** UX/API + perf only. Functionality gaps and test gaps remain out of scope.

**Regression bar:** the 401-test validation suite at `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` PLUS every migrated-kernel pytest covering the touched-kernel cluster — every commit, every step. The verifier runs both.

### Active directive stack (D1–D6)

| ID | Summary | Lands in commit |
|----|---------|------------------|
| **D1** | Delete `eltwise_pipeline_init`. Caller writes `compute_kernel_hw_startup(...)` themselves in `MAIN()`. Per-element bootstrap (`binary_op_init_common`, etc.) stays inside the chain. | U2 |
| **D2** | Pipeline-internal compile-time prev-CB tracking. Drop `OldCb*` from streaming elements; `prev_cb_for_idx` fold over the chain element pack at compile time; per-element reconfig becomes `if constexpr (current_cb != prev_cb)`. | P1+U1-OldCb |
| **D3** | Doxygen on helper headers + caller-init contract spec + minimal example per chain shape. Header doxygen carries: caller's init contract, lifecycle expectations, table mapping chain shape → required pre-chain init, 2-3 example kernel snippets per chain shape. | U6 |
| **D4** | "No production callers" is NOT a kill criterion — grep raw-LLK pattern in unmigrated kernels. Delete only what is provably absent codebase-wide (`EltwiseChainOptions`; streaming `OldCb*`). Keep `BinaryFpuOutputPolicy::HoistAcquireRelease`, `PackTileIndexMode::{BlockIter,Pinned,Absolute}`, `WaitUpfrontPopAtEnd`/`UpfrontReservePushAtEnd`, `OutputConditional`, `FillBitcast`, `FillInt`, `BWaitTiles`, per-side `AIndex/BIndex`, Block-element `OldCb*`. | U1, U3 |
| **D5** *(new)* | `compute_kernel_hw_startup` placement contract: caller calls it as the **first statement of `MAIN()`** if the chain shape needs it; otherwise omits it (calling it elsewhere is undefined). Doxygen encodes a chain-shape → boot-needed table; the U2 sweep enforces top-of-`MAIN()` placement and removes spurious mid-`MAIN()` calls. | U2 (sweep), U6 (doc) |
| **D6** *(new)* | `enable_fp32_dest_acc` is **per-element**, not chain-level. Each chain element type carries a `bool EnableFp32DestAcc = false` template param. Pipeline machinery composes per-element fp32-dest-acc transitions into the prev-CB fold. `EltwiseChainOptions::enable_fp32_dest_acc` (chain-level) is deleted; the field's intent is replaced by per-element control via `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` LLK toggles (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:96-120`). | U1 (delete `EltwiseChainOptions`), U3 (per-element flag added to `BinaryFpu`/`BlockBinaryFpu`), P1 (transition logic in fold), U6 (doc) |

### Refinement → audit-finding → commit-group → LOC delta

| Ref-ID | Audit IDs                            | Commit group              | Files touched                                  | LOC delta (≈)        |
|--------|--------------------------------------|---------------------------|------------------------------------------------|----------------------|
| R-1    | F-UX-1, F-UX-12 (re-do), F-UX-16, **D5** | U2 (delete pipeline_init + hw_startup placement sweep) | `eltwise_chain.{hpp,inl}`, 26 prod + 13 test kernels | +60 / -200 |
| R-2    | F-UX-7, F-PERF-1 follow-up, **D6 transition fold** | P1+U1-OldCb (fused) | `eltwise_chain.{hpp,inl}`, `eltwise_block.hpp` | +95 / -90 |
| R-3    | F-UX-7 sweep tail, **D6 `EltwiseChainOptions` delete** | U1 (genuinely-dead-only) | `eltwise_chain.hpp` (decls), 17+ binary kernels | +0 / -85 |
| R-4    | F-UX-2, F-UX-5, **D6 per-element flag** | U3 (`BinaryFpu` params)   | `eltwise_chain.{hpp,inl}`, `eltwise_block.hpp`, ~17 kernels | +35 / -100 |
| R-5    | F-UX-1 wrapper                       | U4 (`eltwise_chain_with_init`) | `eltwise_chain.hpp`, ≤25 kernels (sweep set)   | +30 / -150 |
| R-6    | F-UX-8                               | U5 (`OptionalChainElement` adoption) | `logit_kernel.cpp`, `where_tss_kernel.cpp`, new test kernel + py | +60 / -40 |
| R-7    | F-UX-1 docs, F-UX-16 docs, **D5 placement table**, **D6 per-element notes** | U6 (Doxygen + spec) | `eltwise_chain.hpp`, `eltwise_block.hpp`, key element headers | +210 / 0 |

Net LOC reclaim: ~665 removed, ~490 added. D5 adds no helper-side LOC (it's a documentation contract enforced by U2's sweep); D6 adds ~50 LOC of per-element template plumbing in U3 but recoups ~15 LOC by removing `EltwiseChainOptions` in U1 and ~30 LOC by hoisting `enable_fp32_dest_acc()` into the per-element transition path in P1 (some kernels currently keep `#ifdef FP32_DEST_ACC_EN` blocks around DST capacity / scalar packing — those are NOT eliminated by D6, since D6 only governs the helper's fp32 transitions, not the kernel's own DST sizing).

---

## Section B — Refinement plan

### Group U2 — Delete `eltwise_pipeline_init`; caller owns `compute_kernel_hw_startup` (commit 1) — D1 + **D5**

#### Commit subject
`eltwise v2: drop eltwise_pipeline_init; caller owns compute_kernel_hw_startup`

#### Audit findings addressed
- **F-UX-1** (chain typed twice) — partially addressed; the deduced wrapper in U4 finishes it.
- **F-UX-12** (binary CB pair) — addressed by relocation: the boot is now in caller hands and caller writes the correct three-CB call, so the silent-miscompile site no longer exists in the helper. The patch in `16f0b759c93` (the `first_cb_b` walk) is reverted as part of this commit. RandTile fix from the same earlier commit stays.
- **F-UX-16** (only 26/78 use pipeline_init) — addressed: there is now a single boot pattern, every kernel writes `compute_kernel_hw_startup(...)` directly, and the inconsistency disappears.
- **D5** (placement contract) — addressed: the sweep enforces top-of-`MAIN()` placement per the chain-shape table; doxygen in U6 records the rule.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — delete `eltwise_pipeline_init`/`eltwise_pipeline_init_for` declarations at lines 560-570.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — delete `EltwiseChainPipelineInit::run()` at lines 821-834, `EltwisePipelineInitDispatch` at lines 841-852, `eltwise_pipeline_init` definition at lines 849-852, the `first_cb_a`/`first_cb_b`/`first_pack_cb` finders at lines 770-799 (no other consumer), and `is_reader_pred`/`is_writer_pred`/`is_binary_pred` at lines 802-805 (no other consumer).
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

Sample rewrite (`mish_kernel.cpp`):

```cpp
// Before (cur HEAD):
void MAIN() {
    using Chain = EltwiseChain<...>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(num_tiles, ...);
}

// After:
void MAIN() {
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);   // top of MAIN — D5 placement
    eltwise_chain(num_tiles, ...);
}
```

`logit_kernel.cpp` (multi-stage) rewrite — Stage 1 at top, Stage 2 mid-`MAIN()` explicitly per the table's row 4:

```cpp
void MAIN() {
    // Stage 1 — top of MAIN — input → cb_tmp0
    compute_kernel_hw_startup(cb_input, cb_input, cb_tmp0);
    eltwise_chain(num_tiles, ...);

    // Stage 2 — cb_tmp0 → cb_output (different output CB, new pack programming)
    // EXCEPTION to D5: re-boot is required between stages with different CB triples.
    compute_kernel_hw_startup(cb_tmp0, cb_tmp0, cb_output);
    eltwise_chain(num_tiles, ...);
}
```

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

#### Acceptance criteria
1. 401-test suite green: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`.
2. Migrated-kernel cluster green: `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `data_movement/test_bcast.py`, `test_moreh_adam.py`, `test_moreh_softmax_backward.py`.
3. Build clean — no `eltwise_pipeline_init` symbol remains anywhere (`grep -rl eltwise_pipeline_init ttnn/` returns empty).
4. **D5 audit:** in every modified kernel, `grep -A5 'void MAIN'` shows either `compute_kernel_hw_startup(...)` as the first non-decl statement OR no `compute_kernel_hw_startup` at all. No mid-`MAIN()` placement allowed.

---

### Group P1+U1-OldCb — Pipeline-internal compile-time prev-CB tracking; drop `OldCb*` from API (commit 2) — D2 + **D6 transition fold**

#### Commit subject (single)
`eltwise v2: compile-time prev-CB tracking + init hoist`

#### Audit findings addressed
- **F-UX-7** — `OldCb*` template parameters dead in streaming chain elements.
- **F-PERF-1 follow-up** — the FPU-clash gate (commit `ac595549b36`) hoists to boot-time emission; this commit closes the loop by deduceing the prev-CB-for-srca/-srcb/-pack at each element's compile-time position so the boot-time emissions don't redundantly reprogram state.
- **D6 transition fold** — the same `prev_cb_for_idx` walk also tracks per-element `EnableFp32DestAcc`. When element I has `EnableFp32DestAcc != prev`, the fold emits `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:96-120`) before that element's `init()` fires. Same compile-time-elision discipline as the prev-CB tracker.

This is the missing piece run8 has and run7 doesn't (run7-vs-run8 §4 line 133): `prev_cb_for_idx` fold over the element pack. Adding it makes the per-tile gate (already landed) emit the right minimal sequence at boot. **Note re D6:** run8 has *no* fp32_dest_acc handling in its chain header (`git show astancov/eltwise_run8:ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` finds zero matches for `fp32` / `dest_acc`). Run7's per-element D6 design is therefore **ahead** of run8 on this axis — a deliberate design choice run7 makes alone.

#### Bisect ergonomics — single commit or split?

The directive allows splitting into `(a) tracking infrastructure` and `(b) hoist gate`. **Recommendation: single commit.** Rationale:
1. The infrastructure has no other consumer than the hoist gate. Split (a) would land dead code that gets exercised only when (b) lands — a bisect failure between (a) and (b) reaches the same place.
2. The hoist gate cannot exist without the infrastructure, so commit (b) alone won't compile.
3. Both are within `eltwise_chain.inl`, single-file diff, no kernel sweep — the size is bounded.
4. **D6's transition fold** lives in the same body of code — splitting into (a)/(b) would create a third dimension of split.

If the implementer finds the diff unwieldy in review, splitting after-the-fact is cheap. **Default: single commit.**

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — delete `OldCb` / `OldCbA` / `OldCbB` / `OldCbOut` template parameters from streaming elements at lines 439, 454-456, 469, 479-480, 489, 498. Public API loses these params.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — delete `OldCb*` from `CopyTile` (line 78), `PackTile` (167), `PackTileBlock` (243), `BinaryFpu` (306-308), `DestReuseBinary` (429), `UnaryBcast` (503-504). None of the implementations reference them — verified by reading the lines above.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — keep the Block elements' `OldCb*` for now (`BlockCopyTile::init()` at line 72 still calls `copy_tile_to_dst_init_short_with_dt(OldCb, Cb, 0)`; `BlockPackTile::init()` at line 235 calls `pack_reconfig_data_format(OldCb, Cb)`). These have a real LLK consumer. They migrate when the prev-CB tracking is generalized to the block path; that's a follow-up commit (not this one), and gated on a separate audit because the `_with_dt` two-arg form has FP32_DEST_ACC semantics that single-arg lacks (`with_dt_tree.md` §Tile-copy lines 49-63).
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

**Step 3b — D6 transition fold.** Augment the per-element walk with a parallel scan over each element's `EnableFp32DestAcc` template flag:

```cpp
// Symmetric to prev_cb_for_idx, but tracking the boolean fp32-dest-acc flag.
// Walks Es[0..I-1] backwards; returns the most recent EnableFp32DestAcc value, or
// the chain's default (false) if no prior element set it.
template <std::size_t I, class... Es>
constexpr bool prev_fp32_dest_acc_for_idx() { /* index_sequence fold over Es */ }
```

Per-element dispatch:

```cpp
template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    constexpr bool prev_fp32 = detail::prev_fp32_dest_acc_for_idx<I, Es...>();
    constexpr bool curr_fp32 = E::EnableFp32DestAcc;
    if constexpr (curr_fp32 != prev_fp32) {
        if constexpr (curr_fp32) { enable_fp32_dest_acc(); }
        else                     { disable_fp32_dest_acc(); }
    }
    // ... then the prev-CB reconfig elision per Step 3 ...
}
```

`enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` are documented as "lightweight, standalone reconfiguration that is safe to call mid-kernel without re-running compute_kernel_hw_startup" (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:84-91`). They are the right primitive for mid-chain transitions.

The fold's start-of-chain "prev" is the chain default (`false` per Q6 below). That means: if the first element has `EnableFp32DestAcc=true`, the fold emits one `enable_fp32_dest_acc()` at chain entry. If the first element has `EnableFp32DestAcc=false`, no transition emits (the kernel either had hw_startup running with bf16 default or had previously set fp32 mode itself — but the latter is not the helper's contract). This matches caller expectations: `compute_kernel_hw_startup` does NOT touch fp32-dest-acc mode, so the chain inherits whatever the kernel set up before chain entry.

**Step 4 — Drop `OldCb*` from streaming elements' template params** (the dead bookkeeping). The four trailing literal zeros (`OldCbA, OldCbB, OldCbOut`) at ~17 kernel call sites fall off in the U1 sweep. `CbOut` stays — real semantics; U3 reorders it.

#### Why this group?
F-UX-7 (kill `OldCb*`) is bookkeeping cleanup; the bookkeeping was deleted in commit `381e193d18f` and the params have been dead since. The clean reclaim is mechanical. Bundling with the prev-CB tracking is the right call because:
1. The prev-CB infrastructure replaces what `OldCb*` was meant to thread (run-7-original-design carry-over of last-CB-on-srca per element).
2. Both touch the same per-element init bodies.
3. The D6 transition fold is structurally identical (same walk, different per-element axis); landing all three together keeps the reconfig-emission code in one logical block.
4. Bisect surface stays single (a kernel that breaks here, breaks against the new compile-time tracker — and that's where the tests want to find the regression).

#### Risk assessment

| Failure mode | Detector |
|---|---|
| Per-element `reconfig_*_cb` sentinel set wrong on some element kind | 401-suite hits each element kind. Mismatch → wrong dtype reconfig emitted at boot → 401's `test_3_*` `fp32_dest_acc=True` fails. |
| `prev_cb_for_idx` walks the wrong direction | Fold is unit-testable in isolation; trivial to verify with a static_assert in the test kernel `multi_chain.cpp`. |
| Block-mode regresses because Block elements still use `OldCb*` while streaming elements no longer have it | Block elements untouched in this commit; their `OldCb` template params remain. No interference. |
| Compile-time elision elides a needed reconfig (e.g. fp32_dest_acc-gated `_with_dt` form vs single-arg form) | Streaming chain uses single-arg `reconfig_data_format_srca(Cb)` only — there is no two-arg path to elide differently. The `_with_dt` form is block-only. |
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

### Group U1 — Genuinely-dead surface only (commit 3) — D4 + **D6 chain options delete**

#### Commit subject
`eltwise v2: drop OldCb padding zeros + EltwiseChainOptions struct (~17 kernels)`

#### Audit findings addressed
- **F-UX-7** — finishes the streaming `OldCb*` removal at call sites (the `eltwise_chain.{hpp,inl}` decl removal is in U2+P1; this commit is the kernel sweep alone).
- **F-UX-3** — deletes `EltwiseChainOptions` struct entirely (zero callers per the per-Appendix-B-candidate evidence below).

This was previously bundled with U1 in the original design. Per Directive 4, the sweep is narrowed to **only** the items provably absent codebase-wide. **D6** subsumes the v2 design's "delete `EltwiseChainOptions::enable_fp32_dest_acc`" line: per-element `EnableFp32DestAcc` on each element type replaces the chain-wide flag.

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
| `OutputConditional` (`PackTileReconfig`) | 0 callers | **KEEP** | Two-arg `pack_reconfig_data_format(old, new)` form alive in: `ssm/hc_sum_reduce/.../ssm_1d_sum_reduce.cpp:51,65`, `ssm_eltwise_mul.cpp:44,69,144`, `attn_matmul/.../transformer_attn_matmul.cpp:86`, `group_attn_matmul/.../transformer_group_attn_matmul.cpp:163`, `rotary_embedding/.../rotary_embedding.cpp:101,129`, `rotary_embedding_single_tile.cpp:76`. Two-arg form has FP32_DEST_ACC-gated semantics single-arg lacks (`with_dt_tree.md:23-32`). 8+ kernels. |
| `FillBitcast` | 0 callers | **KEEP** | `randn/.../compute_standard_normal.cpp:42` — `fill_tile_bitcast(2, two_pi)`. `eltwise/unary/common/unary_op_utils.cpp:150` — host emits `fill_tile_bitcast({}, {:#x}u);`. `reduction/accumulation/.../accumulation_program_factory.cpp:120` — `defines_kernel_args["FILL_TILE"] = "fill_tile_bitcast"`. |
| `FillInt<DF, Slot>` | 2 callers | **KEEP** | `ternary_addcmul_int_sfpu_bcast.cpp:62`, `binary_ng/.../binary_ng_program_factory.cpp:776`. Already used. |
| `OldCb*` on streaming elements | 0 callers, dead since `381e193d18f` | **DELETE — bookkeeping, replaced by Directive 2** | Per Directive 4 carve-out: `OldCb*` is bookkeeping (not a feature). The bookkeeping was deleted in `381e193d18f`. The compile-time prev-CB fold replaces it. Four trailing zeros fall off ~17 kernels' templates. |
| `OldCb*` on **Block** elements | 0 chain-passed, internally used | **KEEP** | `eltwise_block.hpp:72` calls `copy_tile_to_dst_init_short_with_dt(OldCb, Cb, 0)`; `eltwise_block.hpp:235` calls `pack_reconfig_data_format(OldCb, Cb)`. Two-arg form has FP32_DEST_ACC semantics single-arg lacks. Future commit extends prev-CB fold to block path. |
| `chain_has_non_copy_tile_fpu_clash_v` | already wired | **KEPT** | Wired by `ac595549b36` as `clash_gate`. |
| `chain_is_hoist_safe_v`, `chain_loads_share_cb_v` | computed-not-used | **KEEP** | Defensive predicates for F-PERF-5 (gate narrowing) and F-UX-21 (streaming dup-cb). |
| `BWaitTiles` (BlockBinaryFpu) | internal | **KEEP** | `eltwise_block.hpp:118` handles `BIndex == FirstTile` scalar-bcast wait shape (B = single scalar, A = block). Required for `eltwise_binary_scalar.cpp` correctness. |

**Summary of Appendix B re-evaluation:**

| Provably absent (DELETE) | Pattern alive in unmigrated kernels (KEEP — future migration targets) |
|---|---|
| `EltwiseChainOptions` struct + both fields (replaced by D6 per-element + auto-block) | `BinaryFpuOutputPolicy::HoistAcquireRelease` (`eltwise_binary_no_bcast.cpp` raw pattern) |
| `OldCb*` on streaming elements (bookkeeping) | `PackTileIndexMode::{BlockIter, Absolute, Pinned}` (rotary_embedding_llama, all_reduce_create_qkv_heads, moreh_adam) |
|  | `WaitUpfrontPopAtEnd` / `UpfrontReservePushAtEnd` (conv3d) |
|  | `OutputConditional` (ssm, attn_matmul, rotary_embedding) |
|  | `FillBitcast` (randn, unary_op_utils) |
|  | `FillInt` (ternary_addcmul_int, binary_ng_program_factory) |

#### Files to touch (U1)
~17 binary kernels: drop the four trailing zeros (`OldCbA, OldCbB, OldCbOut`) from `BinaryFpu<…>` and `BlockBinaryFpu<…>` template lists. The `CbOut` immediately following is **not** removed — U3 reorders it.

Plus `eltwise_chain.hpp:304-312` (`EltwiseChainOptions` struct) and `eltwise_chain.hpp:579` (the NTTP on `eltwise_chain` declaration) and `eltwise_chain.inl:821,835-839` (the NTTP on the function template definition + the `Opts.upfront_block_size` references).

Sweep set: `data_movement/bcast/.../{bcast_h,hw,w}.cpp` (3), `eltwise/binary/.../{bcast_h,hw,w}.cpp` (3), `eltwise/binary_ng/.../{eltwise_binary_no_bcast,eltwise_binary_scalar,eltwise_binary_sfpu_no_bcast,eltwise_binary_sfpu_scalar,eltwise_where_no_bcast,eltwise_where_sfpu,eltwise_where_sfpu_scalar}.cpp` (7), `eltwise/binary_ng/.../kernels_ng/{eltwise_binary_col_bcast,row_bcast,row_col_bcast,scalar_bcast,sfpu_row_bcast,where_sfpu_row_bcast}.cpp` (6), `ternary/.../ternary_addc_ops_*.cpp`, `experimental/ssm/prefix_scan/.../ssm_prefix_scan.cpp`, `experimental/reduction/deepseek_grouped_gate/...`. Plus test kernels: `binary_block,binary_fpu,binary_fpu_same_cb,binary_fpu_bcast,inplace_accumulate,multi_chain`.

#### Acceptance criteria
1. 401-test suite green.
2. `test_binary_bcast.py`, `test_binary_ng.py` green.
3. `grep -rn EltwiseChainOptions ttnn/` returns empty.

---

### Group U3 — `BinaryFpu` 15→9-effective template params (commit 4) — D4 + **D6 per-element flag**

#### Commit subject
`eltwise v2: collapse BinaryFpu to 9 effective template params; reorder for callsite ergonomics`

#### Audit findings addressed
- **F-UX-2** — `BinaryFpu` 15-param surface.
- **F-UX-5** — `BinaryFpuOutputPolicy` enum dead-default → optional.
- **D6** — adds `EnableFp32DestAcc` template param to `BinaryFpu` and `BlockBinaryFpu`.

#### Pre-conditions
- U2 shipped (callers already moved off `eltwise_pipeline_init`).
- P1+U1-OldCb shipped (`OldCb*` removed from `BinaryFpu` template list; D6 transition fold consumes the new flag).
- U1 shipped (`EltwiseChainOptions` deleted; the `enable_fp32_dest_acc` chain-level flag is gone, so per-element is the only path).

#### Directive 4 — `AIndex/BIndex` collapse vs retain (Q4)

The original design recommended retaining the per-side split. **Directive 4 forces the search.** Result:

- `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:55-56` — passes `CbIndexMode::BlockIter` for AIndex and `CbIndexMode::FirstTile` for BIndex (the `add_bias` chain — A is pre-waited block, B is per-tile scalar).
- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp:44-45` — sets per-side mode conditionally on `IdxA == 0` and `IdxB == 0` independently.
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_{h,w,c_large,h_large,w_large}.cpp:34-35` — same pattern.

**Verdict: KEEP per-side split.** Asymmetric A vs B index mode is alive in 6+ migrated kernels.

`APolicy/BPolicy` (per-side `CopyTilePolicy`) — same conclusion. Migrated kernels routinely pass mismatched policies (`A: NoWaitNoPop, B: WaitAndPop` for pre-waited A pattern). Keep split.

#### Final shape (12 named, 9 effective)

After U2 and P1+U1-OldCb the surviving param list is `{CbA, CbB, Op, Bcast, OutPolicy, DfReconfig, APolicy, BPolicy, AIndex, BIndex, DstSlot, CbOut}` = 12 params.

D6 adds one more: `EnableFp32DestAcc`. The audit's "~8" target is approximate; the U3 commit lands `9 effective + 4 rarely-overridden trailing defaults` = **13 named params**.

Reductions for U3:

**Step 1 — Move `CbOut` from position 12 to position 3.** It's the most frequently overridden param (~10 binary kernels need it).

**Step 2 — `OutPolicy` becomes default-trailing.** `BinaryFpuOutputPolicy::PerTile` is the default (currently and after); `HoistAcquireRelease` stays as a value (Directive 4: pattern alive in `eltwise_binary_no_bcast.cpp`). The param moves to the end of the list with `= PerTile` default. 99% of callers stop spelling it.

**Step 3 — D6 add `EnableFp32DestAcc`.** New trailing default. Position: end of the trailing-defaults block, AFTER `OutPolicy` and `DstSlot` so prior callsite ordering is preserved up to that point. Default value: see Q6 below — recommend `false` for run7 compatibility, with a forward path for transparent migration via `FP32_DEST_ACC_EN` macro derivation.

**Step 4 — Final shape:**

```cpp
template <
    uint32_t CbA,                                                           // 1
    uint32_t CbB,                                                           // 2
    uint32_t CbOut = 0,                                                     // 3 — promoted
    BinaryFpuOp Op = BinaryFpuOp::Add,                                      // 4
    BroadcastDim Bcast = BroadcastDim::None,                                // 5
    BinaryDataFormatReconfig DfReconfig = BinaryDataFormatReconfig::InputAndOutput,  // 6
    CopyTilePolicy APolicy = CopyTilePolicy::WaitAndPop,                    // 7
    CopyTilePolicy BPolicy = CopyTilePolicy::WaitAndPop,                    // 8
    CbIndexMode AIndex = CbIndexMode::FirstTile,                            // 9
    CbIndexMode BIndex = CbIndexMode::FirstTile,                            // 10
    Dst DstSlot = Dst::D0,                                                  // 11 — trailing default
    BinaryFpuOutputPolicy OutPolicy = BinaryFpuOutputPolicy::PerTile,       // 12 — trailing default
    bool EnableFp32DestAcc = false>                                         // 13 — D6, new trailing default
struct BinaryFpu;
```

That's 13 named params, but 99% of callers use ≤9 positionally (everything past `BIndex` is rarely overridden — `DstSlot` only in 1 kernel, `OutPolicy` only when `HoistAcquireRelease` is opted in, `EnableFp32DestAcc` only when a kernel wants to fold fp32 mode into the chain).

The audit's "~8" target is approximate. **9 effective + 4 rarely-overridden trailing defaults** is within tolerance and preserves every Directive-4-surviving feature plus D6's per-element fp32 control.

**Step 5 — `BlockBinaryFpu`** at `eltwise_block.hpp:107-122` mirrors the same reorder. `BWaitTiles` stays (real semantics per Directive 4 evidence). `OldCbA/B/Out` stay in the block element until the prev-CB fold is extended there (out of scope). `EnableFp32DestAcc` added as trailing default in the same position as `BinaryFpu`.

**Step 6 — D6 on every other streaming element.** Each chain element type carries `EnableFp32DestAcc` as a trailing default:

| Element            | Position of `EnableFp32DestAcc` |
|--------------------|---------------------------------|
| `CopyTile`         | trailing (after current `Reconfig` param) |
| `BinaryFpu`        | trailing (Step 4 above) |
| `DestReuseBinary`  | trailing (after `IndexMode`) |
| `UnaryBcast`       | trailing (after `Reconfig`) |
| `PackTile`         | trailing (after `Reconfig`) |
| `PackTileBlock`    | trailing (after `Reconfig`) |
| `BlockCopyTile`    | trailing (after current params) |
| `BlockBinaryFpu`   | trailing (Step 5 above) |
| `BlockPackTile`    | trailing (after current params) |
| `FillScalar`, `FillInt`, `FillBitcast`, `RandTile` | trailing (after `DstSlot`) |
| SFPU op-structs (CRTP base `UnaryOp`/`BinaryOp`/`TernaryOp`/`QuaternaryOp`) | inherited via the base; default `false` |

Each element exposes `static constexpr bool EnableFp32DestAcc = ...;` so the P1 fold's `prev_fp32_dest_acc_for_idx<I, Es...>()` walk can read it uniformly.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — `BinaryFpu` decl at lines 442-458 → new 13-param shape; same shape on every other streaming element (`CopyTile` at 432-440, `DestReuseBinary` at 460-470, `UnaryBcast` at 472-481, `PackTile` at 483-490, `PackTileBlock` at 492-499, `FillScalar`/`FillInt`/`FillBitcast`/`RandTile` at 502-509).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — `BinaryFpu` impl at line 295-416 (template-arg-list head only; body untouched). Other elements: same surgical head-only edit. CRTP bases (`UnaryOp`/`BinaryOp`/`TernaryOp`/`QuaternaryOp` at lines 314-329 of `eltwise_chain.hpp`) get a `static constexpr bool EnableFp32DestAcc = false;` declaration that derived structs can override.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — `BlockBinaryFpu` decl at 107-122; `BlockCopyTile` and `BlockPackTile` at their respective decls.
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

### Group U4 — `eltwise_chain_with_init` deduced wrapper (commit 5)

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

```cpp
/// Deduces the chain element pack at compile time and emits
/// `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` as the first statement of the wrapper, then
/// runs `eltwise_chain(...)`. This wrapper MUST itself be the first statement of `MAIN()` per D5.
///
/// Chain shape resolution (compile-time):
///   - `cb_a` ← first reader's `cb_a_id()` (from `is_cb_reader_op_v` trait).
///   - `cb_b` ← first binary-op's `cb_b_id()` (from `is_binary_fpu_op_v ||
///     is_dest_reuse_binary_op_v`); falls back to `cb_a` if no binary element
///     (unary chain).
///   - `cb_out` ← first writer's `pack_cb_id()`.
///
/// **Use this only for single-stage kernels** — multi-stage kernels (different
/// PACK output CB per stage) must call `compute_kernel_hw_startup` themselves
/// per stage; the deduced wrapper would emit it once and the second stage's
/// PACK would target the wrong CB.
template <class... Es>
ALWI void eltwise_chain_with_init(uint32_t n_tiles, Es... elts) {
    // Chain-pack-deduced boot CBs.
    constexpr uint32_t cb_in_a = /* fold over Es using is_cb_reader_op_v */ ...;
    constexpr uint32_t cb_in_b = /* fold over Es using is_binary_fpu_op_v || is_dest_reuse_binary_op_v */ ...;
    constexpr uint32_t cb_in_2 = (cb_in_b != 0) ? cb_in_b : cb_in_a;
    constexpr uint32_t cb_out  = /* fold over Es using is_cb_writer_op_v */ ...;
    static_assert(cb_in_a != 0 && cb_out != 0,
                  "eltwise_chain_with_init: chain must have a reader and a writer");
    compute_kernel_hw_startup(cb_in_a, cb_in_2, cb_out);
    eltwise_chain(n_tiles, elts...);
}
```

The CB folds are the same compile-time walk that was inside the old `EltwiseChainPipelineInit::run()`; resurrect them inside `eltwise_chain.hpp` (pure header — keep them inline).

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

### Group U5 — `OptionalChainElement` adoption (commit 6)

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

**`logit_kernel.cpp` rewrite — `#ifdef CLAMP` collapses to compile-time bool flag and `OptionalChainElement<COND, Clamp<...>>`:**

```cpp
#ifdef CLAMP
constexpr bool DO_CLAMP = true;
#else
constexpr bool DO_CLAMP = false;
#endif

const Clamp<Dst::D0> clamp_inner = DO_CLAMP
    ? Clamp<Dst::D0>{packed_scalar1, packed_scalar2}
    : Clamp<Dst::D0>{};

void MAIN() {
    // Stage 1 — top of MAIN per D5
    compute_kernel_hw_startup(cb_input, cb_input, cb_tmp0);
    eltwise_chain(num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        OptionalChainElement<DO_CLAMP, Clamp<Dst::D0>>{clamp_inner},
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

    // Stage 2 (unconditional — no OptionalChainElement) — re-boot per D5 multi-stage exception
    compute_kernel_hw_startup(cb_tmp0, cb_tmp0, cb_output);
    eltwise_chain(num_tiles, ...);
}
```

`logit_kernel.cpp` stays on explicit `compute_kernel_hw_startup` per stage (multi-stage exclusion from U4 sweep). U5 adopts `OptionalChainElement` inside the existing shape.

**`where_tss_kernel.cpp` rewrite (per Directive 1 / Q3):**

The kernel currently has two `#ifdef`-gated halves (`INT32/UINT32` vs `FLOAT/FLOAT32`) substituting `FillInt<DataFormat::Int32, ...>` for `FillScalar<...>`. The Q3 directive: *"kernel's existing macro layer defines a `constexpr bool` flag; chain consumes it as `OptionalChainElement<COND, Inner>`."*

Mapping:

```cpp
#if defined(INP_INT32) || defined(INP_UINT32)
constexpr bool USE_INT_FILL = true;
#else
constexpr bool USE_INT_FILL = false;
#endif

void MAIN() {
    // Single chain definition — both fill kinds present, only one is live per build.
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);  // top of MAIN per D5
    eltwise_chain(num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        OptionalChainElement<USE_INT_FILL,  FillInt<DataFormat::Int32, Dst::D1>>{packed_scalar1},
        OptionalChainElement<USE_INT_FILL,  FillInt<DataFormat::Int32, Dst::D2>>{packed_scalar2},
        OptionalChainElement<!USE_INT_FILL, FillScalar<Dst::D1>>{*true_value},
        OptionalChainElement<!USE_INT_FILL, FillScalar<Dst::D2>>{*false_value},
        WhereSfpu{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
```

The `OptionalChainElement<false, …>` specialization is a tag-only no-op (`eltwise_optional.hpp:55-84`), so the dead branch costs nothing at runtime and its inner element is never instantiated. The chain pipeline sees a uniform 7-element list both ways. `#ifdef` is gone from the chain body.

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

### Group U6 — Doxygen documentation pass (commit 7) — D3 + **D5** + **D6**

#### Commit subject
`eltwise v2: doxygen + caller-init contract spec on chain helper headers`

#### Audit findings addressed
- **D3** Doc — supports F-UX-1, F-UX-16 (caller-init contract was implicit).
- **D5** placement contract table.
- **D6** per-element fp32_dest_acc usage notes.

#### Files to touch
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — top-of-file doxygen block; per-element `///` blocks on `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`, `eltwise_chain`, `eltwise_chain_with_init`.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp` — top-of-file doxygen + per-element on `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`.
- Key element headers — `eltwise_fill.hpp`, `eltwise_optional.hpp`, `eltwise_rand.hpp`, `eltwise_helpers.hpp` (aggregator).

#### Required content per file

**`eltwise_chain.hpp` — top-of-file doxygen block** with these sub-sections:

- `@section caller_init_contract` — **D5 placement table** (the table from U2 verbatim) with chain shape → required `compute_kernel_hw_startup` invocation → placement constraint.
- `@section hw_startup_placement` — **D5 explicit rule:**
  > `compute_kernel_hw_startup` must be called as the first statement of `MAIN()` if the chain shape requires it. Calling it elsewhere — after any other init, after a per-tile loop, or repeatedly within a stage — is **undefined**. The only documented exception is multi-stage kernels (e.g. `logit_kernel.cpp`) where each stage with a different CB triple re-issues `compute_kernel_hw_startup` immediately before that stage's chain call (NEVER inside a per-tile loop). LLK rationale: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:26-30` documents the MMIO writes as "almost exclusively requir[ing] the idle state of the execution units that should be configured. ... It is unsafe to call this function in the middle of a kernel execution."
- `@section per_element_fp32_dest_acc` — **D6 explicit rule:**
  > Each chain element type carries an `EnableFp32DestAcc` template parameter (default `false`). When elements with different values are chained, the pipeline's compile-time fold emits `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` (`compute_kernel_hw_startup.h:96-120`) at the transition points. The kernel must be built with `FP32_DEST_ACC_EN` defined if any element opts in (`static_assert` enforces this); the per-element flag does NOT itself resize DST. The chain inherits the kernel's pre-chain fp32-dest-acc state at element 0 (i.e. if the kernel called `enable_fp32_dest_acc()` before chain entry and element 0 has `EnableFp32DestAcc=false`, the chain emits `disable_fp32_dest_acc()` before element 0 runs). Authors who do not need fp32 transitions should leave the default `false` everywhere.
- `@section deduced_wrapper` — when to use `eltwise_chain_with_init` vs. the explicit boot.
- `@section lifecycle` — acquire/release window shape, when `init()` re-fires (gated on `chain_has_non_copy_tile_fpu_clash_v`), when `pack_reconfig` runs (boot only after P1+U1-OldCb), when **D6** `enable_fp32_dest_acc()` runs (at every per-element transition; compile-time elision when no transition needed).
- `@section examples` — three canonical kernel snippets:
  1. Unary chain (CopyTile + SFPU + Pack).
  2. Binary chain (BinaryFpu + Pack).
  3. Two-stage kernel with re-boot between stages (logit-style).
  4. **D6 mixed-fp32 chain example** — one element opting in, the next opting out.

**Per-element `///` blocks** on `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock` — `@tparam` for every parameter (including the new `EnableFp32DestAcc`); explicit "the element does NOT emit `compute_kernel_hw_startup` — caller's pre-chain init must cover its CBs" reminder; cross-link `@ref caller_init_contract` and `@ref per_element_fp32_dest_acc`.

**`eltwise_block.hpp`** — top-of-file doxygen explaining "block elements still own their `OldCb*` (the streaming-side prev-CB fold has not been generalized to block path yet)" + same D6 per-element fp32 notes; per-element `///` blocks.

**Element headers** — `eltwise_fill.hpp`, `eltwise_optional.hpp`, `eltwise_rand.hpp` — short header doxygen + per-struct `@brief` + `@tparam` lines (including any `EnableFp32DestAcc` for fill/rand structs, although these inherit from `DestOnlyTag`/`UnaryOp` and the default `false` is right for them in the common case).

**Aggregator** `eltwise_helpers.hpp` — list which element-family headers it pulls and what each ships.

#### Why ship as a separate commit?
- Pure documentation; cannot regress runtime.
- Bisect surface zero.
- Easier review (a single "doc commit" is grep-friendly).
- Captures the U2 boot-contract decision so future kernel authors don't re-discover it from code.
- Encodes D5 + D6 explicitly so future implementers don't need to reverse-engineer the placement and per-element-fp32 invariants.

#### Risk assessment
None — pure comments / doc strings.

#### Acceptance criteria
1. 401-suite green (no functional change).
2. `git diff --stat` shows only comment additions; reviewer can confirm by inspection.

---

## Section C — Sequencing & dependencies

```
U2 ──> P1+U1-OldCb ──> U1 ──> U3 ──> U4 ──> U5 ──> U6
```

### Hard sequencing rules

1. **U2 first.** Boot semantics are the foundation. Every later commit references the new caller-owns-boot contract — including U6's documentation. **D5 placement contract** is a U2 sweep concern, not a helper-side change.
2. **P1+U1-OldCb before U1.** The helper-side decl removal happens in P1+U1-OldCb (it's bundled because the prev-CB fold replaces what `OldCb*` was bookkeeping for). The kernel-side sweep (just deletes `0, 0, 0, 0` literals) is U1. If we ship U1 first, kernels' templates have a deleted param position → don't compile. **D6 transition fold** lands in P1 because the same compile-time walk handles both prev-CB and prev-fp32 transitions; splitting them would create two identical fold infrastructures.
3. **U1 before U3.** U3 reorders the surviving params and ADDS the per-element `EnableFp32DestAcc`. Reordering with deletions interleaved is one tangled diff; serializing produces two clean diffs. U1 also deletes `EltwiseChainOptions` (whose `enable_fp32_dest_acc` field motivates D6's per-element replacement) — landing the chain-wide field deletion before the per-element addition makes the migration story coherent: "chain-wide knob deleted, per-element flag added in its place."
4. **U3 before U4.** U4 sweeps every chain caller for the `eltwise_chain_with_init` rewrite. If U3 hasn't reordered first, U4 sweeps with the old shape; then U3 re-sweeps. Two passes over the same files is double the bisect surface.
5. **U5 after U4.** U5 migrates `where_tss` to a single-stage shape that uses U4's wrapper.
6. **U6 last.** Documentation reflects the final shape. **D5 placement table** and **D6 per-element fp32 notes** can only be authored after U2 (D5 sweep proven) and U3 (D6 per-element flag landed).

### Files-touched matrix (overlap risk)

| Commit | `eltwise_chain.hpp` | `eltwise_chain.inl` | `eltwise_block.hpp` | Migrated kernels | Test kernels |
|--------|---------------------|---------------------|---------------------|------------------|--------------|
| U2     | yes (delete pipeline_init decl) | yes (delete pipeline_init impl + finders) | no | 26 | 13 |
| P1+U1-OldCb | yes (decls) | yes (impl + new fold + D6 transition fold) | partial (no Block changes) | no | no |
| U1     | yes (delete `EltwiseChainOptions`) | yes (delete `Opts` NTTP) | no | ~17 (literal sweep) | ~6 |
| U3     | yes (BinaryFpu decl + per-element `EnableFp32DestAcc`) | yes (head only + CRTP base) | yes (BlockBinaryFpu + Block elements) | ~17 | ~6 |
| U4     | yes (new wrapper) | no | no | ~25 | no |
| U5     | no | no | no | 2 (logit, where_tss) + new test kernel | yes (1 new) |
| U6     | yes (doxygen) | maybe (doxygen on impl-internal types) | yes (doxygen) | no | no |

U2 and U4 both sweep largely-overlapping kernel sets — but at different abstraction layers (U2: explicit boot, U4: deduced wrapper). Serializing means each commit is bisectable on its own concern. U1 and U3 both touch `eltwise_chain.hpp` head — U1 deletes one struct (`EltwiseChainOptions`) and unbinds the NTTP; U3 modifies struct decls (add `EnableFp32DestAcc`). The two are non-overlapping at the file-region level.

---

## Section D — Acceptance test plan

### Per-commit gate

Every commit:

1. 401-test suite green: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`.
2. Migrated-kernel pytest sample for the touched-kernel cluster:

| Commit | Pytest set                                                                                          |
|--------|-----------------------------------------------------------------------------------------------------|
| U2     | `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `test_bcast.py` (data_movement), `test_moreh_adam.py`, `test_moreh_softmax_backward.py` (D5: row-5 omit verification) |
| P1+U1-OldCb | `test_binary_ng.py`, `test_binary_bcast.py`, `test_unary.py` (clash-free chains: mish, hardswish, identity, tanhshrink), **D6**: any 401-row already running with `fp32_dest_acc=True` parametrize (the existing `test_3_*` rows in the 401-suite already exercise this — verify D6 doesn't regress them) |
| U1     | `test_binary_bcast.py`, `test_binary_ng.py` |
| U3     | same as U1 + `test_moreh_softmax_backward.py`, `test_moreh_adam.py`, `test_moreh_layer_norm.py` + **D6**: new `test_binary_fpu_per_element_fp32_dest_acc` 401-suite row exercising mixed-mode chain |
| U4     | `test_unary.py`, `test_binary_bcast.py`, `test_binary_ng.py`, `test_ternary.py`, `test_bcast.py` |
| U5     | `test_unary.py::test_logit`, `test_unary.py::test_where`, `test_eltwise.py::test_optional_chain_element` |
| U6     | 401-suite only (no functional change) |

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

### D6 per-element-fp32 coverage gate (U3 / P1)

At minimum one row in the 401-suite must exercise:
- A chain with element 0 `EnableFp32DestAcc=true`, element 1 `=false`, element 2 `=true`. Expected MOP listing (advisory, via `DPRINT` in `--dev` mode): `enable_fp32_dest_acc`, `disable_fp32_dest_acc`, `enable_fp32_dest_acc` at element starts (in addition to per-element reconfig and init calls).
- A chain run twice — once with kernel built with `FP32_DEST_ACC_EN` defined, once without. The default `EnableFp32DestAcc=false` chain produces identical numerical results in both builds (confirms D6's no-op default).

If the existing 401-suite already has a row with `fp32_dest_acc=True` parametrize (per audit hint), confirm it still passes with the per-element flag default `false` — this proves D6 has not silently downgraded any existing case.

### End-of-pipeline gate
Full ~74 migrated-kernel pytest set under `tests/ttnn/unit_tests/operations/eltwise/` + every moreh kernel test touching a chain-using kernel, run with both `fp32_dest_acc=True` and `False` parametrize.

---

## Section E — Out of scope (deferred); also Directive-4 outcomes

### Directive-4 outcome summary

Per the per-candidate evidence in U1 above, the run-7 helper retains far more surface than the original audit's Appendix B suggested. The only items deleted are:

1. `EltwiseChainOptions` struct + both fields (zero callers, zero raw-LLK pattern). The `enable_fp32_dest_acc` field's intent migrates to D6's per-element flag.
2. `OldCb*` template params on streaming elements (bookkeeping replaced by Directive 2's compile-time prev-CB fold).

Items kept (un-migrated callers exist or pattern is alive in raw-LLK convention):

| Construct | Future migration target |
|---|---|
| `BinaryFpuOutputPolicy::HoistAcquireRelease` | `eltwise_binary_no_bcast.cpp` raw acquire/release pattern |
| `PackTileIndexMode::{BlockIter, Pinned, Absolute}` | `rotary_embedding_llama*`, `all_reduce_create_qkv_heads/reduction.cpp`, `topk.cpp` raw `pack_tile(j, …, j)` / `pack_tile(idx, …, abs)` patterns |
| `WaitUpfrontPopAtEnd` / `UpfrontReservePushAtEnd` | `experimental/conv3d/.../compute.cpp` upfront wait/pop pattern |
| `OutputConditional` (`PackTileReconfig`) | `ssm/*`, `attn_matmul/*`, `rotary_embedding/*` two-arg `pack_reconfig_data_format(old, new)` callers |
| `FillBitcast` | `randn/.../compute_standard_normal.cpp`, host-emitted `fill_tile_bitcast` |
| `FillInt<DF, Slot>` | `ternary_addcmul_int_sfpu_bcast.cpp`, `binary_ng/.../FILL_LLK = fill_tile_int<...>` macro injection |
| `OldCb*` on Block elements | block-mode prev-CB fold extension (future commit) |

### Truly out of scope for this run

| ID            | Why deferred                                                                                                              |
|---------------|---------------------------------------------------------------------------------------------------------------------------|
| F-UX-9        | Family-split SFPU collapse — entangled with test suite. Future pass.                                                      |
| F-UX-13/18    | Default reconfig vs convenience wrapper inconsistency — effectively dead after P1+U1-OldCb (compile-time tracker handles it). Document only via U6. |
| F-UX-14       | `DEST_TO_SRCA` SHOUTING_CASE rename — cosmetic.                                                                            |
| F-UX-15       | `BroadcastDim` constexpr switch — theoretical risk; `ckernel::BroadcastType` is stable.                                   |
| F-UX-17       | `UnaryBcast` 3-caller audit — functionality-deletion question, out of UX/perf scope.                                      |
| F-UX-19       | `BinarySfpu` enum unification — 3 kernel rewrites + test updates.                                                          |
| F-UX-20       | Local op-struct dedup — needs ~5 new helper structs each with their own test.                                              |
| F-UX-21       | Defensive streaming dup-CB predicate — no production failure pattern.                                                     |
| F-PERF-5      | `count > 1` gate narrowing — only after P1+U1-OldCb lands.                                                                 |
| F-PERF-6      | Combined `reconfig_data_format(CbA, CbB)` — folded into P1+U1-OldCb if convenient (the prev-CB tracker can emit the combined form when both sides change at the same element). |
| F-PERF-7      | TU bloat coupled to F-UX-9.                                                                                               |
| F-PERF-8      | Mixed-upfront chain dispatch — no production evidence.                                                                    |
| Block-mode prev-CB fold | Generalize Directive 2 to `eltwise_block.hpp` Block elements. Out of scope; future commit. |
| Block-mode D6 fp32-dest-acc transitions | Generalize D6 transition fold to block path. The block path's per-iter init shape interacts with per-iter mode toggles non-trivially. Future commit. |
| Audit §3 test gaps | Closed by U5 for `OptionalChainElement`; rest out of scope per user.                                                |

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

**Resolved: default `false` for run7 compatibility; transparent migration via macro is a recommended *non-default* pattern.**

Rationale + alternatives considered:

| Option | Behavior | Verdict |
|---|---|---|
| (A) **Default `false` everywhere** (chosen) | Every existing call site compiles unchanged. Per-element fp32 must be opted in explicitly. | **Chosen.** Zero call-site churn. Run7-compat. |
| (B) Default derived from `FP32_DEST_ACC_EN` macro at element instantiation | `static constexpr bool EnableFp32DestAcc = bool(FP32_DEST_ACC_EN)` — every element auto-enables when the kernel is built with fp32. | Rejected as default. Kernels currently use `#ifdef FP32_DEST_ACC_EN` blocks for DST sizing / scalar packing — making elements auto-enable would mid-chain-toggle fp32 mode for kernels that have always treated fp32 as a build-time global. Risk: silent behavior change in 15+ migrated moreh kernels. |
| (C) Require explicit opt-in per element with no default | Every element must spell `EnableFp32DestAcc=false` or `=true`. | Rejected — every existing call site breaks. |

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

**Resolved: yes — symmetric fold, same compile-time walk, separate logical axis.**

Run8 §4 (per `eltwise_helper_run7_vs_run8.md:114-144`) does NOT model fp32-dest-acc transitions — run8 has no per-element fp32 flag at all (`git show astancov/eltwise_run8:eltwise_chain.hpp` returns zero matches for `fp32`/`dest_acc`). Run7's D6 design is *ahead* of run8 on this axis.

The fold structure:

```cpp
template <std::size_t I, class... Es>
constexpr bool prev_fp32_dest_acc_for_idx() {
    // Walk Es[0..I-1] backwards; return the most recent `EnableFp32DestAcc` value.
    // If no prior element, return chain default (false per Q6).
    if constexpr (I == 0) return false;
    // ... constexpr fold from I-1 down to 0 ...
}
```

The CB-side and fp32-side folds are independent — they walk the same element pack but project different axes. At each element, the dispatch emits:
1. Per-element fp32 transition if `prev_fp32 != curr_fp32` (D6).
2. Per-side CB reconfig if `prev_cb != curr_cb` for srca/srcb/pack (D2).
3. Per-element `init()` (existing).
4. Per-element `exec()` (existing).

Order is significant: fp32 transitions go FIRST, before reconfigs and `init()`. Rationale: `enable_fp32_dest_acc()` reprograms math/pack state; subsequent `reconfig_data_format_*` calls must observe the new mode. (LLK reference: `compute_kernel_hw_startup.h:96-101` — `enable_fp32_dest_acc()` calls `llk_math_set_fp32_dest_acc(true)` and `llk_pack_set_fp32_dest_acc(true)`.)

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

**Resolved: `static_assert` on every element with `EnableFp32DestAcc=true`.**

```cpp
template <..., bool EnableFp32DestAcc = false>
struct BinaryFpu {
    static_assert(!EnableFp32DestAcc || DST_ACCUM_MODE,
                  "BinaryFpu<...EnableFp32DestAcc=true> requires kernel built with "
                  "FP32_DEST_ACC_EN (DST_ACCUM_MODE must be 1).");
    // ...
};
```

`DST_ACCUM_MODE` is a compile-time integer the kernel build defines as `1` when `FP32_DEST_ACC_EN` is set, `0` otherwise (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:43,45,46`). The `static_assert` rejects element-level `EnableFp32DestAcc=true` if the kernel did not opt into fp32 DST sizing — preventing the silent garbage-DST-readout failure mode noted in U3's risk table.

### Q11 (deferred — Bisect ergonomics of P1+U1-OldCb)
The directive allows splitting into `(a) tracking infrastructure` and `(b) hoist gate`. **Single commit.** Rationale: infrastructure has no consumer until the hoist gate lands; splitting creates a phantom-feature commit. D6's transition fold lives in the same body of code — splitting would create three identical fold infrastructures.

### Q12 (deferred) — Should `OldCb*` be removed from Block elements in P1+U1-OldCb?
**No.** Block elements at `eltwise_block.hpp:72,235` actively call `_with_dt` two-arg LLK forms. These have FP32_DEST_ACC-gated semantics that single-arg lacks (`with_dt_tree.md` lines 23-32). Removing `OldCb*` from block path requires generalizing the prev-CB fold over the block path's per-iter init shape — out of scope for this commit. Future commit, after the streaming path's elision is exercised in production.

---

## Final ordered commit list

| # | Subject (suggested)                                                                       | Scope                                                                                              |
|---|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| 1 | `eltwise v2: drop eltwise_pipeline_init; caller owns compute_kernel_hw_startup`           | F-UX-1 (partial), F-UX-12 (relocated), F-UX-16, **D5** placement contract — delete helper, sweep 26 prod + 13 test kernels with top-of-`MAIN()` placement enforcement. |
| 2 | `eltwise v2: compile-time prev-CB tracking + init hoist`                                   | F-UX-7 (decls only), F-PERF-1 follow-up, **D6 transition fold** — new `prev_cb_for_idx` + `prev_fp32_dest_acc_for_idx` folds; drop `OldCb*` from streaming element decls. |
| 3 | `eltwise v2: drop dead OldCb padding from binary kernels (~17) + delete EltwiseChainOptions` | F-UX-7, F-UX-3 (D6 chain options delete) — kernel-side sweep; only the four trailing zeros at call sites + `EltwiseChainOptions` struct + NTTP. |
| 4 | `eltwise v2: collapse BinaryFpu to 9 effective template params; reorder for callsite ergonomics; add per-element EnableFp32DestAcc` | F-UX-2, F-UX-5, **D6 per-element flag** — promote `CbOut` to position 3, demote `OutPolicy` to trailing default, add `EnableFp32DestAcc` trailing default; ~17 kernel rewrites. |
| 5 | `eltwise v2: add eltwise_chain_with_init deduced wrapper; sweep callers`                   | F-UX-1 (final), F-UX-16 — collapse boot+chain to one call; ≤25 kernels swept.                       |
| 6 | `eltwise v2: adopt OptionalChainElement in logit + where_tss; ship test kernel`            | F-UX-8 — adopt in 2 production kernels; new test kernel + parametrize.                              |
| 7 | `eltwise v2: doxygen + caller-init contract spec on chain helper headers`                  | Directive 3 — header-level doxygen, per-element annotations, **D5 placement table**, **D6 per-element fp32 notes**, examples. |

Per-commit regression bar: 401-suite + ALL ~74 migrated-kernel pytests covering the touched-kernel cluster, per Section D table.

Open after this run: F-UX-9 family-split, F-UX-19 BinarySfpu enum unification, F-UX-20 local-op dedup, block-mode prev-CB fold extension, block-mode D6 fp32-dest-acc transitions, F-PERF-5 gate narrowing — each tracked in Section E.
