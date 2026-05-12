# Partial-Kernel Audit v2 — Post Reg A/B/C Re-audit

Branch: `astancov/eltwise_run7_refined` HEAD `bcfed8161c5`.
Scope: same 47-file enumeration as v1 (`partial_kernel_audit.md` Section 1) — every kernel that imports `eltwise_chain.hpp` or family headers.
Goal: refresh classification per current helper state. Identify (a) blocks the Type-1 sweep already migrated, (b) NEWLY-Type-1 blocks unlocked by helper fixes OR by correcting v1 misclassifications, (c) remaining Type-2 catalog, (d) Type-3 patterns.

This audit re-reads each kernel at HEAD against `partial_kernel_audit.md` (v1, committed `d1d1f9246c5`).

---

## Section 1 — Method & helper-state delta

### Helper changes since v1 audit

The v1 audit was written at HEAD `3b0cc6026e8` (run7 refinement landed but Type-1 sweep not yet started). Since then:

| Change | Commit | Helper impact |
|---|---|---|
| **Type-1 sweep** (9 commits) | `71d31fab5a9` … `bd6cfdfb9ed` | 37 v1 Type-1 blocks migrated across 15 production kernels (see v1 Section 3 IDs T1.01–T1.37). All landed without helper-API changes. |
| **Reg A — BinaryFpu Bcast init pair** | `be533cfbbf4` | `BinaryFpu<..., Bcast != None>` no longer emits per-tile BIG init mid-MAIN. Uses `llk_math_eltwise_binary_init_with_operands<et, bt, FID>(CbA, CbB) + llk_unpack_AB_init<bt>(CbA, CbB)` short pair (`eltwise_chain.inl:528-532`). D8-compliant. Fixes softmax LARGE_H hang. **No new chain shapes unlocked** — broadcast was already a v1 Type-1 candidate (T1.18–T1.24); Reg A removed the actual-execution regression on those already-migrated sites. |
| **Reg B — moreh_sum/moreh_norm host fix** | `7060e1245a3` | `padded_shape().rank()` → `logical_shape().rank()`. Affects test reliability for moreh_sum/moreh_norm 1D inputs only. **Zero effect on chain-helper API or classification.** |
| **Reg C — PackTile reconfig fold guard removed** | `7ab7da6027c` | `hoisted_init_for_each` (`eltwise_chain.inl:1031-1045`) no longer skips PackTile elements. Non-clash chains (`CopyTile + SFPU + PackTile`) now correctly fire `pack_reconfig_data_format` when `PackTileReconfig::Output` is declared. Fixes moreh_adam T1.32/T1.33 (Power chain) which lost pack reconfig pre-fix. **No new chain shapes unlocked** — affected chains were already v1 Type-1 candidates; Reg C removed the actual-execution regression. |
| **fp32_dest_acc annotation pattern** | `c090b2fd14c`, `98fe0ae2ebc`, `8301d9d7677` | Methodology: `EnableFp32DestAcc=DST_ACCUM_MODE` + `CopyTileReconfig::Input` + `PackTileReconfig::Output` on per-element opt-in for chains that previously used `_with_dt` raw forms. Recovered the +68 fail regression class flagged in verification_report.md Section 2. **Not an API change** — these template params already existed. |

### Type-1/Type-2 boundary delta

**The Reg A and Reg C fixes did NOT promote any v1 Type-2 pattern into Type-1.** Both were corrections of execution-time bugs on already-Type-1-classified shapes:

- Reg A fixed the BinaryFpu Bcast EXECUTION path. The CLASSIFICATION (v1 T1.18–T1.24 are Type-1) was correct in v1; the helper just couldn't actually run those chains correctly mid-MAIN until Reg A.
- Reg C fixed the PackTile reconfig EMISSION. The CLASSIFICATION (CopyTile+SFPU+PackTile is Type-1) was correct; emission of `pack_reconfig_data_format` was just dropped on the hoisted path.

**The fp32 annotation pattern is methodology, not API.** Same template params, just used more carefully.

### v1 misclassification recovery (discovered during this re-audit)

While re-reading source at HEAD, I identified that **v1 conservatively classified the "asymmetric per-side Pinned tile-idx" pattern as Type-2** ("After Q4 collapse, no symmetric mode covers `A=BlockIter, B=FirstTile`"). This is incorrect. Q4 collapse merged the per-side MODE template param (`AIndex`/`BIndex` → single `Index`), but the runtime PER-ELEMENT tile-index member fields (`a_tile_idx`, `b_tile_idx`) remain independent (`eltwise_chain.inl:502-504,556-562`). With `Index=Pinned`, callers set `elt.a_tile_idx = wt; elt.b_tile_idx = 0;` and the helper computes `a_idx=wt, b_idx=0` correctly. The migrated `moreh_layer_norm_backward_input_grad_small_kernel.cpp:309-328` (cb_ydy chain) and `moreh_layer_norm_backward_input_grad_large_kernel.cpp:333-356` (cb_recip_nrstd chain) already use this exact pattern (`elt.a_tile_idx = 1; elt.b_tile_idx = 0`).

This recovery promotes several Type-2 v1 blocks to Type-1:

- `_small_kernel.cpp` L378-390 (cb_ndy = cb_n_recip_n[0] * cb_dycopy[wt])
- `_small_kernel.cpp` L431-448 (cb_yydysum = cb_y[wt] * cb_ydysum[0] bcast)
- `_large_kernel.cpp` L446-468 (cb_ndy, already migrated post-sweep but originally Type-2 in v1)

I also found one block missed by v1 entirely:

- `moreh_adamw.cpp` L216-225 (cb_tmp1 = cb_one - cb_scalar_args[beta2_tile]) — plain Type-1 BinaryFpu(Sub, Pinned, b_tile_idx=2) + PackTile. The analogous block in `moreh_adam.cpp` was already migrated via `moreh_bin_chain`; `moreh_adamw.cpp` left this one raw without v1 flagging it.

### Read protocol

Same as v1 Section 1: walk every kernel file at HEAD; classify each `tile_regs_acquire()…tile_regs_release()` block + each `*_tile_to_cb` / `power_tile_to_cb` helper invocation. Cross-reference v1 Section 2 entries.

---

## Section 2 — Per-kernel re-inventory

Notation:
- **A** = all chain-region blocks migrated; only legitimate raw scaffolding remains (e.g., outer cb_wait_front / preamble).
- **B-partial** = some chain blocks migrated, some remain (Type-1 or Type-2).
- **C-regressed** = Q4 disposition (c) — block intentionally raw, inline-commented.
- **A (post-sweep)** = was B in v1, now A after Type-1 sweep migrated all remaining chain blocks.

### 2.1 binary_ng — kernels/ family

#### `eltwise/binary_ng/.../eltwise_binary_no_bcast.cpp`
**Current:** B-partial (unchanged from v1).
**v1 Type-1 blocks migrated:** none (v1 reported none).
**Remaining blocks:**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L48-77 | activations branch — `PREPROCESS` + `BINARY_OP` + `PROCESS_POST_ACTIVATIONS` macros, multi-DEST | 2 | 2 | (MACRO-INJECT + MULTI-DEST). Unchanged. |
| L86-104 | remainder block — same macro-injected multi-DEST | 2 | 2 | (MACRO-INJECT + MULTI-DEST). Unchanged. |

#### `eltwise/binary_ng/.../eltwise_binary_sfpu_no_bcast.cpp`
**Current:** B-partial. Unchanged. `process_sfpu_tiles` L140-179 still Type-2 (MACRO-INJECT + MULTI-DEST).

#### `eltwise/binary_ng/.../eltwise_binary_sfpu_scalar.cpp`
**Current:** B-partial. Unchanged. `process_sfpu_scalar_tiles` L129-164 still Type-2.

#### `eltwise/binary_ng/.../eltwise_where_sfpu.cpp`
**Current:** B-partial. Outer freq-loop framing still Type-2 (OUTER-FREQ-LOOP).

#### `eltwise/binary_ng/.../eltwise_where_sfpu_scalar.cpp`
**Current:** B-partial. L139 cb_wait_front legitimate scaffolding (Type-3).

### 2.2 binary_ng — kernels_ng/ family

| Kernel | v1 Type | v2 Type | Note |
|---|---|---|---|
| `eltwise_binary_col_bcast.cpp` | 2 | 2 | Unchanged. OUTER-FREQ-LOOP + MACRO-INJECT. |
| `eltwise_binary_row_bcast.cpp` | 2 | 2 | Unchanged. |
| `eltwise_binary_row_col_bcast.cpp` | 2 | 2 | Unchanged. |
| `eltwise_binary_scalar_bcast.cpp` | 2 | 2 | Unchanged. |
| `eltwise_binary_sfpu_row_bcast.cpp` | 2 | 2 | Unchanged. |
| `eltwise_where_sfpu_row_bcast.cpp` | 2 | 2 | Unchanged. |

### 2.3 ternary — sfpu broadcast variants
All three kernels (`ternary_addc_ops_sfpu_bcast.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, `ternary_sfpu_row_bcast_ttt.cpp`) — v1 Type-2 (OUTER-FREQ-LOOP). Unchanged.

### 2.4 experimental/bcast_to/ — multi-axis early-exit
All three kernels — v1 Type-2 (MULTI-AXIS-EARLY-EXIT). Unchanged.

### 2.5 experimental/ccl/moe — matmul + PACK-thread SFPU
Both `moe_compute/compute.cpp` and `moe_gpt/compute.cpp` — v1 Type-3 (matmul + PACK-thread SFPU). Unchanged.

### 2.6 experimental/reduction/deepseek_grouped_gate
**Current:** B-partial (`scale` migrated by sweep, rest unchanged).
**v1 Type-1 blocks migrated:** L326-343 scale block (T1.37, commit `71d31fab5a9`).
**Remaining blocks:** unchanged from v1.
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L39-66 add_bias | Q4 asymmetric `BinaryFpu<A=BlockIter, B=FirstTile>` | C (Q4-regressed) | C | Inline-commented at L41-52. Still raw. |
| L70-126 process_and_sort_tiles | transpose + topk | 2/3 | 2/3 | (TRANSPOSE/TOPK). |
| L128-147 sum_top_experts_per_group | held-DEST multi-write | 2 | 2 | (HELD-DEST). |
| L149-182 topk_group_scores | acquire_dst + topk | 3 | 3 | Deprecated dst-sync. |
| L184-203 transpose_and_pack | transpose | 2/3 | 2/3 | |
| L205-267 topk | topk family | 2/3 | 2/3 | |
| L269-324 normalize_scores | held-DEST + reduce-fold | 2 | 2 | |
| L326-343 scale | already migrated | **A** | A | (sweep) |

### 2.7 experimental/ssm/prefix_scan
`ssm_prefix_scan.cpp` — v1 Type-3 (tilize/untilize). Unchanged.

### 2.8 moreh — moreh_adam / moreh_adamw

#### `moreh/moreh_adam/.../moreh_adam.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L270-281 `cb_tmp1 = pow(beta2, step)` (T1.32, commit `1711213980e`)
- L378-389 `cb_tmp2 = pow(beta1, step)` (T1.33, commit `1711213980e`)
- L318-328 (AMSGRAD) copy chain (T1.34, commit `1711213980e`)

**Remaining blocks (v1 Type-2 confirmed):**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L295-309 `cb_tmp1 = 1/(1 - cb_tmp1)` | HELD-CB on cb_tmp1 + held-DEST sub+recip | 2 | 2 | Unchanged. |
| L313-326 (AMSGRAD) `tmp_cb_max_exp_avg_sq = max(...)` | held-DEST + `binary_max_tile` | 2 | 2 | Unchanged. |
| L333-358 `sqrt(exp_avg_sq * cb_tmp1)` | HELD-CB on cb_tmp1 + held-DEST mul+sqrt | 2 | 2 | Unchanged. |
| L361-375 `cb_tmp1 = 1/(cb_tmp1+eps)` | HELD-CB on cb_tmp1 + held-DEST add+recip | 2 | 2 | Unchanged. |
| L403-417 `cb_tmp2 = 1/(1 - cb_tmp2)` | HELD-CB on cb_tmp2 + held-DEST sub+recip | 2 | 2 | Unchanged. |

#### `moreh/moreh_adamw/.../moreh_adamw.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L281-292 `cb_tmp1 = 1/(1 - cb_beta2_exponent)` (T1.35, commit `1711213980e`)
- L359-371 `cb_tmp2 = 1/(1 - cb_beta1_exponent)` (T1.36, commit `1711213980e`)

**v2 NEWLY-Type-1 block (missed by v1):**
| Block | Pattern | v1 Type | v2 Type | Helper mapping | Notes |
|---|---|---|---|---|---|
| L216-225 `cb_tmp1 = cb_one - cb_scalar_args[beta2_tile]` | seed-only (no held-CB; no pop on cb_tmp1) | **not classified** | **1** | `BinaryFpu<cb_one, cb_scalar_args, cb_tmp1, Sub, BroadcastDim::None, ..., CopyTilePolicy::NoWaitNoPop, Pinned, Dst::D0>{} + PackTile<cb_tmp1, ..., Output, EnableFp32DestAcc=DST_ACCUM_MODE>{}` with `b_tile_idx=beta2_tile=2` | The analogous block in `moreh_adam.cpp` is already migrated via `moreh_bin_chain`. v1 audit Section 2.8 enumeration jumped from L281-292 to L294-309, missing this earlier seed. |

**Remaining blocks (v1 Type-2 confirmed):**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L302-316 (AMSGRAD) max | held-DEST + `binary_max_tile` | 2 | 2 | Unchanged. |
| L322-345 sqrt(mul) | HELD-CB on cb_tmp1 + held-DEST | 2 | 2 | Unchanged. |
| L347-361 `1/(x+eps)` | HELD-CB on cb_tmp1 + held-DEST | 2 | 2 | Unchanged. |

### 2.9 moreh — clip_grad_norm_step1

`moreh_clip_grad_norm_step1_kernel.cpp` — Sweep did NOT touch this kernel (v1 had zero Type-1 targets). Unchanged: Type-2 (MASK-INJECT + POWER-MULTISTAGE + HELD-CB).

### 2.10 moreh — layer_norm forward (large + small)

Both `moreh_layer_norm_small_kernel.cpp` and `moreh_layer_norm_large_kernel.cpp` — Sweep did NOT touch (v1 had zero Type-1 targets). Unchanged.

### 2.11 moreh — layer_norm backward family

#### `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L191-213 cb_y = cb_xmm * cb_rstd (T1.24, commit `5460fdbbbe3`)

**Remaining blocks:**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L82-113 cb_dycopy | mid-loop conditional mask | 2 | 2 | (MASK-INJECT). |
| L131-146 cb_dyadd fold | held-CB on cb_dyadd | 2 | 2 | (HELD-CB). |
| L150-189 cb_xmm sub | mid-loop conditional mask | 2 | 2 | (MASK-INJECT). |
| L263-280 cb_ydyadd fold | held-CB on cb_ydyadd | 2 | 2 | (HELD-CB). |

#### `moreh_layer_norm_backward_input_grad_small_kernel.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L140-159 cb_y mul bcast no-mask (T1.22, commit `a2104f28449`)
- L377-398 cb_ndymdysum (T1.23, commit `a2104f28449`)
- (plus seed chains and final dx chain — already in v1 as A)

**v2 NEWLY-Type-1 blocks (v1 misclassified as Type-2 — recoverable now):**
| Block | Pattern | v1 Type | v2 Type | Helper mapping | Notes |
|---|---|---|---|---|---|
| L378-390 cb_ndy = cb_n_recip_n[0] * cb_dycopy[wt] | per-iter Pinned-asymmetric, no held-CB, no mask, both pinned | 2 (called "Q4 asymmetric") | **1 (was Type-2 in v1)** | `BinaryFpu<cb_n_recip_n, cb_dycopy, cb_ndy, Mul, BroadcastDim::None, ..., NoWaitNoPop, NoWaitNoPop, Pinned>` with `a_tile_idx=0, b_tile_idx=wt` set per-iter + `PackTile<cb_ndy, ..., Output>` | Q4 collapse merged MODE template, NOT per-element tile_idx member fields (`eltwise_chain.inl:502-504,556-562`). Same Pinned mode as the migrated cb_recip_nrstd L78-96 and cb_ydy L309-328 chains. |
| L431-448 cb_yydysum = cb_y[wt] * cb_ydysum[0] bcast | per-iter Pinned-asymmetric, bcast Cols/Scalar, no mask | 2 (called "Q4 asymmetric") | **1 (was Type-2 in v1)** | `BinaryFpu<cb_y, cb_ydysum, cb_yydysum, Mul, BCAST_DIM, ..., NoWaitNoPop, NoWaitNoPop, Pinned>` with `a_tile_idx=wt, b_tile_idx=0` + `PackTile<cb_yydysum, ..., Output>`. BCAST_DIM = Col if `is_lastdim_layernorm` else Scalar. | Same recovery as L378-390. Reg A's `_init_with_operands` form ensures this bcast init pair is D8-compliant. |

**Remaining blocks (v1 Type-2 confirmed):**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L101-138 cb_xmm sub-with-mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L181-223 cb_dycopy gamma-path with mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L225-256 cb_dycopy !gamma-path with mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L263-292 cb_dyadd fold | held-CB on cb_dyadd | 2 | 2 | (HELD-CB). |
| L331-361 cb_ydyadd fold | held-CB on cb_ydyadd | 2 | 2 | (HELD-CB). |

#### `moreh_layer_norm_backward_input_grad_large_kernel.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L77-98 cb_xmm sub bcast no-mask (T1.18, commit `bd6cfdfb9ed`)
- L451-477 → now L471-505 cb_ndymdysum (T1.19, commit `bd6cfdfb9ed`)
- L514-540 → now L546-580 cb_y mul bcast no-mask (T1.20, commit `bd6cfdfb9ed`)
- L542-562 → now L583-617 cb_yydysum (T1.21, commit `bd6cfdfb9ed`)
- L446-468 cb_ndy mul (post-sweep refactor — v1 didn't have this as a separate ID but it's the analogous shape to small-kernel L378-390 and was migrated by `bd6cfdfb9ed`)

**Remaining blocks (v1 Type-2 confirmed):**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L114-150 cb_y mul bcast WITH mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L154-197 cb_dycopy gamma-path with mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L198-230 cb_dycopy !gamma-path with mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L233-262 cb_dyadd fold | held-CB on cb_dyadd | 2 | 2 | (HELD-CB). |
| L289-319 cb_ydyadd fold | held-CB on cb_ydyadd | 2 | 2 | (HELD-CB). |
| L366-442 second cb_dycopy gamma/!gamma blocks with mask (in second wt loop) | mid-loop mask | 2 | 2 | (MASK-INJECT). |
| L507-544 cb_xmm sub-with-mask | mid-loop mask | 2 | 2 | (MASK-INJECT). |

### 2.12 moreh — moreh_mean_nc
`moreh_mean_nc.cpp` — Sweep did NOT touch. Unchanged: Type-2 (RUNTIME-CB + HELD-CB).

### 2.13 moreh — nll_loss family

#### `moreh_nll_loss_step2_kernel.cpp`
**Current:** **A (post-sweep)** — promoted from B-partial.
**v1 Type-1 blocks migrated:** L43-61 (T1.28), L63-80 (T1.29), L82-97 (T1.30), L106-130 (T1.31) — all in commit `6ea995dc66d`. Also re-annotated with `EnableFp32DestAcc=DST_ACCUM_MODE` in commit `98fe0ae2ebc` (sweep fix).
**Remaining blocks:** none beyond legitimate scaffolding.

#### `moreh_nll_loss_backward_kernel.cpp`
**Current:** **A (post-sweep)** — promoted from B-partial.
**v1 Type-1 blocks migrated:** L48-63 (T1.25), L65-79 (T1.26), L82-100 (T1.27) — all in commit `6ea995dc66d`, re-annotated in `98fe0ae2ebc`.
**Remaining blocks:** none beyond legitimate scaffolding.

### 2.14 moreh — moreh_norm (6 kernels)

| Kernel | Sweep touch | v1 Type-1 → v2 status |
|---|---|---|
| `moreh_norm/moreh_norm_h/.../moreh_norm_h_kernel.cpp` | none | unchanged: Type-2 (MASK-INJECT + POWER-MULTISTAGE + HELD-CB) |
| `moreh_norm/moreh_norm_other/.../moreh_norm_other_kernel.cpp` | none | unchanged: Type-2 (POWER-MULTISTAGE + HELD-CB) |
| `moreh_norm/moreh_norm_w/.../moreh_norm_w_kernel.cpp` | none | unchanged: Type-2 |
| `moreh_norm/ord_other/moreh_norm_h/.../moreh_norm_h_kernel.cpp` | none | unchanged: Type-2 (MASK-INJECT + HELD-CB) |
| `moreh_norm/ord_other/moreh_norm_nc/.../moreh_norm_nc_kernel.cpp` | **T1.17 migrated** (commit `aae64d80993`) | B-partial; L82-109 remaining HELD-CB + binary_max_tile (under !IS_ZERO) |
| `moreh_norm/ord_other/moreh_norm_w/.../moreh_norm_w_kernel.cpp` | none | unchanged: Type-2 |

### 2.15 moreh — moreh_sgd
`moreh_sgd.cpp` — Sweep did NOT touch. Unchanged: Type-2 (RUNTIME-CB + RUNTIME-POP).

### 2.16 moreh — softmax (3 kernels)

#### `moreh_softmax_c_large.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L128 SOFTMAX `exp_tile_to_cb` (T1.11, commit `dfadf7b344b`, via `moreh_unary_chain<Exp,...>`)
- L140 else SOFTMIN `rexp_tile_to_cb` (T1.12, commit `dfadf7b344b`, via `moreh_rexp_chain`)
- L160 LOG `log_tile_to_cb` (T1.13, commit `dfadf7b344b`)
- L163 else `recip_tile_to_cb` (T1.14, commit `dfadf7b344b`)
- Plus fp32 annotation pass (commit `c090b2fd14c`)

**Remaining blocks:**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L139-160 (i>0 binary_max block) | held-CB on cb_max + `binary_max_tile` | 2 | 2 | (HELD-CB + binary_max_tile missing helper struct). |

#### `moreh_softmax_h_large.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:** L156, L169, L263, L288 (4× `exp_tile_to_cb` / `rexp_tile_to_cb`) — T1.15 collapsed, via `moreh_unary_chain` / `moreh_rexp_chain` templates; commit `dfadf7b344b`, fp32 annotation `c090b2fd14c`.
**Remaining blocks:**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L137, L146 `mask_tile_to_cb` | mask injection | 2 | 2 | (MASK-INJECT). |
| L173, L182 `exp_tile_and_mask_tile_to_cb` / `rexp_tile_and_mask_tile_to_cb` | mask injection inside exp/rexp | 2 | 2 | (MASK-INJECT inside unary chain). |

#### `moreh_softmax_w_large.cpp`
**Current:** B-partial. Mirror of `_h_large.cpp` with Col bcast.
**v1 Type-1 blocks migrated:** L190, L203, L297, L322 (T1.16 collapsed) — same pattern as `_h_large.cpp`.
**Remaining blocks:** 4× `mask_tile_to_cb` / `*_and_mask_tile_to_cb` (Type-2 MASK-INJECT).

### 2.17 moreh — softmax_backward (5 kernels)

#### `moreh_softmax_backward_c_large.cpp`
**Current:** **A (post-sweep)** — promoted from B-partial (no mask in this kernel; only Type-1 blocks remained).
**v1 Type-1 blocks migrated:**
- L102 `exp_tile_to_cb` (T1.01, commit `7dd7cf3824a`)
- L182 `mul_tiles_and_negative_to_cb` (T1.02, commit `7dd7cf3824a`)
- fp32 annotation pass (commit `8301d9d7677`)

**Remaining blocks:** none beyond legitimate scaffolding.

#### `moreh_softmax_backward_h.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:**
- L129 `exp_tile_to_cb(cb_y, cb_exp, w, ...)` (T1.03, commit `7dd7cf3824a`)
- L204 `mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx, h, 0, ...)` (T1.04, commit `7dd7cf3824a`)

**Remaining blocks (v1 Type-2 confirmed):**
| Block | Pattern | v1 Type | v2 Type | Notes |
|---|---|---|---|---|
| L95, L106 `mask_tile_to_cb` | mask injection | 2 | 2 | (MASK-INJECT). |
| L161, L162 (and L235 in newer state) `mul_tiles_and_mask_tile_to_cb` | mask injection inside mul | 2 | 2 | (MASK-INJECT). |

#### `moreh_softmax_backward_h_large.cpp`
**Current:** B-partial.
**v1 Type-1 blocks migrated:** L131 `exp_tile_to_cb` (T1.05), L227 `mul_tiles_and_negative_to_cb` (T1.06) — commit `7dd7cf3824a`.
**Remaining blocks:** L169, L180 `mask_tile_to_cb`, L235 `mul_tiles_and_mask_tile_to_cb` (all MASK-INJECT, Type-2).

#### `moreh_softmax_backward_w.cpp`
**Current:** B-partial. Mirror of `_h.cpp` with Col bcast.
**v1 Type-1 blocks migrated:** L128 (T1.07), L203 (T1.08).
**Remaining blocks:** mask injection sites (Type-2 MASK-INJECT).

#### `moreh_softmax_backward_w_large.cpp`
**Current:** B-partial. Mirror of `_h_large.cpp`.
**v1 Type-1 blocks migrated:** L131 (T1.09), L225 (T1.10).
**Remaining blocks:** mask injection sites (Type-2 MASK-INJECT).

---

## Section 3 — Newly Type-1 (post-fix or post-correction) target list

Sorted by leverage (ease × isolation × low-risk). 4 new IDs T1.5-01 .. T1.5-04.

| ID | Kernel | Block (line range) | Helper pattern | Migration cost (LOC ~) | Pytest | Reason newly Type-1 |
|----|--------|--------------------|----------------|-------------------------|--------|---------------------|
| **T1.5-01** | `moreh_adamw/.../moreh_adamw.cpp` | L216-225 | `BinaryFpu<cb_one, cb_scalar_args, cb_tmp1, Sub, Bcast=None, ..., NoWaitNoPop, NoWaitNoPop, Pinned, Dst::D0, EnableFp32DestAcc=DST_ACCUM_MODE>{}` with `b_tile_idx=2` + `PackTile<cb_tmp1, ..., Output, DST_ACCUM_MODE>{}` | -12/+12 | `test_moreh_adamw.py` | Missed by v1 audit (gap in Section 2.8 between L292 and L294). Analogous block in `moreh_adam.cpp` is already migrated via `moreh_bin_chain`. |
| **T1.5-02** | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_small_kernel.cpp` | L378-390 | `BinaryFpu<cb_n_recip_n, cb_dycopy, cb_ndy, Mul, Bcast=None, ..., NoWaitNoPop, NoWaitNoPop, Pinned, Dst::D0>{}` with `a_tile_idx=0, b_tile_idx=wt` set per-iter + `PackTile<cb_ndy, ..., Output>{}` | -12/+18 | `test_moreh_layer_norm_backward.py` | v1 misclassified as Type-2 ("Q4 asymmetric"). Per-element `a_tile_idx`/`b_tile_idx` are independent runtime member fields — Q4 collapse only merged the MODE template (`eltwise_chain.inl:502-504,556-562`). |
| **T1.5-03** | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_small_kernel.cpp` | L431-448 | `BinaryFpu<cb_y, cb_ydysum, cb_yydysum, Mul, Bcast=BCAST_DIM, ..., NoWaitNoPop, NoWaitNoPop, Pinned, Dst::D0>{}` with `a_tile_idx=wt, b_tile_idx=0` + `PackTile<cb_yydysum, ..., Output>{}`. BCAST_DIM = Col if `is_lastdim_layernorm` else Scalar. | -22/+28 | `test_moreh_layer_norm_backward.py` | Same v1 misclassification recovery as T1.5-02. Bcast init now D8-compliant via Reg A. |
| **T1.5-04** | `experimental/reduction/deepseek_grouped_gate/.../deepseek_grouped_gate.cpp` | L39-66 add_bias | Could be Type-1 via the same per-element Pinned recovery (A.tile_idx=t, B.tile_idx=0). **However:** the v1 Q4 disposition document (`capabilities.md:114`) labels this kernel as disposition-(c) regress-to-raw. Marked as **Type-1 candidate pending re-verification** — the bias_b CB is `cb_in_bias` per-tile and the bcast nature (scalar bias per row) may demand Bcast=Scalar with B.tile_idx=0. Needs concrete re-read. | (TBD) | `test_deepseek_grouped_gate.py` | Speculative. Listed for completeness; recommend skipping in any sweep until verified. |

**Net Type-1.5 headroom: 3 confidently-Type-1 blocks (T1.5-01, T1.5-02, T1.5-03) + 1 speculative (T1.5-04).** Total ~50 LOC removed, ~60 LOC added — net +10 LOC across 3 commits-or-1-commit, low risk.

---

## Section 4 — Still-Type-2 catalog (deferred — would unlock more migrations)

Ranked by kernel count.

| Helper feature gap | Kernels unlocked | v1 reference | Estimated new helper LOC | Notes |
|---|---|---|---|---|
| **MASK-INJECT** — `MaskInject<COND, MaskCb, MaskIdx>` chain element to inject `mask_tile(dst_data, dst_mask)` after another element (mid-loop conditional). Composed with `OptionalChainElement<COND, ...>` for runtime gates. | `moreh_clip_grad_norm_step1`, `moreh_norm_h`, `moreh_norm_w`, `moreh_norm_other`, `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_w`, `moreh_layer_norm_small`, `moreh_layer_norm_large`, `moreh_layer_norm_backward_gamma_beta_grad`, `moreh_layer_norm_backward_input_grad_small`, `moreh_layer_norm_backward_input_grad_large`, `moreh_softmax_h_large`, `moreh_softmax_w_large`, `moreh_softmax_backward_h`, `moreh_softmax_backward_h_large`, `moreh_softmax_backward_w`, `moreh_softmax_backward_w_large` (17 kernels) | v1 Section 4 row 1 | ~80 LOC | **#1 highest leverage.** Same as v1; still 17 kernels. |
| **HELD-CB** — `HeldCbBinaryFpu<...>` element with `cb_pop_front(X) + cb_push_back(X)` accumulator on same CB. | `moreh_adam`, `moreh_adamw`, `moreh_clip_grad_norm_step1`, `moreh_layer_norm_small`, `moreh_layer_norm_large`, `moreh_layer_norm_backward_gamma_beta_grad`, `moreh_layer_norm_backward_input_grad_small`, `moreh_layer_norm_backward_input_grad_large`, `moreh_norm_h`, `moreh_norm_w`, `moreh_norm_other`, `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_nc`, `moreh_norm/ord_other/moreh_norm_w`, `moreh_softmax_c_large`, `moreh_mean_nc` (16 kernels) | v1 Section 4 row 2 | ~60 LOC | **#2 highest leverage.** Up from v1's 13 — adding moreh_norm_nc (held-CB on cb_cal under !IS_ZERO) and re-counting moreh_adam/adamw's distinct held-CB sites. |
| **MOREH `binary_max_tile` op-struct** — wrap moreh's `binary_max_tile(dst0, dst1, dst0)` as a chain `DestReuseBinary`-like element. | `moreh_adam`, `moreh_adamw`, `moreh_softmax_c_large`, `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_nc`, `moreh_norm/ord_other/moreh_norm_w` (6 kernels) | v1 Section 4 row 10 | ~30 LOC | Pairs with HELD-CB for the `max-then-pack` pattern. |
| **MULTI-DEST per-iter (j as DEST slot AND tile index)** — `BlockBinaryFpu<..., MultiDestStride=1>` that uses j∈[0..BlockSize) as both DEST slot and pack index. | All block_size>1 `moreh_layer_norm_{small,large}`, all binary_ng `kernels_ng/*` (8 kernels) | v1 Section 4 row 9 | ~60 LOC | Couples with macro-injection wrapper. |
| **ATOMIC PACK-N-AFTER-N-COPIES** + **MACRO-INJECT WRAPPER** — `BlockPackTileAtomic<N>` + `LocalMacroBlockSfpu<MACRO, BlockSize>` for binary_ng activations branches. | `eltwise_binary_no_bcast`, `eltwise_binary_sfpu_no_bcast`, `eltwise_binary_sfpu_scalar`, all 6 `kernels_ng/*` activations branches (9 kernels) | v1 Section 4 row 4 & 6 | ~120 + ~80 LOC | Joined into one row — they always co-occur. |
| **OUTER-FREQ-LOOP DRIVER** — chain wraps `complete_iterations × tile_freq + remainder` with caller preamble hooks. | `eltwise_where_sfpu`, `eltwise_where_sfpu_scalar`, `ternary_addc_ops_sfpu_bcast`, `ternary_sfpu_col_scalar_bcast_ttt`, `ternary_sfpu_row_bcast_ttt`, all 6 `kernels_ng/*` (11 kernels) | v1 Section 4 row 5 | ~100 LOC | |
| **`copy_tile_init_with_dt(cb, transpose_bool)`** non-standard moreh-side init — would let `moreh_layer_norm_*` `cb_mean → cb_rstd` copies migrate. | `moreh_layer_norm_small`, `moreh_layer_norm_large` (2 kernels) | v1 Section 4 row 7 | ~15 LOC | |
| **`UnaryBcast`-as-preamble chain element** — wraps `unary_bcast_init + unary_bcast(cb_bcast, 0, 0) + pack_tile(0, cb_llk_post)` for `kernels_ng/*` bcast variants. | `kernels_ng/eltwise_binary_col_bcast`, `_row_bcast`, `_row_col_bcast`, `_scalar_bcast`, `_sfpu_row_bcast`, `_where_sfpu_row_bcast` (6 kernels) | v1 Section 4 row 8 | ~50 LOC | |
| **POWER-MULTISTAGE op-struct family** — wrap `power_tile_to_cb` multi-stage helper. | `moreh_norm_*`, `moreh_clip_grad_norm_step1` (5+ kernels) | v1 Section 4 row 11 | ~80 LOC | Very specialised. |
| **`BinaryFpuPerTileScalarB`** (was v1 Section 4 row 3 — ASYMMETRIC-INDEX) | Originally listed as 5+ kernels. **Updated count: reduces to 1–2** since v2 identifies that the per-element `a_tile_idx`/`b_tile_idx` already support asymmetric A/B index values under Pinned mode. Remaining gap is the MODE asymmetry (A=BlockIter, B=FirstTile literally) which only applies to: `deepseek_grouped_gate.cpp::add_bias` (Q4 disposition (c)) and `eltwise_binary_scalar.cpp` no-act fast path. The latter is out of v1 scope. | v1 Section 4 row 3 | ~25 LOC | **Reduced leverage from v1.** v1 over-counted this. Most v1 "asymmetric-index" Type-2 blocks are actually Type-1 (T1.5-02, T1.5-03 above) — the few remaining true-asymmetric MODE callers are in C-regressed kernels. |

### Top 3 highest-leverage Type-2 helper extensions (re-confirmed)

| Rank | Extension | Kernel count | v1 count | Delta |
|---|---|---|---|---|
| **1** | **MaskInject** | 17 | 17 | unchanged |
| **2** | **HeldCbBinaryFpu** | 16 | 13 | +3 (re-counting tighter) |
| **3** | **MultiDestPerIter** + **MacroInjectWrapper** + **AtomicPackN** combined (binary_ng activations + kernels_ng) | 9 | (was split rows) | combined; new joint-feature framing |

**The `BinaryFpuPerTileScalarB` (was v1's #3) drops to 1–2 kernels** after Type-1.5 recovery (T1.5-02/T1.5-03 in Section 3 above migrate the use cases v1 attributed to this gap).

### New extensions emerging from v2 reclassification

None. The Type-2 catalog is stable — the v1 misclassification recovery does not introduce new gaps; it simply moves blocks from Type-2 into Type-1.

---

## Section 5 — Still-Type-3 catalog

Unchanged from v1 Section 5:
- `tilize_block` / `untilize_block` — `ssm_prefix_scan.cpp`.
- `transpose_wh_init_short` / `transpose_wh_tile` — `deepseek_grouped_gate.cpp`.
- `topk_local_sort` / `topk_merge` / `topk_rebuild` — `deepseek_grouped_gate.cpp`.
- `acquire_dst()` / `release_dst()` (deprecated dst-sync) — `deepseek_grouped_gate.cpp`.
- PACK-thread SFPU — `moe_compute/compute.cpp`, `moe_gpt/compute.cpp`.
- `matmul_block` / `mm_init` — `moe_compute`, `moe_gpt`.
- `init_sfpu` alternate engine-boot — `moreh_nll_loss_backward_kernel.cpp:27` (caller-owned, unrelated to chain).

---

## Section 6 — Migration summary stats

| Stat | v1 audit | v2 audit (HEAD) | Δ |
|---|---|---|---|
| Total kernels in scope (file-level) | 47 | 47 | 0 |
| Fully migrated (A class) | 0 | **2** (`moreh_nll_loss_step2_kernel.cpp`, `moreh_nll_loss_backward_kernel.cpp`) + **1** (`moreh_softmax_backward_c_large.cpp`) = **3** | +3 |
| Partial with newly-Type-1 blocks ready to migrate | n/a | 3 (T1.5-01..T1.5-03) | +3 |
| Partial blocked-by-extension (Type-2 only after Type-1 sweep + Type-1.5 recovery) | n/a | 24 | n/a |
| Regressed (Q4 disposition (c)) — count of kernels | 2 (`deepseek_grouped_gate::add_bias`, `eltwise_binary_scalar`) | 2 | 0 |
| Un-migratable (Type-3 only) | 5 | 5 | 0 |
| Type-1 blocks ready to migrate | 37 | 0 remaining of v1 set + 3 newly identified | sweep delivered all 37 |
| **NET HEADROOM** | — | **3 additional Type-1 blocks (1 missed-by-v1 + 2 Pinned-recovery)** across **2 kernels** could migrate in a Type-1.5 sweep | — |

Kernels promoted to A (post-sweep):
- `moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` — all 4 v1 Type-1 blocks (T1.28–T1.31) migrated.
- `moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp` — all 3 v1 Type-1 blocks (T1.25–T1.27) migrated.
- `moreh_softmax_backward/.../moreh_softmax_backward_c_large.cpp` — both T1.01 and T1.02 migrated, no mask in this kernel.

---

## Section 7 — Sweep recommendations

### Verdict

**Type-1.5 sweep is NOT worth doing standalone.** Net headroom is 3 blocks across 2 kernels (~50 LOC). The marginal gain is low and the chain dispatch path has just survived a YELLOW verification pass (verification_report.md Section 2) with +68 fail regression class needing follow-up annotation. Spending a sweep cycle on 3 mechanical blocks adds risk without proportional reward.

**Recommended next pass: Type-2 helper extension pass starting with `MaskInject` (17 kernels) + `HeldCbBinaryFpu` (16 kernels).** These two extensions together unlock ~26 unique kernels into deeper migration depth. The MaskInject extension specifically addresses the `*_tile_and_mask_tile_to_cb` family that v1 enumerated as 17-kernel-wide, and converts a high-volume Type-2 class to Type-1.

If a Type-1.5 sweep IS done (e.g., as a small-and-tidy cleanup commit alongside the Type-2 helper extension landing), the 3 blocks should be batched into a single commit:

1. **Single commit "Type-1.5 cleanup: missed-by-v1 + Pinned-asymmetric recovery"**
   - `moreh_adamw.cpp` L216-225 (T1.5-01)
   - `moreh_layer_norm_backward_input_grad_small_kernel.cpp` L378-390 (T1.5-02)
   - `moreh_layer_norm_backward_input_grad_small_kernel.cpp` L431-448 (T1.5-03)
   - Validation: `test_moreh_adamw.py`, `test_moreh_layer_norm_backward.py`. Both already in the implementer's standard regression battery.
   - Risk hotspot: T1.5-03 has bcast — exercise under `--dev` first to confirm Reg A's `_init_with_operands` form correctly programs the per-iter tensor shape when A.tile_idx changes per iter.

### Risk callouts

- **T1.5-02 / T1.5-03 (Pinned-asymmetric)** — These two migrations rely on the v2 finding that `a_tile_idx` and `b_tile_idx` are independent runtime member fields under `Index=Pinned`. The migrated `cb_recip_nrstd` chain in the same kernel (L78-96) already uses `elt.a_tile_idx = 1; b_tile_idx default 0` — so the pattern is proven. Still, a `--dev` probe before committing is recommended.

- **T1.5-04 (deepseek add_bias)** — Speculative. Marked as out-of-scope unless concrete re-read by a planner agent confirms Bcast=Scalar with B-side Pinned-0 is the correct shape.

### Skipped kernels (Type-2-only or Type-3-only — no Type-1.5 candidates)

Same as v1 Section 6 skip list, plus:
- `moreh_norm/ord_other/moreh_norm_nc/...` was migrated (T1.17); now Type-2-only for the held-CB remainder.

---

## Section 8 — Appendix: verification of the Reg A/C "no new Type-1 promotions" claim

Cross-check: does Reg A or Reg C transition any v1 Type-2 block to Type-1?

**Reg A** (`be533cfbbf4`) only affects `BinaryFpu::init()` when `Bcast != None`. The PRE-Reg-A bug was BIG-init mid-MAIN (D8 violation, undefined behaviour). The fix swaps to `_init_with_operands` short-init pair. This affects whether already-Type-1 bcast chains EXECUTE correctly — it doesn't change which shapes the chain CAN express. The v1 Type-1 bcast IDs (T1.18–T1.24) were correct as Type-1; they just couldn't run mid-MAIN until Reg A.

**Reg C** (`7ab7da6027c`) affects `hoisted_init_for_each` to fire `emit_pre_element_transitions` for PackTile elements. Without this, non-clash chains (CopyTile+SFPU+PackTile) couldn't reconfig pack format declared via `PackTileReconfig::Output`. Again, the chain SHAPE was already Type-1 (T1.32, T1.33, T1.34 for moreh_adam Power); Reg C just made the emission correct. No new shape categories unlocked.

**Conclusion:** the only "new Type-1" findings in v2 come from:
1. v1 missing a block (T1.5-01).
2. v1 conservatism on the per-element tile_idx asymmetry under Pinned mode (T1.5-02, T1.5-03).

Reg A/C/B are correct fixes but expand the EXECUTION envelope of existing Type-1 shapes, not the CLASSIFICATION envelope.

---

*End of audit v2.*
