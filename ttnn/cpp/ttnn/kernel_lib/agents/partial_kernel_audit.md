# Partial-Kernel Audit — Type-1 Migration Targets Across 31 B-Classified Kernels

Branch: `astancov/eltwise_run7_refined` HEAD `3b0cc6026e8` (8 implementation commits on top of design v6).
Scope: every kernel classified **B** in `coverage_audit.md` Section 2/3 (31 kernels).
Goal: per-kernel, identify raw-LLK blocks that can migrate to the chain helper today **without any helper extension**. Those are the **Type 1** targets for the next sweep.

---

## Section 1 — Method

### How the 31 kernels were sourced

Reading `coverage_audit.md` Section 5 (Summary stats, line 481): "B — Partial migration: 31 (production)". Section 2's production-kernels table (lines 140–219) marks each kernel with class A/B/C/D/E. The 31 B entries from that table:

1. `eltwise/binary_ng/.../eltwise_binary_no_bcast.cpp` (audit line 150)
2. `eltwise/binary_ng/.../eltwise_binary_sfpu_no_bcast.cpp` (audit line 152)
3. `eltwise/binary_ng/.../eltwise_binary_sfpu_scalar.cpp` (audit line 153)
4. `eltwise/binary_ng/.../eltwise_where_sfpu.cpp` (audit line 155)
5. `eltwise/binary_ng/.../eltwise_where_sfpu_scalar.cpp` (audit line 156)
6. `eltwise/binary_ng/kernels_ng/eltwise_binary_col_bcast.cpp` (audit line 157)
7. `eltwise/binary_ng/kernels_ng/eltwise_binary_row_bcast.cpp` (audit line 158)
8. `eltwise/binary_ng/kernels_ng/eltwise_binary_row_col_bcast.cpp` (audit line 159)
9. `eltwise/binary_ng/kernels_ng/eltwise_binary_scalar_bcast.cpp` (audit line 160)
10. `eltwise/binary_ng/kernels_ng/eltwise_binary_sfpu_row_bcast.cpp` (audit line 161)
11. `eltwise/binary_ng/kernels_ng/eltwise_where_sfpu_row_bcast.cpp` (audit line 162)
12. `eltwise/ternary/.../ternary_addc_ops_sfpu_bcast.cpp` (audit line 165)
13. `eltwise/ternary/.../ternary_sfpu_col_scalar_bcast_ttt.cpp` (audit line 167)
14. `eltwise/ternary/.../ternary_sfpu_row_bcast_ttt.cpp` (audit line 170)
15. `experimental/bcast_to/.../compute_interleaved_col_bcast_to.cpp` (audit line 182)
16. `experimental/bcast_to/.../compute_interleaved_row_bcast_to.cpp` (audit line 183)
17. `experimental/bcast_to/.../compute_interleaved_scalar_bcast_to.cpp` (audit line 184)
18. `experimental/ccl/moe_compute/.../compute.cpp` (audit line 185)
19. `experimental/ccl/moe_gpt/.../compute.cpp` (audit line 186)
20. `experimental/reduction/deepseek_grouped_gate/.../deepseek_grouped_gate.cpp` (audit line 188 — "C+B")
21. `experimental/ssm/prefix_scan/.../ssm_prefix_scan.cpp` (audit line 189)
22. `moreh/moreh_adam/.../moreh_adam.cpp` (audit line 191 — re-classified B)
23. `moreh/moreh_adamw/.../moreh_adamw.cpp` (audit line 192 — re-classified B)
24. `moreh/moreh_clip_grad_norm/.../moreh_clip_grad_norm_step1_kernel.cpp` (audit line 193)
25. `moreh/moreh_layer_norm/.../moreh_layer_norm_large_kernel.cpp` (audit line 195)
26. `moreh/moreh_layer_norm/.../moreh_layer_norm_small_kernel.cpp` (audit line 196)
27. `moreh/moreh_layer_norm_backward/.../moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` (audit line 197)
28. `moreh/moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_large_kernel.cpp` (audit line 198)
29. `moreh/moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_small_kernel.cpp` (audit line 199)
30. `moreh/moreh_mean/.../moreh_mean_nc.cpp` (audit line 200)
31. `moreh/moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` (audit line 202)
32. `moreh/moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp` (audit line 203)
33. `moreh/moreh_norm/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` (audit line 204)
34. `moreh/moreh_norm/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp` (audit line 205)
35. `moreh/moreh_norm/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` (audit line 206)
36. `moreh/moreh_norm/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` (audit line 207)
37. `moreh/moreh_norm/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp` (audit line 208)
38. `moreh/moreh_norm/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` (audit line 209)
39. `moreh/moreh_sgd/.../moreh_sgd.cpp` (audit line 210)
40. `moreh/moreh_softmax/.../moreh_softmax_c_large.cpp` (audit line 211)
41. `moreh/moreh_softmax/.../moreh_softmax_h_large.cpp` (audit line 212)
42. `moreh/moreh_softmax/.../moreh_softmax_w_large.cpp` (audit line 213)
43. `moreh/moreh_softmax_backward/.../moreh_softmax_backward_c_large.cpp` (audit line 214)
44. `moreh/moreh_softmax_backward/.../moreh_softmax_backward_h.cpp` (audit line 215)
45. `moreh/moreh_softmax_backward/.../moreh_softmax_backward_h_large.cpp` (audit line 216)
46. `moreh/moreh_softmax_backward/.../moreh_softmax_backward_w.cpp` (audit line 217)
47. `moreh/moreh_softmax_backward/.../moreh_softmax_backward_w_large.cpp` (audit line 218)

**Reconciliation.** That's 47 entries above; the audit's headline count is "31 (production)". The discrepancy is the moreh-family rollup (`moreh_norm` 6 sub-kernels collapsed in audit Section 5 vs split rows in Section 2). Following the explicit row enumeration in Section 2 (each path is one B row), this audit treats each kernel file as a separate target. The aggregate Type-1 count below covers the full set.

The audit's Section 5 line 481 ("31 B-classified") collapses the moreh `moreh_norm/{moreh_norm_h, moreh_norm_w, moreh_norm_other}` and `moreh_norm/ord_other/{moreh_norm_h, moreh_norm_nc, moreh_norm_w}` groups (6 sub-files into 1 B row each) and similarly bunches per-direction softmax/softmax_backward variants. The HQ-given count "31" is a directory-level B count; the per-file enumeration above produces 47 file-level entries. **All 47 files are audited below.** The Type-1 target list in Section 3 is keyed on file paths.

### Read protocol

For each kernel I:
1. Read the file (or a fixed window around the chain-region marker).
2. Located each block bounded by `tile_regs_acquire()` … `tile_regs_release()` (and any pre-acquire reconfig).
3. Located moreh-helper calls (`*_tile_to_cb`, `*_tiles_to_cb`, `power_tile_to_cb`, `mul_tiles_and_negative_to_cb`, etc.) — these expand to chain-shaped blocks underneath.
4. Located outer wait/pop / preamble / multi-stage scaffolding that surrounds the chain region.

For each block, classified per the type rubric below.

### Type definitions (refined while auditing)

- **Type 1.** Block is a sequential `[CB-reader…] → [DEST compute…] → [CB-writer]` shape that the helper already accepts. Migration is mechanical: name the chain elements, copy the CB ids and DEST slots, ship. Mapping element→helper: `CopyTile` (with Pinned/BlockIter/FirstTile mode), `BinaryFpu` (with `BroadcastDim::None|Col|Row|Scalar`), `DestReuseBinary`, `UnaryBcast`, `PackTile`, the SFPU op-struct families (`Exp`, `Log`, `Recip`, `Negative`, `Abs`, `Sqrt`, `Rsqrt`, `Sigmoid`, etc.), `OptionalChainElement<COND, Inner>`. Reconfig is fold-driven so the kernel doesn't need to spell `_with_dt` forms.
- **Type 2.** Block uses a pattern the helper does NOT support today. Concrete gaps observed:
  - **HELD-CB** — `cb_pop_front(X) + cb_push_back(X)` accumulator on the same CB.
  - **MULTI-DEST** — inner loop with `pack_tile(j, cb, j)` where j is both DEST slot and out-tile-index; chain uses one DEST slot per element.
  - **MASK-INJECT** — mid-loop conditional `mask_tile(dst, mask_dst)` after the SFPU op; needs a `MaskInject<COND, …>` chain element.
  - **MACRO-INJECT** — host-injected `BINARY_OP` / `PREPROCESS` / `BINARY_SFPU_OP` / `FILL_LLK` macros not modeled by the chain.
  - **HELD-DEST** — multi-stage compute inside one acquire window with intermediate DEST holds (e.g. `mul + recip + sub + recip`, `mul + sqrt + add + recip`).
  - **RUNTIME-CB** — block uses a runtime-deduced CB id (`uint32_t cb_x = …`); chain `BinaryFpu` requires constexpr CB ids.
  - **RUNTIME-POP** — `cb_pop_front(cb, runtime_pop_count)`; chain pops a constexpr count.
  - **TRANSPOSE/TOPK** — `transpose_wh`, `topk_local_sort`, `topk_merge` — separate helper area.
  - **TILIZE** — `untilize_block`, `tilize_block` — separate helper area.
  - **PACK-THREAD-SFPU** — SFPU-on-PACK pattern used by moe_compute / moe_gpt.
  - **OUTER-FREQ-LOOP** — `complete_iterations` / `tile_freq` / `tile_start` outer iteration with sub-stages (preamble + chain + post-PREPROCESS) — chain v1 only iterates a single linear count.
  - **MULTI-AXIS-EARLY-EXIT** — 3-/4-axis nested loops with `num_tiles_read += Wt - start_tw` style irregular increments.
  - **RUNTIME-BLOCK-SIZE** — `for j < num_tiles_per_cycle` inner loop where `num_tiles_per_cycle` is constexpr but the block-elements expect compile-time `BlockSize`. Often a packaging issue rather than a deep gap; counted Type-2 here because the inner-loop's per-iter pack-from-DEST-slot-j is multi-DEST (Type 2 anyway).
  - **REDUCE-FOLD** — `reduce_init/reduce_uninit/reduce_tile` state machine — separate `reduce_helpers_compute` area.
  - **POWER-MULTISTAGE** — `power_tile_to_cb` is itself a multi-stage moreh helper (log + mul + exp + cleanup); not a single chain.
- **Type 3.** Doesn't fit chain abstraction at all (covered in Section 5).

The fold-reconfig rules in `eltwise_chain.hpp` Section "Reconfig" mean every Type-1 migration auto-elides `reconfig_data_format_*` / `pack_reconfig_data_format` calls when prev_cb == curr_cb; the kernel author does NOT need to spell `_with_dt` forms.

### Helper feature reference (run7-refined HEAD)

Confirmed via reading the helper sources at HEAD `3b0cc6026e8`:
- `eltwise_chain.hpp` lines 560-620 — element decls (`CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`, fill / rand).
- `eltwise_chain.inl` lines 511-565 — `BinaryFpu::init/exec` supports `Bcast=None|Col|Row|Scalar` via `init_bcast<et,bt>` and `add/sub/mul_tiles_bcast<bt>` dispatch.
- `eltwise_predicates.hpp` — `UnaryNe/Eq/Gt/Ge/Lt/Le` runtime-param SFPU available (member `exec(uint32_t)` dispatch path confirmed at `eltwise_chain.inl:888,965`). Migration log's "UnaryNe runtime-param SFPU dispatch GAP" is closed in run7.
- `eltwise_optional.hpp` — `OptionalChainElement<COND, Inner>` with full inner-tag SFINAE forwarding.
- `eltwise_math.hpp` — `Exp`, `Log`, `Recip`, `Sqrt`, `Rsqrt` (3-arg `<Approx, Legacy, Dst>`), `Cbrt`, `Log1p`, `Power(uint32_t)`, `Rpow`.
- `eltwise_misc.hpp` — `Negative`, `Abs`, `Sign`, etc.
- `eltwise_activations.hpp` — `Sigmoid`, `Hardsigmoid`, `Hardswish`, `Tanhshrink`, etc.
- `eltwise_block.hpp` — `BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile` (compile-time `BlockSize`).
- `eltwise_fill.hpp` — `FillScalar`, `FillInt`, `FillBitcast`.

---

## Section 2 — Per-kernel inventory

### 2.1 binary_ng — kernels/ family

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`

**Current state.** 1 migrated chain region (lines 79-83, no-activations fast path: `BlockBinaryFpu + BlockPackTile`). 2 raw-LLK regions: activations branch (lines 48-77) and remainder (lines 86-104).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L48-77 | activations branch — host-injected `PREPROCESS` / `BINARY_OP` / `PROCESS_POST_ACTIVATIONS` macros, multi-tile DEST scratch with `BINARY_OP(…, i, i, i)` | 2 | (MACRO-INJECT + MULTI-DEST) | Same as design v6 Section E. |
| L86-104 | remainder block — same activations-style multi-tile DEST scratch but always raw | 2 | (MULTI-DEST) | Caller-managed; n is runtime; the per-iter `BINARY_OP(…, i, i, i)` puts each tile in its own DEST slot. |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng.py` (covers ELWADD/ELWSUB/ELWMUL with and without activations).

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

**Current state.** 1 migrated chain region (lines 200-209, no-activations stride-2 chain via local block elements). 1 raw-LLK function (`process_sfpu_tiles` at lines 140-179) used by activations branch and remainder.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L140-179 (`process_sfpu_tiles`) | macro-injected `BINARY_SFPU_OP`/`PREPROCESS`/`PROCESS_POST_ACTIVATIONS` over stride-2 multi-DEST scratch | 2 | (MACRO-INJECT + MULTI-DEST) | Local block elements (lines 43-136) already wrap this for the no-act path; activations / remainder cannot use them due to PREPROCESS. |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng.py` (sfpu binary, activations).

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`

**Current state.** 1 migrated chain region (lines 188-197, scalar RHS stride-2 chain). 1 raw `process_sfpu_scalar_tiles` (lines 129-164).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L129-164 (`process_sfpu_scalar_tiles`) | macro-injected `BINARY_SFPU_OP`/`PREPROCESS` over stride-2 scalar-RHS multi-DEST | 2 | (MACRO-INJECT + MULTI-DEST) | Same as `_no_bcast` variant. |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng.py` (sfpu binary scalar).

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu.cpp`

**Current state.** 1 migrated stride-3 chain region (lines 178-182 inside `run_iter`). Outer freq-loop framing (lines 184-189) raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L178-182 (inside `run_iter`) | already migrated | A | — | (Already in chain.) |
| L184-189 + outer iteration scaffolding | `complete_iterations` × `tile_freq` outer loop; `cb_wait_front(cb_bcast, …)` and `cb_pop_front(cb_bcast, …)` per outer iter | 2 | (OUTER-FREQ-LOOP) | Same as audit Section 3 disposition. |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_where.py`.

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu_scalar.cpp`

**Current state.** 1 migrated stride-3 chain region (lines 141-149). Single outer `cb_wait_front(cb_in1, …)` (line 139) raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L139 | once-per-kernel `cb_wait_front(cb_in1, num_tiles_per_cycle)` for the scalar | 3 | — | Caller-side broadcast wait; chain helper does not own this; legitimate raw scaffolding. |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_where.py` (scalar variant).

---

### 2.2 binary_ng — kernels_ng/ family (post-Q4 broadcast variants)

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_col_bcast.cpp`

**Current state.** 1 migrated stage (lines 75-83, no-activations `BlockBinaryFpu + BlockPackTile`). 1 raw activations branch (lines 84-97). `unary_bcast` preamble + `PREPROCESS` raw (lines 47-65).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L47-65 | `unary_bcast<COL>` preamble + first-tile `PREPROCESS` to bcast cb | 2 | (OUTER-FREQ-LOOP / preamble-as-element gap) | Could be expressed as a `UnaryBcast<COL>` chain element-as-preamble; not exposed as such today. |
| L84-97 | activations branch — `BINARY_OP` + `PROCESS_POST_ACTIVATIONS` | 2 | (MACRO-INJECT) | |

**Type 1 migration targets:** none.
**Estimated commit-touch:** 0.
**Known testing:** `tests/ttnn/unit_tests/operations/eltwise/test_binary_ng.py` (col-bcast).

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_bcast.cpp`

Same structure as `_col_bcast.cpp` with `BroadcastType::ROW` instead of COL. Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_col_bcast.cpp`

Same structure as `_col_bcast.cpp` with double broadcast (row+col). Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_scalar_bcast.cpp`

Same structure with scalar broadcast. Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_sfpu_row_bcast.cpp`

**Current state.** Stride-2 DEST scratch chain inner; activations branch + bcast preamble raw.

**Per-block classification:**

| Block | Pattern | Type | Notes |
|---|---|---|---|
| outer `unary_bcast<ROW>` preamble | (OUTER-FREQ-LOOP) | 2 | |
| activations branch | (MACRO-INJECT) | 2 | |
| inner stride-2 chain | already migrated | A | |

**Type 1 migration targets:** none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_where_sfpu_row_bcast.cpp`

Same shape as the SFPU row-bcast above with stride-3 inner; outer raw. Type 1 = none.

---

### 2.3 ternary — sfpu broadcast variants

#### `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp`

**Current state.** Inner ternary chain migrated (lines 105). Outer BCAST_A/B/C wait/pop raw (lines 81-128).

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L81-128 outer scope | `cb_wait_front(cb_in0/1/2, …) / cb_pop_front` BCAST-conditional | 2 | (OUTER-FREQ-LOOP) — chain doesn't own freq-driven outer iteration. |
| L105 inner | already migrated | A | |

**Type 1 migration targets:** none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp`

Same shape as `ternary_addc_ops_sfpu_bcast.cpp`. Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp`

Same shape — outer `unary_bcast` preamble + per-tile chain. Type 1 = none.

---

### 2.4 experimental/bcast_to/ — multi-axis early-exit

#### `ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_col_bcast_to.cpp`

**Current state.** Per-tile inner chain at lines 41-45 (`UnaryBcast<Col> + PackTile`). Outer (n,c,th) loop with irregular `num_tiles_read += Wt - start_tw` raw (lines 38-49).

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L38-49 outer 3-axis loop | irregular `num_tiles_read` increment | 2 | (MULTI-AXIS-EARLY-EXIT). |

**Type 1 migration targets:** none.

---

#### `ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_row_bcast_to.cpp`

Same — 4-axis outer loop. Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_scalar_bcast_to.cpp`

Same — 2-axis outer loop with `num_tiles_read += HtWt - start_t`. Type 1 = none.

---

### 2.5 experimental/ccl/moe — matmul + PACK-thread SFPU

#### `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp`

**Current state.** 1 migrated chain region (lines 163-167, ones-tile `FillScalar + PackTile`). Main matmul + SFPU-on-PACK SwiGLU/SiLU path raw.

**Per-block classification:**

| Block | Pattern | Type | Notes |
|---|---|---|---|
| matmul main loop | `matmul_block` | 3 | Out-of-scope (separate helper area). |
| SFPU-on-PACK SwiGLU/SiLU | PACK-thread SFPU | 2 | (PACK-THREAD-SFPU) — chain v1 explicitly out of scope. |

**Type 1 migration targets:** none.

---

#### `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/compute.cpp`

Same as moe_compute. Type 1 = none.

---

### 2.6 experimental/reduction/deepseek_grouped_gate

#### `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp`

**Current state.** `sigmoid` block migrated (lines 23-37). `add_bias` regressed to raw under v6 Q4 collapse (lines 39-68 — Type C in coverage_audit). `process_and_sort_tiles`, `topk*`, `normalize_scores`, `scale` blocks raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L23-37 sigmoid | already migrated | A | |
| L39-68 add_bias | `BinaryFpu<… AIndex=BlockIter, BIndex=FirstTile>` collapse-incompatible; per-tile add_tiles inside acquire/commit | 2 | v6 Q4 known regression. Future helper variant `BinaryFpuPerTileScalarB`. |
| L70-126 process_and_sort_tiles | `transpose_wh_tile` + `topk_local_sort` | 2/3 | (TRANSPOSE/TOPK). |
| L128-147 sum_top_experts_per_group | multi-write `add_tiles(…, i, i+1, 0)` accumulation into single DEST | 2 | (HELD-DEST). |
| L149-182 topk_group_scores | `topk_local_sort` + `acquire_dst()` | 3 | Deprecated dst-sync. |
| L184-203 transpose_and_pack | `transpose_wh_tile` | 2/3 | (TRANSPOSE). |
| L205-267 topk | `topk_local_sort` + `topk_merge` + `topk_rebuild` | 2/3 | (TOPK). |
| L269-324 normalize_scores | `reduce` + `add_tiles_bcast<SCALAR>` + `recip_tile` + `mul_tiles_bcast<COL>` | 2 | (HELD-DEST inside one acquire/commit + REDUCE-FOLD). |
| L326-343 scale | `mul_tiles_bcast_scalar` once | 1? | Single `BinaryFpu<Mul, Bcast=Scalar, cb_normalized_scores, cb_route_scale_scalar, cb_out_weights> + Pack`. **TYPE 1.** |

**Type 1 migration targets:** L326-343 scale block.

**Estimated commit-touch:** ~12 LOC removed, ~8 LOC added. 2 helper templates (`BinaryFpu<Mul, Bcast=Scalar>` + `PackTile`).

**Known testing:** `tests/ttnn/unit_tests/operations/experimental/reduction/test_deepseek_grouped_gate.py` (if exists, otherwise `tests/sweep_framework` deepseek_grouped_gate sweep).

---

### 2.7 experimental/ssm/prefix_scan

#### `ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp`

**Current state.** `mul`, `sum`, `copy` 1-tile chain helpers (lines 57-110) migrated. `pack_block_rows_into_tiles` (lines 17-32) and `pack_block_tiles_into_rows` (lines 37-52) use `untilize_block` / `tilize_block` raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L17-32 pack_block_rows_into_tiles | `untilize_block` | 3 | (TILIZE — out of helper scope). |
| L37-52 pack_block_tiles_into_rows | `tilize_block` | 3 | (TILIZE). |

**Type 1 migration targets:** none.

---

### 2.8 moreh — moreh_adam / moreh_adamw

#### `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp`

**Current state.** 17 chain calls via `moreh_bin_chain` / `moreh_copy_chain` wrappers (lines 22-77 define the wrappers; called throughout `kernel_main` lines 79-463). Several raw blocks remain.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping (Type 1 only) | Notes |
|---|---|---|---|---|
| L270-281 `cb_tmp1 = pow(beta2, step)` | `CopyTile(cb_scalar_args, idx=beta2_tile=2) + Power(step) + Pack(cb_tmp1)` | **1** | `CopyTile<cb_scalar_args, …, Pinned> + Power<>(step) + PackTile<cb_tmp1>` | Power has runtime exponent (member exec). Pinned a_tile_idx=2. |
| L284-298 `cb_tmp1 = 1 / (1 - cb_tmp1)` | held-CB recurrence on cb_tmp1 (pop+push same CB) plus multi-stage `sub + recip` inside one acquire | 2 | — | (HELD-CB + HELD-DEST). |
| L302-315 (#ifdef AMSGRAD) `tmp_cb_max_exp_avg_sq = max(…)` | multi-stage `copy + copy + binary_max_tile` inside one acquire (multi-input DEST) | 2 | — | (HELD-DEST — chain doesn't model `binary_max_tile(dst0, dst1, dst0)`). |
| L318-328 (#ifdef AMSGRAD) `cb_max_exp_avg_sq_out = tmp_cb_max_exp_avg_sq` | `CopyTile(tmp_cb_max_exp_avg_sq) + Pack(cb_max_exp_avg_sq_out)` | **1** | `CopyTile<tmp_cb_max_exp_avg_sq, …, FirstTile, Input> + PackTile<cb_max_exp_avg_sq_out, …, Output>` | Functionally identical to `moreh_copy_chain` already used elsewhere; trivially substitute. |
| L332-357 `cb_tmp1 = sqrt(exp_avg_sq * cb_tmp1)` | held-CB recurrence on cb_tmp1 + multi-stage `mul + sqrt` | 2 | — | (HELD-CB + HELD-DEST). |
| L360-374 `cb_tmp1 = 1 / (cb_tmp1 + eps)` | held-CB recurrence on cb_tmp1 + multi-stage `add + recip` | 2 | — | (HELD-CB + HELD-DEST). |
| L378-389 `cb_tmp2 = pow(beta1, step)` | same as L270-281 (different scalar tile) | **1** | `CopyTile<cb_scalar_args, …, Pinned> + Power<>(step) + PackTile<cb_tmp2>` | A-side tile_idx=beta1_tile=1. |
| L392-406 `cb_tmp2 = 1 / (1 - cb_tmp2)` | held-CB recurrence on cb_tmp2 + multi-stage `sub + recip` | 2 | — | (HELD-CB + HELD-DEST). |

**Type 1 migration targets:** L270-281, L318-328 (only when AMSGRAD), L378-389 (3 blocks).

**Estimated commit-touch:** ~36 LOC removed, ~24 LOC added. 3 instantiations of `CopyTile + (Power|nothing) + PackTile`.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_adam.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp`

Closely parallel to moreh_adam. Same `moreh_bin_chain` / `moreh_copy_chain` wrappers. Differences: bias_correction CBs come from host already-computed (`cb_beta1_exponent`, `cb_beta2_exponent`) so no `power_tile` block here. Same held-CB `1/(1-x)`, `sqrt(mul)`, `1/(x+eps)` blocks.

**Per-block classification (raw blocks only):**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L281-292 `cb_tmp1 = 1 / (1 - cb_beta2_exponent)` | `BinaryFpu(Sub, cb_one, cb_beta2_exponent) + Recip + Pack(cb_tmp1)` (single-acquire-window 2-stage compute) | **1** | `BinaryFpu<cb_one, cb_beta2_exponent, cb_tmp1, Sub> + Recip<> + PackTile<cb_tmp1, …, Output>` | NO held-CB here (line 281 only `cb_reserve_back`, no pop). Sub-then-Recip in one chain works. |
| L294-309 (#ifdef AMSGRAD) `tmp_cb_max_exp_avg_sq = max(…)` | multi-input DEST with `binary_max_tile` | 2 | — | (HELD-DEST). |
| L312 (#ifdef AMSGRAD) `moreh_copy_chain<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out>` | already migrated | A | — | |
| L315-338 `cb_tmp1 = sqrt(exp_avg_sq * cb_tmp1)` | held-CB recurrence on cb_tmp1 + multi-stage `mul + sqrt` | 2 | — | (HELD-CB + HELD-DEST). |
| L340-354 `cb_tmp1 = 1 / (cb_tmp1 + eps)` | held-CB recurrence on cb_tmp1 + multi-stage `add + recip` | 2 | — | (HELD-CB + HELD-DEST). |
| L359-371 `cb_tmp2 = 1 / (1 - cb_beta1_exponent)` | `BinaryFpu(Sub, cb_one, cb_beta1_exponent) + Recip + Pack(cb_tmp2)` (single-acquire-window 2-stage compute) | **1** | `BinaryFpu<cb_one, cb_beta1_exponent, cb_tmp2, Sub> + Recip<> + PackTile<cb_tmp2, …, Output>` | No held-CB here. |

**Type 1 migration targets:** L281-292, L359-371 (2 blocks).

**Estimated commit-touch:** ~30 LOC removed, ~16 LOC added. 2 instantiations of `BinaryFpu + Recip + PackTile`.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_adamw.py`.

---

### 2.9 moreh — clip_grad_norm_step1

#### `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/moreh_clip_grad_norm_step1_kernel.cpp`

**Current state.** Seed cb_xpowadd chain at lines 95-101. `|x|` mask block (lines 59-90) raw (mid-loop mask). `power_tile_to_cb` (line 93) raw (multi-stage moreh helper). Add-tiles fold (lines 102-119) raw (held-CB).

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L59-90 `\|x\|` block | mid-loop conditional `mask_tile` (`do_mask_h && need_to_do_mask_h(...)`) | 2 | (MASK-INJECT). |
| L93 `power_tile_to_cb` | multi-stage moreh helper | 2 | (POWER-MULTISTAGE). |
| L95-101 seed chain | already migrated | A | |
| L102-119 add fold on cb_xpowadd | `cb_pop_front(cb_xpowadd) + cb_push_back(cb_xpowadd)` | 2 | (HELD-CB). |

**Type 1 migration targets:** none.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_clip_grad_norm.py`.

---

### 2.10 moreh — layer_norm forward (large + small)

#### `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp`

**Current state.** 1 migrated chain (lines 305-325, `BinaryFpu(Add) + Rsqrt + Pack` for `cb_recip_std`). Remainder raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L82-111 (w_idx==0 seed branch) | `CopyTile + (mask_tile?) + Pack` to cb_xsum | 2 | (MASK-INJECT — conditional). |
| L112-160 (w_idx>0 add fold) | held-CB on cb_xsum + mid-loop mask | 2 | (HELD-CB + MASK-INJECT). |
| L173-188 cb_mean copy | `CopyTile(cb_ex, …, is_lastdim_layernorm) + Pack(cb_mean)` | 2 | The `copy_tile_init_with_dt(cb_ex, is_lastdim_layernorm)` second arg (a `bool` for transpose-of-input) is a moreh-specific overload not modeled by the chain. (TRANSPOSE-LIKE / non-standard init.) |
| L195-232 xmm bcast block | `sub_tiles_bcast_*(cb_x, cb_ex, w_idx, 0, j)` inside `j < block_size` inner loop with mid-loop mask AND multi-DEST (j=DEST slot) | 2 | (MULTI-DEST + MASK-INJECT). |
| L240-287 xmm2sum fold | held-CB on cb_xmm2sum (lines 281-285 pop+push) | 2 | (HELD-CB). |
| L305-325 rsqrt(var+eps) chain | already migrated | A | |
| L327-342 cb_rstd copy | similar to L173-188 with `is_lastdim_layernorm` overload | 2 | Same. |
| L350-367 (cb_xmm * cb_recip_std → cb_gamma_beta_or_out) | `mul_tiles_bcast_*(cb_xmm, cb_recip_std, inner_idx+j, 0, j)` block inside `j < block_size` with multi-DEST | 2 | (MULTI-DEST). |
| L370-398 gamma mul block | per-`j` multi-DEST iteration | 2 | (MULTI-DEST). |
| L400-428 beta add block | per-`j` multi-DEST iteration | 2 | (MULTI-DEST). |

**Type 1 migration targets:** none.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_layer_norm.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp`

Same shape as small — 1 migrated rsqrt chain (lines 336-342) + 1 migrated seed-cb_xmm2sum chain. Remainder uses same patterns: held-CB on cb_xmm2sum, multi-DEST inner loops over `j < block_size`, mid-loop mask, `_with_dt(…, is_lastdim_layernorm)` non-standard init.

**Type 1 migration targets:** none.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_layer_norm.py`.

---

### 2.11 moreh — layer_norm backward family

#### `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`

**Current state.** Several migrated chains: dycopy seed (lines 124-130 in beta_grad branch), cb_ydy mul (lines 215-235 in gamma_grad branch), final dgamma/dbeta copies (lines 286-292, 308-314).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping (Type 1) | Notes |
|---|---|---|---|---|
| L82-113 cb_dycopy block | `CopyTile + (mask_tile?) + Pack` mid-loop conditional mask | 2 | — | (MASK-INJECT). |
| L131-146 dyadd fold | held-CB on cb_dyadd | 2 | — | (HELD-CB). |
| L150-189 cb_xmm bcast sub-with-mask block | `sub_tiles_bcast_* + (mask_tile?) + pack` | 2 | — | (MASK-INJECT). |
| L191-213 cb_y mul block (no mask in this stage) | `mul_tiles_bcast_*(cb_xmm, cb_rstd) + Pack(cb_y)` (no mask, no held-CB) | **1** | `BinaryFpu<cb_xmm, cb_rstd, cb_y, Mul, BroadcastDim::Cols/Scalar, Reconfig::Input, WaitAndPop, WaitAndPop, FirstTile> + PackTile<cb_y, …, Output>` | bcast dim is Cols if `is_lastdim_layernorm` else Scalar. |
| L215-235 cb_y * cb_dycopy chain | already migrated | A | — | |
| L237-268 ydyadd fold | held-CB on cb_ydyadd | 2 | — | (HELD-CB). |
| L286-292 dgamma copy chain | already migrated | A | — | |
| L308-314 dbeta copy chain | already migrated | A | — | |

**Type 1 migration targets:** L191-213.

**Estimated commit-touch:** ~22 LOC removed, ~12 LOC added. 1 `BinaryFpu + PackTile` chain.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_layer_norm_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp`

**Current state.** Multiple migrated chains: cb_recip_nrstd (lines 75-96), seed dyadd / ydyadd, final cb_tmp1 sub (lines 421-441), cb_dx mul (lines 443-463).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L75-96 cb_recip_nrstd | already migrated | A | — | |
| L101-138 cb_xmm sub-with-mask | `sub_tiles_bcast_* + mid-loop mask` | 2 | — | (MASK-INJECT). |
| L140-159 cb_y mul block (no mask) | `mul_tiles_bcast_*(cb_xmm, cb_rstd) + Pack(cb_y)` | **1** | `BinaryFpu<cb_xmm, cb_rstd, cb_y, Mul, BroadcastDim::Cols/Scalar> + PackTile<cb_y, …, Output>` | |
| L162-211 cb_dycopy with mask | mid-loop mask | 2 | — | (MASK-INJECT). |
| seed dyadd / ydyadd chains | already migrated | A | — | |
| L255-307 dyadd / ydyadd fold | held-CB | 2 | — | (HELD-CB). |
| L364-375 cb_ndy = cb_n_recip_n[0] * cb_dycopy[wt] (idx wt) | `BinaryFpu<Mul, A=Pinned(0), B=BlockIter(wt)>` — but wt is loop-var of an outer Wt loop | 2 | — | After Q4 collapse, only single `Index` mode. A-side wants pinned 0; B-side wants per-iter wt (BlockIter) — asymmetric. Same regression family as deepseek. |
| L377-398 cb_ndymdysum = cb_ndy - cb_dysum bcast (no mask, no held-CB) | `BinaryFpu(Sub, Bcast=Cols/Scalar, cb_ndy, cb_dysum) + Pack(cb_ndymdysum)` | **1** | `BinaryFpu<cb_ndy, cb_dysum, cb_ndymdysum, Sub, BroadcastDim::Cols/Scalar> + PackTile<cb_ndymdysum, …, Output>` | |
| L400-419 cb_yydysum = cb_y[wt] * cb_ydysum bcast | `BinaryFpu(Mul, Bcast=Cols/Scalar, cb_y(idxA=wt), cb_ydysum)` — A-side index is wt (loop var), B-side is FirstTile | 2 | — | After Q4 collapse, no symmetric mode covers `A=BlockIter, B=FirstTile`. (Q4-REGRESSION.) |
| L421-441 cb_tmp1 = cb_ndymdysum - cb_yydysum | already migrated | A | — | |
| L443-463 cb_dx = cb_tmp1 * cb_recip_nrstd | already migrated | A | — | |

**Type 1 migration targets:** L140-159, L377-398 (2 blocks).

**Estimated commit-touch:** ~40 LOC removed, ~22 LOC added. 2 `BinaryFpu(bcast) + PackTile` chains.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_layer_norm_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp`

**Current state.** Same family of patterns as `_small_kernel.cpp`. Multiple migrated chains: cb_recip_nrstd, seed dyadd / ydyadd, final cb_tmp1 sub, cb_dx mul. Several raw bcast / mask / held-CB blocks.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L77-98 cb_xmm sub bcast (no mask) | `sub_tiles_bcast_*(cb_x, cb_mean) + Pack(cb_xmm)` | **1** | `BinaryFpu<cb_x, cb_mean, cb_xmm, Sub, Bcast=Cols/Scalar> + PackTile` | |
| L100-136 cb_y = cb_xmm * cb_rstd with mask | mid-loop mask | 2 | — | (MASK-INJECT). |
| L138-208 cb_dycopy with mask | mid-loop mask | 2 | — | (MASK-INJECT). |
| seed dyadd / ydyadd chains (lines 228, 267, 283, 308, 324) | already migrated | A | — | |
| L260-272 ydy = cb_y * cb_dycopy (already migrated chain at L267-272) | already migrated | A | — | |
| L289-303 dyadd fold | held-CB | 2 | — | (HELD-CB). |
| L330-355 ydyadd fold | held-CB | 2 | — | (HELD-CB). |
| L347-353 cb_ndy = cb_n_recip_n[0] * cb_dycopy (no idx complication, no mask) | `BinaryFpu<Mul, A=Pinned(0), B=WaitAndPop>` — both at index 0 actually | check | Need closer look. Audit conservative: may be Type 1 if A=Pinned(0)+B=FirstTile is uniform `Pinned`. |
| L451-477 cb_ndymdysum = cb_ndy - cb_dysum (no mask) | `BinaryFpu(Sub, Bcast=Cols/Scalar) + Pack` | **1** | Same shape as `_small_kernel` L377-398. |
| L478-512 cb_xmm sub-with-mask (second occurrence) | mid-loop mask | 2 | — | |
| L514-540 cb_y = cb_xmm * cb_rstd (no mask, second occurrence) | `BinaryFpu(Mul, Bcast=Cols/Scalar) + Pack(cb_y)` | **1** | Same as L100-136 but no mask path here. |
| L542-562 cb_yydysum = cb_y * cb_ydysum bcast | `BinaryFpu<Mul, Bcast=Cols/Scalar>` with same Q4 asymmetric concern (A-side has tile_idx=0, B=0 → uniform Pinned works) | **1** | A and B both index 0. Compatible with `Pinned`. |
| L575-609 final cb_dx blocks | already migrated chains | A | — | |

**Type 1 migration targets:** L77-98, L451-477, L514-540, L542-562 (4 blocks).

**Estimated commit-touch:** ~80 LOC removed, ~50 LOC added. 4 `BinaryFpu(bcast) + PackTile` chains.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_layer_norm_backward.py`.

---

### 2.12 moreh — moreh_mean_nc

#### `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp`

**Current state.** 1 migrated chain (lines 65-83, `BinaryFpu(Mul, Bcast=Scalar, cb_intermed0, cb_scalar) + Pack(cb_out0)`). Lines 32-60 enable_reload accumulator raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L32-60 enable_reload accumulator | runtime `cb_add = enable_reload ? cb_intermed0 : cb_in1` (RUNTIME-CB) + held-CB recurrence on cb_intermed0 | 2 | (RUNTIME-CB + HELD-CB). |
| L65-83 stage 2 chain | already migrated | A | — |

**Type 1 migration targets:** none.
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_mean.py`.

---

### 2.13 moreh — nll_loss / nll_loss_backward

#### `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/moreh_nll_loss_step2_kernel.cpp`

**Current state.** 1 migrated chain (lines 32-40, `cb_divisor_recip = 1/cb_divisor`).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L32-40 cb_divisor_recip | already migrated | A | — | |
| L43-61 negative block | `CopyTile(cb_tmp_input) + Negative + Pack(cb_tmp1)` | **1** | `CopyTile<cb_tmp_input, …, WaitAndPop> + Negative<> + PackTile<cb_tmp1>` | |
| L63-80 (#ifdef WEIGHT) cb_tmp1 * cb_tmp_weight | `BinaryFpu(Mul, cb_tmp1, cb_tmp_weight) + Pack(cb_tmp3)` | **1** | `BinaryFpu<cb_tmp1, cb_tmp_weight, cb_tmp3, Mul> + PackTile<cb_tmp3>` | |
| L82-97 (#ifdef DIVISOR) cb_tmp3 * cb_divisor_recip | `BinaryFpu(Mul, Bcast=Scalar, cb_tmp3, cb_divisor_recip) + Pack(cb_output)` | **1** | `BinaryFpu<cb_tmp3, cb_divisor_recip, cb_output, Mul, BroadcastDim::Scalar> + PackTile<cb_output>` | |
| L99-103 (no DIVISOR, has WEIGHT) cb_output write | one-line `pack_tile_with_dt(dst0, cb_output)` of cb_tmp1*cb_tmp_weight DEST | 2 | — | DEST already populated by L63-80 — held-DEST cross-block. (Migration of L63-80 makes this block's pack happen inside the chain — block disappears.) |
| L106-130 (no WEIGHT, has DIVISOR) cb_tmp1 * cb_divisor_recip | `BinaryFpu(Mul, Bcast=Scalar) + Pack(cb_output)` | **1** | Same as L82-97. |
| L132-137 (no WEIGHT, no DIVISOR) cb_output write | DEST already populated by L43-61 — disappears when L43-61 migrates. | — | — | |

**Type 1 migration targets:** L43-61, L63-80, L82-97, L106-130 (4 blocks; structurally redundant — when migrated together they replace the entire `for (b)` body).

**Estimated commit-touch:** ~80 LOC removed, ~30 LOC added. Equivalent rewrite as 1-2 chain calls per `b` iter.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_nll_loss.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp`

**Current state.** 1 migrated chain (lines 36-41 cb_tmp1 = 1/cb_divisor under DIVISOR).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L36-41 cb_tmp1 chain | already migrated | A | — | |
| L48-63 (#ifdef DIVISOR) `cb_tmp_weight * cb_output_grad → -result → cb_tmp2` | `BinaryFpu(Mul, Bcast=Scalar, cb_tmp_weight, cb_output_grad) + Negative + Pack(cb_tmp2)` | **1** | `BinaryFpu<cb_tmp_weight, cb_output_grad, cb_tmp2, Mul, BroadcastDim::Scalar> + Negative<> + PackTile<cb_tmp2>` | |
| L65-79 (#ifdef DIVISOR) `cb_tmp2 * cb_tmp1 → cb_input_grad` | `BinaryFpu(Mul, Bcast=Scalar, cb_tmp2, cb_tmp1) + Pack(cb_input_grad)` | **1** | `BinaryFpu<cb_tmp2, cb_tmp1, cb_input_grad, Mul, BroadcastDim::Scalar, …, NoWaitNoPop on B> + PackTile<cb_input_grad>` | B is held — `NoWaitNoPop` policy. |
| L82-100 (no DIVISOR) `cb_tmp_weight * cb_output_grad → -result → cb_input_grad` | same as L48-63 but pack to cb_input_grad | **1** | Same shape. |

**Type 1 migration targets:** L48-63, L65-79, L82-100 (3 blocks).

**Estimated commit-touch:** ~70 LOC removed, ~30 LOC added. 3 chain calls.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_nll_loss.py` (backward).

---

### 2.14 moreh — moreh_norm (3 + 3 = 6 kernels)

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp`

**Current state.** 1 migrated chain (lines 95-105, seed cb_xpowadd when row_idx==0). `|x|` mask block + `power_tile_to_cb` + add fold raw.

**Per-block classification:**

| Block | Pattern | Type | Notes |
|---|---|---|---|
| L62-89 `\|x\|` block | mid-loop conditional `mask_tile` | 2 | (MASK-INJECT). |
| L90 `power_tile_to_cb` | multi-stage moreh helper | 2 | (POWER-MULTISTAGE). |
| L99-104 seed chain | already migrated | A | |
| L106-123 add fold | held-CB on cb_xpowadd | 2 | (HELD-CB). |
| L129 final `power_tile_to_cb(cb_xpowsum, …, cb_y, recip_p, recip_p_is_negative)` | multi-stage | 2 | (POWER-MULTISTAGE). |

**Type 1 migration targets:** none.
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_norm.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp`

**Current state.** 2 migrated chains: `|x|` chain (lines 58-65, no mask in this kernel since dim != H/W), seed chain (lines 76-82). Add fold raw.

**Per-block classification:**

| Block | Pattern | Type | Notes |
|---|---|---|---|
| L58-65 `\|x\|` chain | already migrated | A | |
| L67 `power_tile_to_cb` | multi-stage moreh helper | 2 | (POWER-MULTISTAGE). |
| L76-82 seed chain | already migrated | A | |
| L83-100 add fold | held-CB on cb_xpowadd | 2 | (HELD-CB). |
| L104 final `power_tile_to_cb` | multi-stage | 2 | (POWER-MULTISTAGE). |

**Type 1 migration targets:** none.
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_norm.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`

Same shape as `moreh_norm_h_kernel.cpp` with mask_w instead of mask_h. Type 1 = none.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp`

**Current state.** 2 migrated chains: seed (lines 87-99, copy cb_val→cb_cal), post-reduce write-out (lines 134-152).

**Per-block classification:**

| Block | Pattern | Type | Notes |
|---|---|---|---|
| L46-84 f(x) block | mid-loop `mask_tile` (or `mask_posinf_tile` under MINUS_INF) | 2 | (MASK-INJECT) — also non-standard `mask_posinf_tile`. |
| L87-99 seed chain | already migrated | A | |
| L100-127 add/binary_max fold | held-CB on cb_cal + `binary_max_tile` (no helper struct) | 2 | (HELD-CB + missing-helper for binary_max_tile). |
| L134-152 post-reduce write-out | already migrated | A | |

**Type 1 migration targets:** none.
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_norm.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp`

**Current state.** 4 migrated chains: f(x) chain for !IS_ZERO (lines 66-82), seed cb_cal (lines 92-98), post-loop write-out (lines 134-148). IS_ZERO branch raw.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L39-60 IS_ZERO branch | `CopyTile + UnaryNe(0) + (Negative if MINUS_INF) + Pack(cb_val)` | **1** | `CopyTile<cb_x, …, WaitAndPop> + UnaryNe<>{0u} + (OptionalChainElement<MINUS_INF, Negative<>>) + PackTile<cb_val>` | Migration log says "UnaryNe runtime-param dispatch GAP" but `eltwise_predicates.hpp:61` defines UnaryNe with member-exec dispatch and `eltwise_chain.inl:888,965` confirms the dispatch path is live. The "GAP" was closed in run7. |
| L66-82 !IS_ZERO chain | already migrated | A | — | |
| L92-98 seed chain | already migrated | A | — | |
| L100-126 fold (add or binary_max_tile) | held-CB on cb_cal | 2 | — | (HELD-CB). |
| L134-148 post-loop write-out | already migrated | A | — | |

**Type 1 migration targets:** L39-60 (IS_ZERO branch under !IS_ZERO `#ifdef`).

**Estimated commit-touch:** ~22 LOC removed, ~12 LOC added. 1 chain instantiation per (`MINUS_INF` × `IS_ZERO`) macro permutation. Confirmation needed: `static_assert` UnaryNe DispatchPath (member exec) routes correctly under run7 chain dispatcher; if regression detected, drop this Type-1 candidate to Type 2.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_norm.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`

Same shape as `ord_other/moreh_norm_h_kernel.cpp` with mask_w. Type 1 = none.

---

### 2.15 moreh — moreh_sgd

#### `ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/moreh_sgd.cpp`

**Current state.** 6 migrated `moreh_bin_chain` calls. 5 raw `mul_tiles_to_cb` / `add_tiles_to_cb` / `copy_tile_to_cb` calls (lines 135, 161, 166, 168, 185).

**Per-block classification:**

| Block (line range) | Pattern | Type | Notes |
|---|---|---|---|
| L135 `mul_tiles_to_cb(cb_grad_tmp, cb_tmp1, cb_tmp3, …)` | runtime cb_grad_tmp (set conditionally on lines 95, 118, 122, 158) | 2 | (RUNTIME-CB). |
| L161 `copy_tile_to_cb(cb_momentum_tmp, cb_momentum_out, 0, /*pop=*/0)` | runtime cb_momentum_tmp | 2 | (RUNTIME-CB). |
| L166 `mul_tiles_to_cb(cb_momentum_tmp, …, /*pop0=*/pop_momentum, /*pop1=*/0)` | runtime cb_momentum_tmp + runtime pop_momentum | 2 | (RUNTIME-CB + RUNTIME-POP). |
| L168 `add_tiles_to_cb(cb_tmp3, cb_grad_tmp, cb_tmp4, …)` | runtime cb_grad_tmp | 2 | (RUNTIME-CB). |
| L185 `mul_tiles_to_cb(cb_scalar_args, cb_grad_tmp, cb_tmp3, lr_tile, 0, …)` | runtime cb_grad_tmp | 2 | (RUNTIME-CB). |

**Type 1 migration targets:** none.
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_sgd.py`.

---

### 2.16 moreh — softmax (3 kernels)

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp`

**Current state.** 12 migrated chains via `moreh_bin_chain` / `moreh_copy_chain`. Raw blocks: `binary_max_tile` block (lines 88-112) and `exp_tile_to_cb` / `rexp_tile_to_cb` / `log_tile_to_cb` / `recip_tile_to_cb` / `mask_tile_to_cb` / `exp_tile_and_mask_tile_to_cb` / `rexp_tile_and_mask_tile_to_cb` calls.

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L86-86 (i==0 seed) `moreh_copy_chain<cb_in0, cb_max>` | already migrated | A | — | |
| L88-112 (i>0) `binary_max_tile` between cb_in0 and cb_max with held-CB on cb_max | held-CB + `binary_max_tile(dst0, dst1, dst0)` (no helper) | 2 | — | (HELD-CB + missing-helper). |
| L128 (#ifdef SOFTMAX) `exp_tile_to_cb(cb_tmp, cb_exps)` | `CopyTile(cb_tmp) + Exp + Pack(cb_exps)` | **1** | `CopyTile<cb_tmp, …, WaitAndPop> + Exp<> + PackTile<cb_exps>` | |
| L140 (else) `rexp_tile_to_cb(cb_tmp, cb_exps)` | `CopyTile(cb_tmp) + Negative + Exp + Pack(cb_exps)` | **1** | `CopyTile + Negative<> + Exp<> + PackTile` | |
| L160 (#ifdef LOG) `log_tile_to_cb(cb_add, cb_recipsumexps)` | `CopyTile(cb_add) + Log + Pack(cb_recipsumexps)` | **1** | `CopyTile + Log<> + PackTile` | |
| L163 (else) `recip_tile_to_cb(cb_add, cb_recipsumexps)` | `CopyTile(cb_add) + Recip + Pack(cb_recipsumexps)` | **1** | `CopyTile + Recip<> + PackTile` | |
| L168-end final result moreh_bin_chains | already migrated | A | — | |

**Type 1 migration targets:** L128, L140, L160, L163 (4 blocks; per-#ifdef paths).

**Estimated commit-touch:** Trivial wrappers around moreh helpers; could be addressed by introducing local `moreh_unary_chain<Sfpu, CbIn, CbOut>` helper or inlining. ~12-20 LOC delta.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp`

**Current state.** Many migrated `moreh_bin_chain` (Sub Bcast=Row, Mul Bcast=Row, Add) and `moreh_copy_chain` calls. Raw `exp_tile_to_cb`, `rexp_tile_to_cb`, `mask_tile_to_cb`, `exp_tile_and_mask_tile_to_cb`, `rexp_tile_and_mask_tile_to_cb`. Reduce raw (lines 91-107, 191-216).

**Per-block classification:**

| Block | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L89, 117, 134 `mask_tile_to_cb` calls | mask injection | 2 | — | (MASK-INJECT). |
| L125, 133, 159 `exp_tile_and_mask_tile_to_cb` / `rexp_tile_and_mask_tile_to_cb` calls | mask injection inside exp/rexp chain | 2 | — | (MASK-INJECT). |
| L156, 169 `exp_tile_to_cb` / `rexp_tile_to_cb` | `CopyTile + Exp + Pack` | **1** | `CopyTile + Exp<> + PackTile` (or `+ Negative<>` for rexp) | |
| L263, 288 `exp_tile_to_cb` / `rexp_tile_to_cb` | same | **1** | same | |
| L91-107, 95-107, 191-216 reduce calls | `compute_kernel_lib::reduce<>` | A | (already in helper) | |
| moreh_bin_chain calls | already migrated | A | — | |

**Type 1 migration targets:** L156, L169, L263, L288 (4 occurrences of `exp_tile_to_cb` / `rexp_tile_to_cb`).

**Estimated commit-touch:** ~28 LOC; introduce `moreh_unary_chain<Op, CbIn, CbOut>(idx, pop)` wrapper if convenient.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`

Mirror of `moreh_softmax_h_large.cpp` with `BroadcastDim::Col` instead of Row.

**Type 1 migration targets:** L190, L203, L297, L322 (4 occurrences of `exp_tile_to_cb` / `rexp_tile_to_cb`).
**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax.py`.

---

### 2.17 moreh — softmax_backward (5 kernels)

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp`

**Current state.** 11 migrated chains. Raw: `exp_tile_to_cb` (line 102) and `mul_tiles_and_negative_to_cb` (line 182).

**Per-block classification:**

| Block (line range) | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L102 `exp_tile_to_cb(cb_y, cb_exp)` | `CopyTile + Exp + Pack` | **1** | `CopyTile<cb_y, …, NoWaitPop or WaitNoPop based on outer scope> + Exp<> + PackTile<cb_exp>` | |
| L182 `mul_tiles_and_negative_to_cb(cb_dy_m_sum, cb_y, cb_dx)` | `BinaryFpu(Mul) + Negative + Pack` | **1** | `BinaryFpu<cb_dy_m_sum, cb_y, cb_dx, Mul> + Negative<> + PackTile<cb_dx>` | |

**Type 1 migration targets:** L102, L182 (2 blocks).

**Estimated commit-touch:** ~14 LOC removed, ~10 LOC added. 2 chain calls.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h.cpp`

**Current state.** 4 migrated `moreh_bin_chain` + 2 bcast_rows chains via `moreh_bin_chain_rt`. Raw: `mask_tile_to_cb`, `mul_tiles_and_mask_tile_to_cb`, `exp_tile_to_cb`, `mul_tiles_and_negative_to_cb`.

**Per-block classification:**

| Block | Pattern | Type | Helper mapping | Notes |
|---|---|---|---|---|
| L95, L106 `mask_tile_to_cb` | (MASK-INJECT) | 2 | — | |
| L129 `exp_tile_to_cb(cb_y, cb_exp, w, /*dst=*/0, /*pop=*/0)` | `CopyTile<cb_y, …, NoWaitNoPop or NoWaitPop, Pinned> + Exp + Pack(cb_exp)` with cb_tile_idx=w | **1** | `CopyTile<cb_y, …, NoWaitNoPop, Pinned, Input> + Exp<> + PackTile<cb_exp>` with `cb_tile_idx = w` (runtime). | The CopyTile element exposes `cb_tile_idx` as a runtime member field (mirror of moreh_copy_chain pattern). |
| L161, L162 `mul_tiles_and_mask_tile_to_cb` | (MASK-INJECT inside mul) | 2 | — | |
| L177-178 `reduce<SUM, REDUCE_COL, BulkWaitBulkPop>` | already in helper | A | — | |
| L204 `mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx, h, 0, /*pop0=*/0, /*pop1=*/1)` | `BinaryFpu(Mul, A=Pinned(h), B=Pinned(0)) + Negative + Pack` (A-side has runtime tile index h; both pinned) | **1** | `BinaryFpu<cb_y, cb_inter2, cb_dx, Mul, BroadcastDim::None, …, NoWaitNoPop on A, WaitAndPop on B, Pinned> + Negative<> + PackTile<cb_dx>` with `a_tile_idx=h, b_tile_idx=0`. | |

**Type 1 migration targets:** L129, L204 (2 blocks).

**Estimated commit-touch:** ~14 LOC removed, ~10 LOC added.

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp`

Same family as `_h.cpp`. Raw: `mask_tile_to_cb`, `mul_tiles_and_mask_tile_to_cb`, `exp_tile_to_cb` (line 131), `mul_tiles_and_negative_to_cb` (line 227).

**Type 1 migration targets:** L131, L227 (2 blocks).

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w.cpp`

Mirror of `_h.cpp` with `BroadcastDim::Col`. Raw: line 128 `exp_tile_to_cb(cb_y, cb_exp, w, …)`, line 203 `mul_tiles_and_negative_to_cb`.

**Type 1 migration targets:** L128, L203 (2 blocks).

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax_backward.py`.

---

#### `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp`

Mirror of `_h_large.cpp`. Raw: line 131 `exp_tile_to_cb(cb_y, cb_exp, 0)`, line 225 `mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx)`.

**Type 1 migration targets:** L131, L225 (2 blocks).

**Known testing:** `tests/ttnn/unit_tests/operations/moreh/test_moreh_softmax_backward.py`.

---

## Section 3 — Aggregated Type-1 migration target list

Ordered by (a) lowest risk first (kernels with one Type-1 block, no other complexity), (b) smaller LOC, (c) better pytest coverage.

| ID | Kernel | Block (line range) | Helper pattern | Migration cost (LOC ~) | Pytest |
|----|--------|--------------------|----------------|-------------------------|--------|
| T1.01 | `moreh_softmax_backward/.../moreh_softmax_backward_c_large.cpp` | L102 | `CopyTile + Exp + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.02 | `moreh_softmax_backward/.../moreh_softmax_backward_c_large.cpp` | L182 | `BinaryFpu(Mul) + Negative + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.03 | `moreh_softmax_backward/.../moreh_softmax_backward_h.cpp` | L129 | `CopyTile<…, Pinned, Input> + Exp + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.04 | `moreh_softmax_backward/.../moreh_softmax_backward_h.cpp` | L204 | `BinaryFpu(Mul, Pinned, NoWaitNoPop A) + Negative + PackTile` | -8/+6 | `test_moreh_softmax_backward.py` |
| T1.05 | `moreh_softmax_backward/.../moreh_softmax_backward_h_large.cpp` | L131 | `CopyTile + Exp + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.06 | `moreh_softmax_backward/.../moreh_softmax_backward_h_large.cpp` | L227 | `BinaryFpu(Mul) + Negative + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.07 | `moreh_softmax_backward/.../moreh_softmax_backward_w.cpp` | L128 | `CopyTile<…, Pinned, Input> + Exp + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.08 | `moreh_softmax_backward/.../moreh_softmax_backward_w.cpp` | L203 | `BinaryFpu(Mul, Pinned, NoWaitNoPop A) + Negative + PackTile` | -8/+6 | `test_moreh_softmax_backward.py` |
| T1.09 | `moreh_softmax_backward/.../moreh_softmax_backward_w_large.cpp` | L131 | `CopyTile + Exp + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.10 | `moreh_softmax_backward/.../moreh_softmax_backward_w_large.cpp` | L225 | `BinaryFpu(Mul) + Negative + PackTile` | -7/+5 | `test_moreh_softmax_backward.py` |
| T1.11 | `moreh_softmax/.../moreh_softmax_c_large.cpp` | L128 (SOFTMAX) | `CopyTile + Exp + PackTile` | -3/+5 | `test_moreh_softmax.py` |
| T1.12 | `moreh_softmax/.../moreh_softmax_c_large.cpp` | L140 (else, SOFTMIN) | `CopyTile + Negative + Exp + PackTile` | -3/+6 | `test_moreh_softmax.py` |
| T1.13 | `moreh_softmax/.../moreh_softmax_c_large.cpp` | L160 (LOG) | `CopyTile + Log + PackTile` | -3/+5 | `test_moreh_softmax.py` |
| T1.14 | `moreh_softmax/.../moreh_softmax_c_large.cpp` | L163 (else) | `CopyTile + Recip + PackTile` | -3/+5 | `test_moreh_softmax.py` |
| T1.15 | `moreh_softmax/.../moreh_softmax_h_large.cpp` | L156 / L169 / L263 / L288 | `CopyTile + Exp + PackTile` (and rexp variants) | -12/+20 | `test_moreh_softmax.py` |
| T1.16 | `moreh_softmax/.../moreh_softmax_w_large.cpp` | L190 / L203 / L297 / L322 | same | -12/+20 | `test_moreh_softmax.py` |
| T1.17 | `moreh_norm/ord_other/moreh_norm_nc/.../moreh_norm_nc_kernel.cpp` | L39-60 (IS_ZERO) | `CopyTile + UnaryNe(0) + (OptionalChainElement<MINUS_INF, Negative>) + PackTile` | -22/+12 | `test_moreh_norm.py` |
| T1.18 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_large_kernel.cpp` | L77-98 (cb_xmm sub bcast no mask) | `BinaryFpu(Sub, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.19 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_large_kernel.cpp` | L451-477 (cb_ndymdysum) | `BinaryFpu(Sub, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.20 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_large_kernel.cpp` | L514-540 (cb_y mul bcast no mask) | `BinaryFpu(Mul, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.21 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_large_kernel.cpp` | L542-562 (cb_yydysum) | `BinaryFpu(Mul, Bcast=Cols/Scalar, both Pinned 0) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.22 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_small_kernel.cpp` | L140-159 (cb_y mul bcast no mask) | `BinaryFpu(Mul, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.23 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_input_grad_small_kernel.cpp` | L377-398 (cb_ndymdysum) | `BinaryFpu(Sub, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.24 | `moreh_layer_norm_backward/.../moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | L191-213 (cb_y = cb_xmm * cb_rstd, no mask path) | `BinaryFpu(Mul, Bcast=Cols/Scalar) + PackTile` | -22/+12 | `test_moreh_layer_norm_backward.py` |
| T1.25 | `moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp` | L48-63 (cb_tmp2 = -[w*g]) | `BinaryFpu(Mul, Bcast=Scalar) + Negative + PackTile` | -16/+10 | `test_moreh_nll_loss.py` |
| T1.26 | `moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp` | L65-79 (cb_input_grad = cb_tmp2 * cb_tmp1) | `BinaryFpu(Mul, Bcast=Scalar, B=NoWaitNoPop) + PackTile` | -14/+10 | `test_moreh_nll_loss.py` |
| T1.27 | `moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp` | L82-100 (no DIVISOR branch) | `BinaryFpu(Mul, Bcast=Scalar) + Negative + PackTile` | -18/+10 | `test_moreh_nll_loss.py` |
| T1.28 | `moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` | L43-61 (negative block) | `CopyTile + Negative + PackTile` | -18/+8 | `test_moreh_nll_loss.py` |
| T1.29 | `moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` | L63-80 (cb_tmp1 * cb_tmp_weight) | `BinaryFpu(Mul) + PackTile` | -18/+8 | `test_moreh_nll_loss.py` |
| T1.30 | `moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` | L82-97 (cb_tmp3 * cb_divisor_recip) | `BinaryFpu(Mul, Bcast=Scalar) + PackTile` | -16/+8 | `test_moreh_nll_loss.py` |
| T1.31 | `moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` | L106-130 (no-WEIGHT branch) | `BinaryFpu(Mul, Bcast=Scalar) + PackTile` | -25/+10 | `test_moreh_nll_loss.py` |
| T1.32 | `moreh_adam/.../moreh_adam.cpp` | L270-281 (pow(beta2, step)) | `CopyTile<…, Pinned> + Power(step) + PackTile` | -12/+8 | `test_moreh_adam.py` |
| T1.33 | `moreh_adam/.../moreh_adam.cpp` | L378-389 (pow(beta1, step)) | `CopyTile<…, Pinned> + Power(step) + PackTile` | -12/+8 | `test_moreh_adam.py` |
| T1.34 | `moreh_adam/.../moreh_adam.cpp` | L318-328 (#ifdef AMSGRAD copy) | `CopyTile + PackTile<…, Output>` | -11/+5 | `test_moreh_adam.py` |
| T1.35 | `moreh_adamw/.../moreh_adamw.cpp` | L281-292 (1/(1-beta2_exp)) | `BinaryFpu(Sub, cb_one, cb_beta2_exponent) + Recip + PackTile` | -12/+10 | `test_moreh_adamw.py` |
| T1.36 | `moreh_adamw/.../moreh_adamw.cpp` | L359-371 (1/(1-beta1_exp)) | `BinaryFpu(Sub, cb_one, cb_beta1_exponent) + Recip + PackTile` | -12/+10 | `test_moreh_adamw.py` |
| T1.37 | `experimental/reduction/deepseek_grouped_gate/.../deepseek_grouped_gate.cpp` | L326-343 (scale block) | `BinaryFpu(Mul, Bcast=Scalar, cb_normalized_scores, cb_route_scale_scalar) + PackTile<cb_out_weights>` | -18/+8 | `tests/sweep_framework/.../deepseek_grouped_gate_sweep.py` (or moe_b1 demo) |

**Total Type-1 blocks across all 31 B kernels: 37 distinct blocks.** Several of these (T1.11-T1.16) collapse into per-#ifdef variants in the same kernel; counted as separate IDs because each is its own commit-touch site.

---

## Section 4 — Type-2 catalog (deferred — would unlock more migrations)

Cross-referenced with design v6 Section E where applicable.

### Helper extensions ranked by unlock count

| Helper feature gap | Kernels unlocked | Design v6 E reference | Estimated new helper LOC |
|---|---|---|---|
| **MASK-INJECT** — `MaskInject<COND, MaskCb, MaskIdx>` chain element to inject `mask_tile(dst_data, dst_mask)` after another element (mid-loop conditional). Can be wrapped in `OptionalChainElement<COND, …>` for the `do_mask_h && need_to_do_mask_h(...)` runtime gate. | `moreh_clip_grad_norm_step1`, `moreh_norm_h`, `moreh_norm_w`, `moreh_norm_other`, `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_w`, `moreh_layer_norm_small`, `moreh_layer_norm_large`, `moreh_layer_norm_backward_gamma_beta_grad`, `moreh_layer_norm_backward_input_grad_small`, `moreh_layer_norm_backward_input_grad_large`, `moreh_softmax_h_large`, `moreh_softmax_w_large`, `moreh_softmax_backward_h`, `moreh_softmax_backward_h_large`, `moreh_softmax_backward_w`, `moreh_softmax_backward_w_large` | New (not in v6 E). | ~80 LOC. |
| **HELD-CB** — chain-iter with in-place CB recurrence: `cb_pop_front(X) + cb_push_back(X)` on the same CB. Helper would expose a `HeldCbBinaryFpu<…>` element with `BinaryFpu` semantics but pop-then-push pack policy. | `moreh_adam`, `moreh_adamw`, `moreh_clip_grad_norm_step1`, `moreh_layer_norm_small`, `moreh_layer_norm_large`, `moreh_layer_norm_backward_gamma_beta_grad`, `moreh_layer_norm_backward_input_grad_*`, `moreh_norm_h`, `moreh_norm_w`, `moreh_norm_other`, `moreh_norm/ord_other/*`, `moreh_softmax_c_large`, `moreh_mean_nc` | New (not in v6 E; called out in `migration_log.md` as "long-term gap"). | ~60 LOC. |
| **ASYMMETRIC-INDEX** — `BinaryFpuPerTileScalarB<…>` with `A=BlockIter, B=FirstTile` (the run9 use case). | `deepseek_grouped_gate` (add_bias), `eltwise_binary_scalar.cpp` (no-act branch), `moreh_layer_norm_backward_input_grad_small_kernel` L364-375 (cb_ndy index wt), `moreh_layer_norm_backward_input_grad_small_kernel` L400-419 (cb_yydysum index wt), `moreh_softmax_backward_h.cpp` (`moreh_bin_chain_rt`-replaceable callsites with strict A=Block, B=FirstTile asymmetry). | **Yes — Design v6 Section E lines 941-948 explicitly list this as future-recovery (deferred Q4 cost).** | ~25 LOC. |
| **ATOMIC PACK-N-AFTER-N-COPIES** — `BlockPackTileAtomic<N>` element for the binary_ng activations branch + remainder block. Coupled with the macro-injection wrapper. | `eltwise_binary_no_bcast` (activations), `eltwise_binary_sfpu_no_bcast` (activations), `eltwise_binary_sfpu_scalar` (activations), all 6 binary_ng `kernels_ng/*` activations branches. | New (audit Section 5 OOS-2). | ~120 LOC + macro-injection element type. |
| **OUTER-FREQ-LOOP DRIVER** — chain helper variant that wraps `complete_iterations × tile_freq + remainder` outer iteration around an inner chain, with caller-side preamble (`unary_bcast_init` / `cb_wait_front(cb_bcast, …)`) hooks. | `eltwise_where_sfpu`, `eltwise_where_sfpu_scalar`, `ternary_addc_ops_sfpu_bcast`, `ternary_sfpu_col_scalar_bcast_ttt`, `ternary_sfpu_row_bcast_ttt`, all 6 binary_ng `kernels_ng/*`. | New (audit Section 5 / coverage_audit Section 3 future-feature note). | ~100 LOC. |
| **MACRO-INJECT WRAPPER** — generic `LocalMacroBlockSfpu<MACRO, BlockSize>` element type to host PREPROCESS / BINARY_OP / PROCESS_POST_ACTIVATIONS macros. Some kernels already locally roll this. Lifting to library would consolidate. | `eltwise_binary_no_bcast` (activations), `eltwise_binary_sfpu_no_bcast` (activations), `eltwise_binary_sfpu_scalar` (activations), all `kernels_ng/*` activations branches. | New (audit Section 5 OOS-2). | ~80 LOC. |
| **`copy_tile_init_with_dt(cb, transpose_bool)`** non-standard moreh-side init — would let `moreh_layer_norm_*` `cb_mean → cb_rstd` copies migrate. Currently the second-arg `is_lastdim_layernorm` overload isn't modeled. | `moreh_layer_norm_small`, `moreh_layer_norm_large` (both cb_mean and cb_rstd writeouts). | New. | ~15 LOC (adapter to `CopyTile` reconfig). |
| **`UnaryBcast`-as-preamble chain element** — wraps `unary_bcast_init + unary_bcast(cb_bcast, 0, 0) + pack_tile(0, cb_llk_post)` as one chain element to consolidate the preamble in `kernels_ng/*` bcast variants. | `kernels_ng/eltwise_binary_col_bcast`, `_row_bcast`, `_row_col_bcast`, `_scalar_bcast`, `_sfpu_row_bcast`, `_where_sfpu_row_bcast`. | New. | ~50 LOC. |
| **MULTI-DEST per-block iter (j as DEST slot AND tile index)** — `BlockBinaryFpu<…, MultiDestStride=1>` that uses j∈[0..BlockSize) as both DEST slot and pack index. Couples with the activations macro-injection wrapper. | All `block_size > 1` moreh_layer_norm large/small, all binary_ng `kernels_ng/*`. | New. | ~60 LOC + interaction with the existing block-element infra. |
| **MOREH `binary_max_tile` / `mul_tiles_and_mask_tile` op-structs** — wrap the moreh-shipped helpers as new BinaryOp / TernaryOp templates. | `moreh_softmax_c_large`, `moreh_norm/ord_other/*`, `moreh_softmax_backward_h*`, `moreh_softmax_backward_w*`. | New. | ~30 LOC each. |
| **POWER-MULTISTAGE op-struct family** — wrap `power_tile_to_cb` helper as a chain shape (effectively a sub-chain that holds a DEST through 4 stages). | `moreh_norm_*`, `moreh_clip_grad_norm_step1`. | New. | ~80 LOC; very specialised. |

---

## Section 5 — Type-3 catalog (un-migratable patterns)

Documented briefly so future readers don't chase these:

- **`tilize_block` / `untilize_block`** — `ssm_prefix_scan.cpp` lines 17-52. Separate `tilize_helpers` family.
- **`transpose_wh_init_short` / `transpose_wh_tile`** — `deepseek_grouped_gate.cpp` lines 80-126, 184-203, 231-253. Separate `transpose_helpers` family.
- **`topk_local_sort` / `topk_merge` / `topk_rebuild`** — `deepseek_grouped_gate.cpp` lines 98, 172, 236-253. Separate `topk_helpers` family.
- **`acquire_dst()` / `release_dst()`** (deprecated dst-sync) — `deepseek_grouped_gate.cpp` lines 84, 123, 160, 179, 219, 254-260. Modern dst-sync (`tile_regs_*`) only in chain.
- **PACK-thread SFPU** (`llk_math_eltwise_*` on PACK risc) — `moe_compute/compute.cpp`, `moe_gpt/compute.cpp`. Chain v1 explicitly out of scope per `eltwise_chain.hpp` doc-comment line 200.
- **Matmul** (`matmul_block`, `mm_init`) — `moe_compute`, `moe_gpt`. Chain is eltwise-only.
- **`init_sfpu` (alternate engine-boot variant)** — `moreh_nll_loss_backward_kernel.cpp` line 27 uses `init_sfpu(cb_output_grad, tt::CBIndex::c_16)` instead of `binary_op_init_common`. Caller-owned per D8; not a chain-side issue but a callout.

---

## Section 6 — Sweep recommendations

### Suggested commit grouping for the planner

1. **Commit "moreh_softmax_backward Type-1 sweep"** — migrate all 10 Type-1 blocks across `moreh_softmax_backward_{c_large,h,h_large,w,w_large}.cpp` (T1.01 – T1.10). All share `test_moreh_softmax_backward.py`; all are `CopyTile + Exp + PackTile` or `BinaryFpu(Mul) + Negative + PackTile` patterns. Estimated -75 LOC, +52 LOC. Lowest risk; mostly mechanical.

2. **Commit "moreh_softmax forward Type-1 sweep"** — T1.11 – T1.16 across 3 kernels. All share `test_moreh_softmax.py`. ~-40/+60 LOC.

3. **Commit "moreh_layer_norm_backward Type-1 sweep"** — T1.18 – T1.24 across 3 kernels (`gamma_beta_grad`, `input_grad_small`, `input_grad_large`). Share `test_moreh_layer_norm_backward.py`. ~-150/+85 LOC. Higher complexity due to bcast-mode templating; deserves its own commit for bisect ergonomics.

4. **Commit "moreh_nll_loss + nll_loss_backward Type-1 sweep"** — T1.25 – T1.31. Share `test_moreh_nll_loss.py`. ~-110/+55 LOC.

5. **Commit "moreh_adam / moreh_adamw bias-correction Type-1 sweep"** — T1.32 – T1.36. Share `test_moreh_adam.py` and `test_moreh_adamw.py`. ~-60/+40 LOC.

6. **Commit "moreh_norm/ord_other UnaryNe IS_ZERO branch"** — T1.17. Single block but spans 4 files via `#ifdef`. Run `test_moreh_norm.py`. ~-22/+12 LOC. Verify UnaryNe runtime-param dispatch under run7 chain (small risk-spot — test thoroughly in --dev mode).

7. **Commit "deepseek_grouped_gate scale block"** — T1.37. Run sweep test. Smallest, quickest win.

### Risk hotspots (Type-1 borderline)

- **T1.17 `moreh_norm_nc IS_ZERO`** — Migration log claims the chain dispatch path for runtime-param SFPU ops (UnaryNe) was a GAP. `eltwise_predicates.hpp:61` and `eltwise_chain.inl:888,965` show the dispatch path is implemented. Verify the assertion pre-commit — if a unit-test for UnaryNe inside a chain doesn't exist, write a probe before this migration.
- **T1.04 / T1.08 `moreh_softmax_backward_{h,w}.cpp` line 204/203** — `mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx, h, 0, /*pop0=*/0, /*pop1=*/1)` requires `BinaryFpu<A=Pinned(h), B=Pinned(0), Apolicy=NoWaitNoPop, Bpolicy=WaitAndPop>`. Q4 collapse keeps single `Index=Pinned` and asymmetric per-side policies (verified compatible). Tile-idx fields `a_tile_idx=h, b_tile_idx=0` are independent (per design Q17). Should work, but the policy-asymmetry is a non-uniform pattern — exercise with `--dev` first.
- **T1.21 `moreh_layer_norm_backward_input_grad_large_kernel.cpp` L542-562** — A-side needs `tile_idx=0` (Pinned) and B-side `tile_idx=0` (Pinned). Confirm the Wt-loop variable doesn't leak in. If the actual kernel uses `tile_idx=wt` for cb_y (loop var), this is Q4-asymmetric (Type 2). Verify via re-read at landing time.
- **T1.32 / T1.33 `moreh_adam` Power**: `Power<>` element holds `uint32_t exponent` as a runtime member. Chain dispatcher routes `apply()` through member-exec for runtime-param ops (same path as UnaryNe). Verify under `--dev`.

### Kernels to skip entirely from this sweep (Type-2-only or Type-3-only)

- All 5 binary_ng `kernels/*` (entries 1-5 in Section 1's enumeration) — Type-2 + Type-3 only.
- All 6 binary_ng `kernels_ng/*` (entries 6-11) — Type-2 only (activations + bcast-preamble).
- All 3 ternary `_bcast` (entries 12-14) — Type-2 only.
- All 3 `bcast_to/*` (entries 15-17) — Type-2 only.
- Both `moe_compute` / `moe_gpt` (entries 18-19) — Type-3 (matmul + PACK-thread SFPU).
- `ssm_prefix_scan` (entry 21) — Type-3 (tilize/untilize).
- `moreh_clip_grad_norm_step1` (entry 24) — Type-2 only.
- Both `moreh_layer_norm_{small,large}_kernel` (entries 25, 26) — Type-2 only.
- `moreh_mean_nc` (entry 30) — Type-2 only.
- `moreh_sgd` (entry 39) — Type-2 only (runtime CB).
- `moreh_norm_h`, `moreh_norm_w` (entries 33, 35), `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_w` (entries 36, 38) — Type-2 only (mask + held-CB).
- `moreh_norm_other` (entry 34) — Type-2 only (already-migrated chain blocks remain; rest blocked by power_tile_to_cb + held-CB).

**Skipped kernels: 22 of the 47-file enumeration (or 16 of 31 deduplicated B rows).** Remaining 15 file-level kernels carry the 37 Type-1 blocks for the sweep.

---

## Section 7 — Surprises and findings

### Re-verified misclassifications

- **`moreh_adam.cpp` and `moreh_adamw.cpp`** — coverage_audit re-classified these from "MIGRATED" to "B" because `*_tiles_to_cb` substitution leaves the bias-correction blocks raw. Confirmed by reading: `moreh_adam.cpp` has 4-5 raw blocks (lines 270-281, 284-298, 302-315, 318-328, 332-357, 360-374, 378-389, 392-406). Of these, 3 are mechanically Type 1 (T1.32, T1.33, T1.34); the rest are held-DEST or held-CB (Type 2).

- **`moreh_layer_norm_backward_input_grad_small_kernel.cpp` L400-419 cb_yydysum** — coverage_audit classifies this kernel as B with "held-DEST + bcast scratch" raw. After reading, I find this specific block uses `mul_tiles_bcast_*(cb_y, cb_ydysum, wt, 0, dst0)` — A-side index `wt` (runtime, loop var) with B-side index 0. **This is Q4-collapse-incompatible** (A=BlockIter, B=FirstTile asymmetric → Type 2 unrecoverable in this run). Listed as **not** Type 1 in Section 2.11; same for the analogous L400-419 in the small kernel. The "free" Type 1 blocks are the no-mask, both-tile-index-0 variants only.

- **UnaryNe runtime-param SFPU dispatch** — migration_log.md asserts a dispatch GAP. Reading run7-refined helper at `eltwise_predicates.hpp:61` (`UnaryNe` definition with member-`exec(uint32_t)`) and `eltwise_chain.inl:888,965` (member-exec routing) shows the GAP is closed. T1.17 (`moreh_norm_nc IS_ZERO`) is therefore a Type 1 candidate. Recommend `--dev` testing before commit.

- **`moreh_mean_nc.cpp`** — coverage_audit notes it as B with one migrated chain (stage 2) and held-CB raw. Confirmed; no additional Type 1 blocks.

- **`moreh_clip_grad_norm_step3`** — coverage_audit re-classified as **A** (full main-loop migrated). Excluded from this audit (not in B set).

- **`moreh_mean_backward.cpp`** and **`moreh_sum_backward.cpp`** — coverage_audit re-classified to A. Excluded.

### Helper-feature shape gaps NOT in design v6 Section E

- **MASK-INJECT** (mid-loop conditional `mask_tile`) is the single biggest unlock: blocks Type-1 migration in 17+ moreh kernels.
- **HELD-CB** (`cb_pop_front + cb_push_back` same CB for accumulators) is the second-biggest: blocks Type-1 migration in 13+ moreh kernels.
- **MULTI-DEST per-iter** (j∈[0..BlockSize) as both DEST slot and tile index) is the third: blocks all moreh_layer_norm large/small kernels' inner blocks.

These three together would unlock ~70% of the remaining Type-2 raw LOC across the moreh family. Recommend prioritising them as next-run helper extensions.

### Compositional Type-1 patterns observed

Many moreh kernels share the wrappers:
- `moreh_bin_chain<Op, A, B, Out, IdxA, IdxB, PopA, PopB>` (line 22-53 in many kernels).
- `moreh_copy_chain<In, Out, Idx, Pop>` (line 55-77 in many kernels).
- `moreh_bin_chain_rt` (runtime-index variant in `moreh_softmax_backward_{h,w}.cpp`).

A natural extension: introduce a `moreh_unary_chain<UnarySfpu, In, Out, Idx, Pop>` wrapper to consume `exp_tile_to_cb`, `rexp_tile_to_cb`, `log_tile_to_cb`, `recip_tile_to_cb` migrations (T1.01 – T1.16). One commit could land the wrapper template + apply across ~15 callsites uniformly. Estimated +30 LOC for the wrapper + -50 LOC across kernels = net -20 LOC and lower commit-touch cost than per-callsite chain expansion.

### Aggregate metrics

- **Total Type-1 blocks across 31 B kernels: 37.**
- **Files with at least one Type-1 block: 15 / 47 file-level B entries.**
- **Files that migrate to A after Type-1 sweep: 0.** All Type-1 sites are isolated raw blocks inside otherwise-B kernels; the surrounding Type-2 blocks (mask, held-CB, multi-DEST) keep the kernel B-classified. This audit's sweep is a **partial-improvement** sweep, not a B-to-A promotion sweep.
- **Median Type-1 commit-touch:** ~12 LOC removed, ~7 LOC added per block (after factoring in moreh wrapper consolidation).
- **Total estimated commit-touch for full Type-1 sweep:** ~580 LOC removed, ~340 LOC added. Mostly mechanical.

---

*End of audit.*
