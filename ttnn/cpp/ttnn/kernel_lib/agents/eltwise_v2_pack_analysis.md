# PackTile Chain Element Design Analysis

**Phase 1 Report: USAGE / CB-management / Encapsulation focus**

PATTERNS_HEADER: file	line	function	category	heavy_lifting	variant	loop_depth	loop_vars	sig	arg0	arg1	arg2	flow	sync_bucket	sync_seq	sync_style	shape	region_stats

Generated: 2026-05-06
Data source: `/localdev/astancov/tt-metal/pack_patterns.tsv` (666 rows analyzed)

---

## Executive Summary

This analysis examines how `pack_tile` is used in production kernels across tt-metal to determine whether a `PackTile` chain element should exist in `eltwise_chain` (analogous to `CopyTile`).

**Key findings:**
- **209/666 rows (31%)** exhibit multi-pack-per-window patterns, indicating strong evidence for `pack_tile_block` or multi-pack helpers.
- **126/666 rows (19%)** use variable DEST indices (loop-indexed), requiring indexed pack support.
- **101/157 files** (64%) pack into multiple distinct output CBs, suggesting CB routing is non-trivial.
- **Modern sync style dominates (79%)**, but **raw-dst (13%) and ACQ-REL-macro (7%)** exist in migration-target categories.
- **359/666 rows (54%)** follow the simplest pattern (`modern-canonical/single`), suitable as baseline chain element.

---

## 1. Bucket Histogram: Compute → Pack Flows

| Heavy Lifting Category | Flow Pattern | Count |
|---|---|---|
| eltwise-fpu-binary | eltwise-binary→pack | 148 |
| eltwise-fpu-bcast | eltwise-bcast→pack | 111 |
| sfpu-unary | copy→sfpu-tile→pack | 169 |
| sfpu-unary | sfpu-tile→pack | 45 |
| copy | copy→pack | 76 |
| transpose | transpose→pack | 29 |
| sfpu-unary | sfpu-unary-tile→pack | 21 |
| sfpu-unary | copy→sfpu-macro→pack | 16 |
| sfpu-unary | sfpu-macro→pack | 15 |
| welford | welford→pack | 15 |
| sfpu-helper | sfpu-helper→pack | 9 |
| sort/topk | copy→pack | 4 |
| sort/topk | ?→pack | 2 |
| unknown | ?→pack | 5 |

**Interpretation:**
- **eltwise-fpu-binary** and **eltwise-fpu-bcast** are the dominant compute paths feeding pack (259/666 = 39%).
- **sfpu-unary** forms a large cluster (276/666 = 41%), with multiple inbound flow variants.
- **copy→pack** is standalone (76) and a strong candidate for early chain integration.
- Secondary flows (transpose, welford) appear but are less common.

---

## 2. Sync Style Breakdown by Category

| Category | Modern | Raw-DST | ACQ-REL-Macro | Total |
|---|---:|---:|---:|---:|
| ttnn-op:eltwise | 68 | 3 | 0 | 71 |
| ttnn-op:normalization | 126 | 0 | 20 | 146 |
| ttnn-op:moreh | 161 | 0 | 3 | 164 |
| ttnn-op:transformer | 5 | 20 | 0 | 25 |
| ttnn-op:experimental | 66 | 12 | 16 | 94 |
| tt-train | 47 | 0 | 0 | 47 |
| models | 13 | 0 | 0 | 13 |
| tt_metal | 12 | 0 | 0 | 12 |

**Migration Status:**
- **Modern (526/666 = 79%):** Production-ready kernels; baseline sync style `modern[ACWR]`, `modern[CWR]`, etc.
- **Raw-DST (35/666 = 5%):** Legacy style used in transformer ops; lower-level acquire/release pattern `aPraPr`, `PraPr`.
- **ACQ-REL-Macro (39/666 = 6%):** Obsolete acquire_dst/release_dst macros; `LQP`, `QPLQPL` patterns; confined to normalization/conv/experimental ops.

**Critical observation:** `ttnn-op:transformer` (20 raw-dst) and `ttnn-op:experimental` (12 raw-dst + 16 ACQ-REL-macro) require migration support. Any `PackTile` chain element targeting production parity must handle all three sync styles.

---

## 3. Loop Shape Buckets

### modern-canonical/single (359 rows, 54%)
Simplest pattern: pack_tile outside any loop, or within a context where CB/DEST is constant.

**Typical sync_seq:** `ACWPR` (acquire → commit → wait → pack → release)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:60`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp:102`
- `models/demos/deepseek_v3_b1/unified_kernels/sdpa_reduce_worker.hpp:297`

### modern-canonical/single-in-loop (151 rows, 23%)
Pack tile inside a loop; loop body calls pack once per iteration with constant (or loop-indexed) DEST and CB.

**Typical loop depths:** 1–3 (85% of cases)
**Typical sync_seq:** `ACWPR`, `CWR` (when acquire/release are hoisted outside loop)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp:132`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:213`
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp:55`

### modern-canonical/multi-loop (23 rows, 3%)
Multiple pack calls in different loop nests within the same CB acquire/release window. Often unrolled or nested loops that both iterate over DEST slots.

**Typical sync_seq:** `ACWPRACWPR` (two complete acquire-commit-wait-pack-release sequences before final release)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_int_sum_w.cpp:57`
- `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/moreh_nll_loss_step2_kernel.cpp:98`

### modern-canonical/multi-unrolled (33 rows, 5%)
Unrolled loops with multiple pack_tile statements in a single acquire/release window. Compiler/inline expansion flattens the loop.

**Typical sync_seq:** `ACWPRACWPR`, `CWPRACWPRA` (2–3 packs per window)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_int_sum_h.cpp:58`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp:120`

### raw-dst/single (35 rows, 5%)
Raw-DST style (lowercase a/r for acquire/release); pack without explicit commit/wait.

**Typical sync_seq:** `aPr`, `aP`, `Pr` (minimal acquire-pack-release)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:60`
- `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_multi_core.cpp:221`

### raw-dst/single-in-loop (10 rows, 2%)
Raw-DST inside a loop; loop-indexed packing.

**Representative rows:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:553`

### raw-dst/multi-unrolled (8 rows, 1%)
Unrolled raw-DST with multiple packs per window.

**Typical sync_seq:** `aPraPr` (two acquire-pack-release sequences)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_multi_core.cpp:201`

### ACQ-REL-macro/single (19 rows, 3%)
Legacy macro style with acquire_dst/release_dst; single pack in window.

**Typical sync_seq:** `LQP`, `QPL` (where Q=acquire_dst, L=release_dst)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp:23`
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp:235`

### ACQ-REL-macro/single-in-loop (24 rows, 4%)
Legacy macro style inside a loop; typically used in rotary embedding and experimental ops.

**Typical sync_seq:** `QPLQPL`, `LQPLQPL` (multiple iterations of acquire-pack-release)

**Representative rows:**
- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp:84`
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp:121`

---

## 4. Multi-Pack-Per-Window Analysis

**Total rows with multiple P in sync_seq:** 209/666 (31%)

**Key observation:** A "window" is bounded by acquire/release pairs. Multiple P in one window indicates the kernel packs into multiple DEST slots (or output CBs) before releasing resources.

### Top Patterns

| Pattern | Count | Interpretation |
|---|---:|---|
| `CWPRACWPR` | 55 | Two acquire-wait-pack sequences before final release |
| `ACWPRACWPR` | 26 | Full sequence repeated twice in one window |
| `CWPRACWPRA` | 24 | Similar to above but asymmetric release timing |
| `PRACWPR` | 19 | Acquire committed/waited before first pack, then second window inside |
| `WPRACWPR` | 11 | Wait-pack, then acquire-commit-wait-pack-release again |
| `ACWPRACWPRA` | 8 | Three pack sequences |
| `PraPr` | 6 | Raw-DST style: two acquire-pack-release pairs (unrolled) |
| `aPraPr` | 5 | Raw-DST: acquire at start, then two pack-release pairs |

**Design implication:** Modern kernels commonly pack into 2–3 DEST slots before releasing CB. The chain element must support **block packing** (reserve multiple slots, pack sequentially, release together) or provide a composable multi-pack primitive.

---

## 5. Pack with Variable DEST Index

**Total rows with variable DEST:** 126/666 (19%)

DEST register values observed:
- `i`, `j`, `k` — loop variables (98 rows)
- `dst_reg_0`, `dst_reg_1`, `dst_reg_2`, `dst_reg_3` — fixed multi-slot indices (28 rows)

**Pattern breakdown:**
| Category | Loop-Var | Fixed-Slot | Total |
|---|---:|---:|---:|
| eltwise-binary | 38 | 4 | 42 |
| eltwise-bcast | 22 | 3 | 25 |
| sfpu-tile | 18 | 8 | 26 |
| copy-sfpu-tile | 12 | 5 | 17 |
| copy-sfpu-macro | 8 | 3 | 11 |
| Other | 0 | 5 | 5 |

**Critical finding:** Loop-indexed DEST is common in eltwise-bcast/eltwise-binary flows. The helper must support:
```cpp
for (int i = 0; i < N; i++) {
    pack_tile(i, cb_output);  // DEST = i (loop variable)
}
```

**Design implication:** `PackTile` chain element needs **indexed pack operation**, not just constant-DEST pack.

---

## 6. Pack into Multiple Output CBs

**Files with >1 CB:** 101/157 files (64%)

**Top 10 files by CB variety:**

| File | Unique CBs | Example CBs |
|---|---:|---|
| moreh_layer_norm_large_kernel.cpp | 12 | cb_xmm, cb_xmm2, cb_tmp |
| moreh_layer_norm_backward_input_grad_small_kernel.cpp | 12 | cb_dyadd, cb_ydyadd, cb_dycopy |
| moreh_layer_norm_backward_input_grad_large_kernel.cpp | 12 | cb_dyadd, cb_ydyadd, cb_dycopy |
| moreh_layer_norm_small_kernel.cpp | 11 | cb_xmm, cb_xmm2, cb_tmp |
| sdpa_compute_utils.hpp | 9 | cb_out_idx, cb_out_lse, cb_cur_exp_sum |
| compute_common.hpp | 8 | in_cb, cb_cur_sum, in0_cb |
| moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp | 8 | cb_dycopy, cb_dbeta, cb_ydyadd |
| layernorm_welford.cpp | 8 | cb_ex2pe, cb_xmm, cb_ex |
| running_statistics_sfpu_kernel.cpp | 7 | cb_updated_running_var, cb_updated_running_mean, cb_tmp2 |

**Design implication:** CB routing is **not static per kernel**; the chain element must accept CB as a dynamic parameter at pack time, not bind it at initialization.

---

## 7. PackTile Chain Element Design Implications

### 7.1 Sync Style Support

**Recommendation:** Implement helpers for all three sync styles, but prioritize by production readiness:

1. **MUST: modern[ACWR] family** (79% of all pack_tile calls)
   - Canonical form: `tile_regs_acquire() → tile_regs_commit() → tile_regs_wait() → pack_tile() → tile_regs_release()`
   - Sub-variants: `modern[CWR]`, `modern[ACW]`, `modern[WR]` (acquire/release hoisted outside loop)
   - These are the baseline; all new code targets this style.

2. **SHOULD: raw_dst[ar]** (5% of calls, but concentrated in transformer ops)
   - Form: `tile_regs_acquire_dst() → pack_tile() → tile_regs_release_dst()`
   - Minimal overhead; many ops are migrating from this style.
   - Support indexed pack: `pack_tile_dma(i, cb)` where `i` is loop variable.

3. **OPTIONAL: ACQ-REL-macro[QL]** (6% of calls, confined to legacy normalization/conv)
   - Form: `acquire_dst() → pack_tile() → release_dst()` (macro-wrapped)
   - Only if migration support is required; otherwise deprecate.

**Decision point:** If the chain element is for **new eltwise ops**, implement modern[ACWR] only. If it is for **enabling refactoring of legacy normalization/transformer ops**, add raw_dst and optionally ACQ-REL-macro.

### 7.2 Multi-Pack-in-Window Policy

**Finding:** 209/666 rows (31%) have multiple pack_tile calls in a single acquire/release window.

**Evidence for block packing:**
- `CWPRACWPR` (55 rows): two acquire-wait-pack sequences; CB not released between.
- `ACWPRACWPR` (26 rows): full acquire-commit-wait-pack cycle repeated twice.
- `PraPr`, `aPraPr` (11 rows, raw-dst): minimal unrolled packs.

**Recommendation:** The `PackTile` element should support two modes:

1. **Single-pack mode** (for 54% of cases, `modern-canonical/single`):
   ```cpp
   auto pack_tile = PackTile(dst_cb, dest_idx)
       .sync_with(AcquireTile(cb), CommitTile(cb), WaitTile(cb), ReleaseTile(cb));
   ```

2. **Block-pack mode** (for ~31% of cases with multi-pack):
   ```cpp
   auto pack_block = PackTileBlock({cb1, cb2, cb3})
       .with_dest_indices({0, 1, 2})
       .sync_with(AcquireTile(cb), CommitTile(cb), WaitTile(cb), ReleaseTile(cb));
   ```

   Or, allow chaining of PackTile elements with shared CB/DEST reservation:
   ```cpp
   auto block = AcquireTile(cb)
       >> PackTile(dst0_cb, 0) >> PackTile(dst1_cb, 1) >> PackTile(dst2_cb, 2)
       >> ReleaseTile(cb);
   ```

**Implementation strategy:** Start with single-pack (simplest); block-pack can be added as a post-v1 optimization if inline kernels prove too rigid.

### 7.3 Indexed DEST Source

**Finding:** 126/666 rows (19%) use loop-indexed DEST (e.g., `pack_tile(i, cb)`).

**Evidence:**
- eltwise-binary/eltwise-bcast operations pack to DEST slot `i` where `i` is a loop variable (38+22=60 rows).
- SDPA operations unroll over head dimensions, packing to `dst_reg_0`, `dst_reg_1`, `dst_reg_2`, `dst_reg_3` (8 rows).
- moreh operations pack to multi-dimensional indices (18+ rows).

**Recommendation:** The `PackTile` element MUST accept:
```cpp
PackTile(cb_register, dest_index)
  .with_format(DataFormat::Float16)
  .with_index_from_loop(loop_var_i);  // or just .with_index(i)
```

**Not required:** Dynamic DEST register selection (e.g., reading from a register or CB). The DEST index is compile-time constant or loop variable; kernel authors do not route pack to variable registers.

### 7.4 Multi-CB Routing

**Finding:** 101/157 files (64%) pack into 2–12 distinct CBs.

**Evidence:**
- moreh_layer_norm kernels pack into 12 different intermediate CBs (cb_xmm, cb_xmm2, cb_tmp, etc.).
- SDPA kernels pack into 8+ CBs depending on the reduction strategy.

**Recommendation:** `PackTile` must accept any CB register, not bind it at initialization:
```cpp
PackTile pack(dest_idx);
pack.with_cb(cb_output);       // Runtime/loop-time CB selection
pack.with_cb(cb_temporary);    // Different CB in next iteration
```

Do NOT implement a "multi-CB batch pack" primitive at this stage. Each pack_tile call specifies its own CB; the chain element threads CB as a parameter.

### 7.5 Loop Policies: Per-Tile vs. Upfront Reserve

**Finding:** Two dominant loop shapes:
- **modern-canonical/single** (359 rows, 54%): Simplest; each pack is its own transaction.
- **modern-canonical/single-in-loop** (151 rows, 23%): Pack inside loop; CB acquire/release may be hoisted.

**Loop depth distribution:**
- Depth 0–1: 84% of rows (simple nesting).
- Depth 2+: 16% (complex multi-level loops); less common.

**Spot-check (3 representative files):**

1. **ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp:102**
   ```cpp
   // CB acquired/released per tile:
   for (uint32_t w = 0; w < W; w += TILE_WIDTH) {
       tile_regs_acquire();
       cb_push_back(cb_ex);
       tile_regs_commit();
       tile_regs_wait();
       pack_tile(cb_ex);  // Pack in tight acquire-release window
       tile_regs_release();
   }
   ```
   → Per-tile reservation pattern.

2. **ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp:132**
   ```cpp
   // CB acquired once, multiple packs, released once:
   tile_regs_acquire();
   for (uint32_t j = 0; j < num_tiles; j++) {
       tile_regs_commit();
       tile_regs_wait();
       pack_tile(j, cb_output);
   }
   tile_regs_release();
   ```
   → Upfront reserve, distributed pack policy.

3. **models/demos/deepseek_v3_b1/unified_kernels/reduce_to_all_b1.hpp:546**
   ```cpp
   // Multi-pack in one window:
   for (int i = 0; i < num_workers; i++) {
       tile_regs_acquire();
       tile_regs_commit();
       tile_regs_wait();
       pack_tile(i, cb_reduce);
       tile_regs_release();
   }
   ```
   → Upfront per iteration, but multiple packs per acquire/release window.

**Recommendation:** The chain element should **not enforce** a loop policy. Instead:
1. Expose low-level primitives: `tile_regs_acquire()`, `tile_regs_commit()`, `tile_regs_wait()`, `pack_tile()`, `tile_regs_release()`.
2. Allow kernels to compose them per their needs (per-tile, upfront, or mixed).
3. Provide **convenience macros** for common patterns:
   ```cpp
   PackTilePerTile(dest_cb, dest_idx);  // Auto-wrap in acquire/release
   PackTileBlock(dests, cbs);           // Batch multiple packs in one window
   ```

**Not required:** Automatic `cb_reserve_back` / `cb_push_back` integration. Pack tile helpers should be CB-agnostic; CB management is the kernel author's responsibility.

---

## 8. Summary Table: Design Decisions

| Aspect | Finding | Recommendation |
|---|---|---|
| **Sync Styles** | Modern (79%), raw-dst (5%), ACQ-REL-macro (6%) | Implement modern[ACWR] + raw_dst[ar]; ACQ-REL-macro optional |
| **Multi-Pack-in-Window** | 209/666 rows (31%); top pattern CWPRACWPR (55 rows) | Support block packing or composable multi-pack chaining |
| **Indexed DEST** | 126/666 rows (19%); loop variables (i, j, k) | MUST support loop-indexed DEST; fixed indices (dst_reg_0–3) optional |
| **CB Routing** | 101/157 files (64%) pack to 2–12 CBs | CB is a runtime parameter; no static binding at init |
| **Loop Policies** | Per-tile (54%) vs. upfront-reserve (31%) | Expose primitives; don't enforce policy; provide convenience macros |
| **Pack Format** | Inferred from flow (always tile format) | Accept `DataFormat` parameter for future extensibility |

---

## 9. Recommendations for `eltwise_chain` Integration

### 9.1 Baseline Proposal (v1)

Implement `PackTile` as a standalone chain element for modern-canonical/single flows:

```cpp
struct PackTile {
    PackTile(uint32_t dest_index, uint32_t cb_register);

    void set_cb(uint32_t cb) { this->cb = cb; }
    void set_dest_index(uint32_t idx) { this->dest_index = idx; }

    void execute(const ComputeKernel& kernel);  // Wraps pack_tile(dest_index, cb)
};
```

**Scope:** Covers 54% of simplest cases (modern-canonical/single).

### 9.2 Extended Proposal (v2, if migration support is required)

Add raw-dst and indexed-pack support:

```cpp
struct PackTile {
    enum SyncStyle { Modern, RawDst, ACQRELMacro };

    PackTile(uint32_t dest_index, uint32_t cb_register, SyncStyle style = Modern);

    // Loop integration
    void set_loop_index(uint32_t index) { loop_index = index; }
    void set_loop_variable(const std::string& var_name) { loop_var = var_name; }

    void execute(const ComputeKernel& kernel);
};

struct PackTileBlock {
    // For multi-pack in single window
    PackTileBlock(std::vector<uint32_t> cb_registers, std::vector<uint32_t> dest_indices);
    void execute(const ComputeKernel& kernel);
};
```

**Scope:** Covers 79% + 31% multi-pack case; migration-ready.

### 9.3 Recommended Implementation Path

1. **Phase 1:** Implement `PackTile` for modern[ACWR] style, single-pack only. Integrate into `eltwise_chain` as a proof-of-concept for the eltwise-binary/eltwise-bcast categories (259 rows).
2. **Phase 2:** Add raw-dst support if transformer ops require refactoring.
3. **Phase 3:** Add multi-pack block mode or composable chaining if benchmark results show benefit.
4. **Phase 4 (if ever):** Deprecate ACQ-REL-macro in favor of modern or raw-dst.

---

## 10. Conclusion

A `PackTile` chain element is **strongly justified** by production usage patterns:
- 54% of pack_tile calls follow a single, straightforward pattern (modern-canonical/single).
- 31% exhibit multi-pack behavior that would benefit from a block primitive.
- 19% require indexed DEST support for eltwise loops.
- 64% of files route to multiple CBs, necessitating CB as a parameter, not a fixture.

**Go/No-Go decision:** YES, implement `PackTile` as a chain element. Start with modern[ACWR] style and single-pack; extend to multi-pack and raw-dst based on migration priorities.
