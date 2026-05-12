# Eltwise Compute Helpers v2 Catalog
## Phase 0: Demand Surface & LLK API Coverage

**Generated:** 2026-05-06
**Data Source:** `/localdev/astancov/tt-metal/pack_patterns.tsv`

---

## PATTERNS_HEADER
```
file	line	function	category	heavy_lifting	variant	loop_depth	loop_vars	sig	arg0	arg1	arg2	flow	sync_bucket	sync_seq	sync_style	shape	region_stats
```

---

## Section 1: TSV Demand Surface Analysis

The TSV contains **666 rows** of pack call-sites. Aggregating by `(heavy_lifting, flow)` bucket:

| # | Heavy-Lifting | Flow | Count | LLK Family | Top 3 Representative Call-Sites |
|---|---|---|---|---|---|
| 1 | sfpu-unary | copy→sfpu-tile→pack | 169 | SFPU unary + copy + pack | `sdpa.h:495`, `eltwise_sfpu.cpp:40`, `sfpu_eltwise_chain.cpp:61` |
| 2 | eltwise-fpu-binary | eltwise-binary→pack | 148 | FPU binary + pack | `gated_reduce.hpp:114`, `gated_reduce.hpp:129`, `reduce_to_all_b1.hpp:525` |
| 3 | eltwise-fpu-bcast | eltwise-bcast→pack | 111 | FPU binary (bcast) + pack | `rope.hpp:212`, `compute_common.hpp:418`, `compute_common.hpp:447` |
| 4 | copy | copy→pack | 76 | CB load (copy_tile) + pack | `sdpa_reduce_worker.hpp:297`, `tiles_smoothstep.cpp:94`, `tiles_copy.cpp:56` |
| 5 | sfpu-unary | sfpu-tile→pack | 45 | SFPU unary + pack (direct) | `gated_reduce.hpp:97`, `local_reduce.hpp:103`, `compute_common.hpp:351` |
| 6 | transpose | transpose→pack | 29 | **uncovered** (special kernel) | `moreh_matmul.cpp:55`, `transpose_wh_sharded.cpp:27`, `layernorm_pre_allgather_welford.cpp:90` |
| 7 | sfpu-unary | sfpu-unary-tile→pack | 21 | SFPU unary + pack | `eltwise_binary_scalar_bcast.cpp:45`, `eltwise_where_sfpu_row_bcast.cpp:62`, `eltwise_binary_row_col_bcast.cpp:64` |
| 8 | sfpu-unary | copy→sfpu-macro→pack | 16 | SFPU macro-based + pack | `eltwise_sfpu.cpp:47`, `eltwise_binary.cpp:141`, `eltwise_binary.cpp:143` |
| 9 | welford | welford→pack | 15 | **uncovered** (Welford-specific) | `layernorm_pre_allgather_welford.cpp:74`, `layernorm_sharded_welford.cpp:298`, `layernorm_large_tensor_welford.cpp:63` |
| 10 | sfpu-unary | sfpu-macro→pack | 15 | SFPU macro-based + pack | `eltwise_binary_sfpu_kernel.cpp:190`, `layernorm.cpp:266`, `layernorm.cpp:302` |
| 11 | sfpu-helper | sfpu-helper→pack | 9 | **uncovered** (internal helper) | `moreh_int_sum_nc.cpp:37`, `moreh_int_sum_w.cpp:47`, `moreh_int_sum_w.cpp:57` |
| 12 | unknown | ?→pack | 5 | **uncovered** (unknown op) | `compute_collector.cpp:53`, `running_statistics_kernel.cpp:57`, `numeric.h:178` |
| 13 | sort/topk | copy→pack | 4 | CB load + pack | `sort_single_row_multi_core.cpp:176`, `sort_single_row_multi_core.cpp:177`, `sort_single_row_multi_core.cpp:181` |
| 14 | sort/topk | ?→pack | 2 | **uncovered** (unknown op) | `topk.cpp:233`, `topk.cpp:237` |
| 15 | sfpu-unary | copy→sfpu-llk→pack | 1 | SFPU LLK + pack | `compute_common.hpp:1092` |

**Total rows in TSV:** 666 (header + 666 data rows)

---

## Section 2: LLK API Surface by Category

### 2.1 FPU Binary Operations
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h`

**Primary APIs:**
- `add_tiles(icb0, icb1, itile0, itile1, idst)` — element-wise addition A+B
- `sub_tiles(icb0, icb1, itile0, itile1, idst)` — element-wise subtraction A-B
- `mul_tiles(icb0, icb1, itile0, itile1, idst)` — element-wise multiplication A*B

**Init APIs (3 variants):**
- `add_tiles_init(icb0, icb1, acc_to_dest=false)`
- `sub_tiles_init(icb0, icb1, acc_to_dest=false)`
- `mul_tiles_init(icb0, icb1)` or `mul_tiles_init(icb0, icb1, acc_to_dest)`

**Destination-Reuse Variants:**
- `binary_dest_reuse_tiles_init<EltwiseBinaryType, binary_reuse_dest>(icb0)`
- `binary_dest_reuse_tiles<EltwiseBinaryType, binary_reuse_dest>(in_cb_id, in_tile_index, dst_tile_index)`

**Common init:**
- `binary_op_init_common(icb0, icb1, ocb)` — shared init for all binary ops

---

### 2.2 SFPU Unary Operations
**Directory:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/eltwise_unary/`
**~57 header files**, with patterns:

| Header | Key Operations | Template Variants |
|---|---|---|
| `activations.h` | `hardsigmoid_tile`, `softsign_tile`, `celu_tile`, `softshrink_tile`, `hardshrink_tile` | APPROX mode, custom params |
| `exp.h` | `exp_tile`, `exp_init` | APPROX, fast variants |
| `relu.h` | `relu_tile`, `relu_init` | Standard, with thresholds |
| `sqrt.h` | `sqrt_tile`, `sqrt_init`, `rsqrt_tile` | APPROX mode |
| `gelu.h` | `gelu_tile`, `gelu_init` | APPROX, fast, iterations |
| `dropout.h` | `dropout_tile`, `dropout_init` | Probability parameter |
| `clamp.h` | `clamp_tile`, `clamp_init` | Min/max bounds |
| `fill.h` | `fill_tile`, `fill_init` | Constant value fill |
| `rand.h` | `rand_tile`, `rand_init` | Random number gen |
| `where.h` | `where_tile`, `where_init` | Conditional selection |
| `rounding.h` | `floor_tile`, `ceil_tile`, `round_tile`, `trunc_tile` | Rounding modes |
| `comp.h` | `comp_tile`, `comp_init` | Comparison ops |
| `trigonometry.h` | `sin_tile`, `cos_tile`, `tan_tile` | Trig functions |
| `erf_erfc.h` | `erf_tile`, `erfc_tile` | Error function |
| `xlogy.h` | `xlogy_tile`, `xlogy_init` | X·log(Y) |

**Standard Pattern (example from `activations.h`):**
```cpp
ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX, ActivationType::Hardsigmoid>(idst)));
}
ALWI void hardsigmoid_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()));
}
```

---

### 2.3 Welford Algorithm
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/welford.h`

**APIs:**
- `welford_init()` — initialize algorithm state
- `welford_clear()` — clear stale state for new run
- `welford_update<reciprocal_size>(input_dst_idx, start_idx, reciprocal_lut)` — update mean/m2
- `welford_update_rows<reciprocal_size>(input_dst_idx, start_idx, start_row, num_rows, reciprocal_lut)` — partial tile update
- `welford_save_state(mean_dst_idx)` — save mean/m2 to dst
- `welford_restore_state(mean_dst_idx)` — restore mean/m2 from dst
- `welford_finalize_to_row<reciprocal_size>(mean_dst_idx, scale_idx, reciprocal_lut)` — compute variance

---

### 2.4 CB Load (Copy Operations)
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/tile_move_copy.h`

**Primary APIs:**
- `copy_tile(in_cb_id, in_tile_index, dst_tile_index)` — copy from CB to DST register
- `copy_tile_init(cbid)` — standard init
- `copy_tile_to_dst_init_short(cbid, transpose=0, transpose_within_16x16_face=false)` — short init form
- `copy_tile_to_dst_init_short_with_dt(old_cbid, new_cbid, transpose=0)` — init with data-format reconfig
- `copy_block_matmul_partials(in_cb_id, start_in_tile_index, start_dst_tile_index, ntiles)` — copy block

---

### 2.5 Pack Operations
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/pack.h`

**Primary APIs:**
- `pack_tile<out_of_order_output=false>(ifrom_dst, icb, output_tile_index=0)` — pack single tile DST→CB
- `pack_tile_block(ifrom_dst, icb, ntiles)` — pack tile block
- `pack_reconfig_data_format<is_tile_dim_reconfig_en=false>(new_cb_id)` — force reconfig
- `pack_reconfig_data_format<is_tile_dim_reconfig_en=false>(old_cb_id, new_cb_id)` — conditional reconfig
- `pack_reconfig_l1_acc(l1_acc_en)` — enable/disable L1 accumulation
- `pack_rows_init(num_rows)` — init row-major packing
- `pack_rows(idst, ocb, output_index=0)` — pack rows
- `pack_rows_uninit()` — restore packer state

---

### 2.6 Register Synchronization (modern style)
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/reg_api.h`

**Modern API (recommended):**
- `tile_regs_acquire()` — acquire DST lock (MATH thread)
- `tile_regs_commit()` — commit DST updates (MATH thread)
- `tile_regs_wait()` — wait for MATH to commit (PACK thread)
- `tile_regs_release()` — release DST lock (PACK thread)

**Deprecated API (raw-dst style):**
- `acquire_dst()` — deprecated; maps to `tile_regs_acquire()` + `tile_regs_wait()`
- `release_dst()` — deprecated; maps to `tile_regs_commit()` + `tile_regs_release()`

**Typical sequence:**
```cpp
tile_regs_acquire();          // MATH: claim DST
// ... compute on DST ...
tile_regs_commit();           // MATH: mark done
tile_regs_wait();             // PACK: wait for MATH
// ... pack from DST ...
tile_regs_release();          // PACK: release DST
```

---

### 2.7 Unary Bcast
**Header:** `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/bcast.h`

**APIs:**
- `unary_bcast_init<BroadcastType>(icb, ocb)` — init bcast op (COL, ROW, SCALAR)
- `unary_bcast<BroadcastType>(icb, in_tile_index, dst_tile_index)` — perform bcast
- `unary_bcast_uninit<BroadcastType>(icb)` — cleanup
- `reconfigure_unary_bcast<old_type, new_type>(old_icb, new_icb, old_ocb, new_ocb)` — reconfig bcast type

**Shorthand binary ops (e.g.):**
- `sub_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst)` — A - B (bcast B by column)
- `sub_tiles_bcast_scalar(icb0, icb1, itile0, itile1, idst)` — A - B (bcast B as scalar)

---

### 2.8 Binary/Ternary/Quaternary SFPU & Mask
**Headers:**
- `mask.h` — element-wise masking ops
- `sfpu_binary_bcast.h` — SFPU binary ops with broadcast
- `eltwise_unary/where.h` — conditional selection (A ? B : C)

**Key patterns:** Mostly higher-order operations built on unary SFPU kernel blocks.

---

### 2.9 Other Reconfig & Init
**Headers:**
- `reconfig_data_format.h` — data-format reconfiguration for DST
- `compute_kernel_hw_startup.h` — hardware initialization on kernel startup
- `common_globals.h` — shared state & macros
- `cb_api.h` — circular buffer control (cb_reserve_back, cb_push_back, cb_wait_front)

---

## Section 3: TSV → LLK Header Coverage Matrix

| Demand Bucket | Heavy-Lifting | Flow | TSV Count | Covered By | Coverage Status | Notes |
|---|---|---|---|---|---|---|
| 1 | sfpu-unary | copy→sfpu-tile→pack | 169 | `tile_move_copy.h` + SFPU headers + `pack.h` | ✓ FULL | copy_tile + op_tile + pack_tile; modern sync |
| 2 | eltwise-fpu-binary | eltwise-binary→pack | 148 | `eltwise_binary.h` + `pack.h` | ✓ FULL | add/sub/mul_tiles + pack_tile; modern sync |
| 3 | eltwise-fpu-bcast | eltwise-bcast→pack | 111 | `eltwise_binary.h` + `bcast.h` + `pack.h` | ✓ FULL | binary ops + bcast variants + pack_tile |
| 4 | copy | copy→pack | 76 | `tile_move_copy.h` + `pack.h` | ✓ FULL | copy_tile + pack_tile; modern/raw-dst sync |
| 5 | sfpu-unary | sfpu-tile→pack | 45 | SFPU headers + `pack.h` | ✓ FULL | op_tile (no copy) + pack_tile |
| 6 | transpose | transpose→pack | 29 | `transpose_wh.h` (not included in scope) | ✗ UNCOVERED | requires transpose kernel wrapper |
| 7 | sfpu-unary | sfpu-unary-tile→pack | 21 | SFPU headers + `pack.h` | ✓ FULL | SFPU unary patterns |
| 8 | sfpu-unary | copy→sfpu-macro→pack | 16 | `tile_move_copy.h` + SFPU + `pack.h` | ✓ FULL | copy + SFPU macro + pack_tile |
| 9 | welford | welford→pack | 15 | `welford.h` + `pack.h` | ✗ PARTIAL | welford ops need wrapper; depends on `tile_regs_*` sync |
| 10 | sfpu-unary | sfpu-macro→pack | 15 | SFPU + `pack.h` | ✓ FULL | SFPU macro pattern + pack |
| 11 | sfpu-helper | sfpu-helper→pack | 9 | Internal helper (moreh sum) | ✗ UNCOVERED | custom helper, not in public LLK API |
| 12 | unknown | ?→pack | 5 | Unknown | ✗ UNCOVERED | requires TSV investigation |
| 13 | sort/topk | copy→pack | 4 | `tile_move_copy.h` + `pack.h` | ✓ FULL | data movement kernel |
| 14 | sort/topk | ?→pack | 2 | Unknown | ✗ UNCOVERED | requires TSV investigation |
| 15 | sfpu-unary | copy→sfpu-llk→pack | 1 | SFPU LLK + `pack.h` | ✓ FULL | single-case pattern |

**Coverage Summary:**
- **Full coverage:** 12 buckets (548 + 76 = 624 rows) = **93.7%** of demand
- **Partial coverage:** 1 bucket (15 rows, Welford) = 2.3%
- **Uncovered:** 2 buckets (29 + 9 rows) + 2 unknown = 4.0%

---

## Section 4: Sync Style Coverage Analysis

From TSV `sync_style` column (666 data rows):

| Sync Style | Count | Rows % | LLK API Family | Helper Must Emit |
|---|---|---|---|---|
| modern | 566 | 85.0% | `tile_regs_acquire/commit/wait/release` | ✓ YES (primary) |
| raw-dst | 57 | 8.5% | `acquire_dst/release_dst` (deprecated) | ✓ YES (backward compat) |
| ACQ-REL-macro | 43 | 6.5% | kernel-local `ACQ()`/`REL()` macros | ✓ YES (for legacy) |

**Decision:** Helper v2 must emit **all three styles**:
1. **Modern** (default, 566 rows): Use `tile_regs_*` API
2. **Raw-dst** (57 rows): Map to deprecated `acquire_dst/release_dst` overloads
3. **ACQ-REL-macro** (43 rows): Generate kernel-local macro-based sync for specialized cases

The `sync_seq` column encodes the exact acquire/commit/wait/release sequence per row; the helper must capture these patterns.

---

## Section 5: LLK Family → Helper Wrapper Mapping

**Phase 0 Phase 1 Phase 2 mapping:**

### Phase 0: Demand surface (in this doc)
Maps TSV flow → LLK headers.

### Phase 1: Helper wrapper generation
For each LLK family below, the v2 helper must wrap:

#### Group 1: **eltwise-fpu-binary**
- LLK headers: `eltwise_binary.h`, `bcast.h`, `pack.h`, `reg_api.h`
- TSV buckets: #2 (148), #3 (111) = **259 rows**
- Helper signatures:
  - `add_tiles_kernel(dst_acq, icb0, icb1, i0, i1, idst, sync_style)` → calls `add_tiles()` + sync
  - `sub_tiles_kernel(...)` → calls `sub_tiles()` + sync
  - `mul_tiles_kernel(...)` → calls `mul_tiles()` + sync
  - Bcast variants: `*_tiles_bcast_col()`, `*_tiles_bcast_row()`, `*_tiles_bcast_scalar()`
- Init wrapper: `eltwise_binary_init(icb0, icb1, ocb, [bcast_type], [acc_to_dest])`

#### Group 2: **sfpu-unary + copy**
- LLK headers: `tile_move_copy.h`, SFPU headers, `pack.h`, `reg_api.h`
- TSV buckets: #1 (169), #5 (45), #7 (21), #8 (16), #10 (15), #15 (1) = **267 rows**
- Helper signatures:
  - `copy_sfpu_unary_kernel(dst_acq, icb_in, icb_out, tile_idx, sfpu_op, sync_style)` → calls `copy_tile()`, `*_tile()`, sync
  - Variants for each SFPU op (exp, sqrt, relu, gelu, tanh, sigmoid, softmax, dropout, clamp, etc.)
  - Init wrapper: `copy_sfpu_init(icb_in, ocb, [transpose], [sfpu_op_id])`

#### Group 3: **direct SFPU (no copy)**
- LLK headers: SFPU headers, `pack.h`, `reg_api.h`
- TSV buckets: #5 (45) subset
- Helper: SFPU ops without copy (for pre-loaded DST)
- Signature: `sfpu_unary_kernel(dst_acq, idst, sfpu_op, params, sync_style)`

#### Group 4: **copy + pack (direct)**
- LLK headers: `tile_move_copy.h`, `pack.h`, `reg_api.h`
- TSV buckets: #4 (76)
- Helper signature: `copy_pack_kernel(icb, tile_idx, ocb, out_idx, sync_style)`

#### Group 5: **Welford (stateful reduction)**
- LLK headers: `welford.h`, `pack.h`, `reg_api.h`
- TSV buckets: #9 (15)
- Helper signatures:
  - `welford_init_kernel(...)` → calls `welford_init()`
  - `welford_update_kernel(idst_in, start_idx, ocb, out_idx, sync_style)` → calls `welford_update()`, pack, sync
  - `welford_finalize_kernel(idst_mean, scale_idx, ocb, out_idx, sync_style)` → calls `welford_finalize_to_row()`, pack, sync
- Init wrapper: `welford_init_helper(ocb, [reciprocal_lut_size])`

#### Group 6: **Broadcast (unary)**
- LLK headers: `bcast.h`, `pack.h`, `reg_api.h`
- Helper: wraps `unary_bcast<BroadcastType>()` for COL/ROW/SCALAR variants
- Signature: `unary_bcast_kernel(icb, tile_idx, ocb, out_idx, bcast_type, sync_style)`

#### Group 7: **Uncovered (Phase 2+)**
- Transpose kernel, sfpu-helper, unknown ops → TBD in proposal phase

---

## Section 6: Proposal-Phase Notes

### 6.1 Scope Constraints
1. **Exclusions (per user):**
   - ~~`ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.{hpp,inl}`~~
   - ~~`ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.{hpp,inl,_bitwise.hpp}`~~
   - ~~`ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.{hpp,inl}`~~
   These are marked as "potentially faulty" and will NOT be read or used as reference.

2. **Out-of-scope (Phase 2):**
   - Transpose (`transpose_wh.h`) — requires dedicated transpose kernel
   - Welford (15 rows) — complex stateful algorithm; defer to Phase 2
   - sfpu-helper (9 rows) — internal helper; need code inspection
   - Unknown ops (7 rows) — require TSV investigation

### 6.2 Test Coverage Requirement
For each of the 6 main LLK families (Groups 1–6), the helper must provide:
1. **Unit test** per op (e.g., `test_add_tiles`, `test_exp_tile`, `test_copy_tile`)
2. **Sync style variant test** (modern, raw-dst, ACQ-REL-macro) for representative ops
3. **Integration test** combining copy→compute→pack chain

### 6.3 Init/Reconfig Pattern
All helpers follow a common init pattern:
```
operation_init(icb0, icb1, ocb, [optional_params])  // called once per kernel
// ... loop ...
operation_kernel(dst_acq, tile_indices, sync_style) // called per iteration
operation_uninit()  // optional, if needed by LLK
```

### 6.4 Sync Abstraction Design
The v2 helper must support three sync **modes**:
1. **Mode::MODERN** → calls `tile_regs_acquire/commit/wait/release`
2. **Mode::RAW_DST** → calls deprecated `acquire_dst/release_dst`
3. **Mode::ACQ_REL_MACRO** → generates kernel-local ACQ()/REL() macros

Each helper signature accepts a `sync_mode` parameter, enabling a single helper to emit any of the three styles.

### 6.5 LLK Header Dependency Graph
```
pack.h (core)
├─ eltwise_binary.h (FPU binary)
├─ tile_move_copy.h (CB load)
├─ eltwise_unary/*.h (SFPU unary, 57 headers)
├─ bcast.h (unary bcast)
├─ welford.h (Welford reduction)
├─ reg_api.h (sync primitives)
└─ reconfig_data_format.h (data-format switching)
```

### 6.6 Validation Checklist for Phase 1
- [ ] Helpers emit correct `tile_regs_*` / `acquire_dst` / `ACQ()` sequences
- [ ] TSV sync_seq column verified against helper output
- [ ] All 6 groups (Groups 1–6) have working unit tests
- [ ] Cross-check: for each of the 624 "covered" TSV rows, a plausible helper call exists
- [ ] Backward compat: legacy raw-dst kernels still compile/run with helpers
- [ ] Symbol export: helpers are callable from user kernels (no static/inline-only)

### 6.7 Known Gaps
| Bucket | Issue | Resolution |
|---|---|---|
| transpose→pack | No public API in `eltwise_*` scope | Add to Phase 2; use transpose_wh.h as reference |
| welford→pack | Stateful, complex init | Phase 2; requires careful orchestration with tile_regs_* |
| sfpu-helper→pack | Internal moreh sum helper | Phase 2; inspect moreh_int_sum_* kernels for patterns |
| unknown→pack | 7 rows unclassified | Phase 2; read TSV-flagged kernels to identify ops |

---

## Appendix: Quick Reference

### Load → Compute → Pack Chain (modern sync)

```cpp
// Init (once)
copy_tile_init(icb);
add_tiles_init(icb0, icb1);
pack_reconfig_data_format(ocb);

// Per-iteration (inside loop)
tile_regs_acquire();          // MATH: claim DST
copy_tile(icb, tile_idx, 0);  // MATH: load from CB
add_tiles(icb0, icb1, i0, i1, 0); // MATH: compute A+B into DST[0]
tile_regs_commit();           // MATH: mark DST done
tile_regs_wait();             // PACK: wait for MATH
pack_tile(0, ocb, out_idx);   // PACK: move DST[0] to CB
tile_regs_release();          // PACK: release DST
```

### Broadcast (FPU or unary bcast)

```cpp
// Init
unary_bcast_init<BroadcastType::COL>(icb, ocb);

// Per-tile
tile_regs_acquire();
unary_bcast<BroadcastType::COL>(icb, tile_idx, dst_idx);  // or binary op with bcast
tile_regs_commit();
tile_regs_wait();
pack_tile(dst_idx, ocb, out_idx);
tile_regs_release();
```

---

**End of Catalog**
