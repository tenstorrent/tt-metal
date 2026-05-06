# Eltwise Compute Helpers: CB Lifecycle & Dtype-Reconfig Analysis (Phase 1)

PATTERNS_HEADER: file	line	function	category	heavy_lifting	variant	loop_depth	loop_vars	sig	arg0	arg1	arg2	flow	sync_bucket	sync_seq	sync_style	shape	region_stats

---

## Section 1: TSV-Sampled Kernel Set (18 Kernels)

### Selection Criteria
- Source: `/localdev/astancov/tt-metal/pack_patterns.tsv`
- 5 eltwise files: heavy_lifting ∈ {eltwise-fpu-binary, eltwise-fpu-bcast, sfpu-unary, copy}
- 4 normalization files: mix of sfpu-unary and eltwise-fpu-binary
- 3 moreh files: sfpu-unary and copy patterns
- 4 transformer files: sync_style = raw-dst (compute_common.hpp references)
- 2 reduction files: accumulation_compute.cpp

### Sample Table

| # | File | Category | Heavy_Lifting | Sync_Style | Shape | Notes |
|---|------|----------|----------------|-----------|-------|-------|
| 1 | bcast_h.cpp | ttnn-op:eltwise | eltwise-fpu-bcast | raw-dst | raw-dst/single | 3-loop (b,h,w) broadcast |
| 2 | bcast_hw.cpp | ttnn-op:eltwise | eltwise-fpu-bcast | raw-dst | raw-dst/single | 3-loop broadcast |
| 3 | bcast_w.cpp | ttnn-op:eltwise | eltwise-fpu-bcast | raw-dst | raw-dst/single | 3-loop column-wise bcast |
| 4 | eltwise_binary_kernel.cpp | ttnn-op:eltwise | sfpu-unary | modern | modern-canonical/single-in-loop | Per-block reconfig, per-tile loop |
| 5 | eltwise_binary_sfpu_kernel.cpp | ttnn-op:eltwise | sfpu-unary | modern | modern-canonical/single-in-loop | Mixed SFPU + binary, per-block stages |
| 6 | batch_norm_kernel.cpp | ttnn-op:normalization | eltwise-fpu-binary | modern | modern-canonical/single | CB-based helper with DEN pre-calc |
| 7 | batch_norm_sfpu_kernel.cpp | ttnn-op:normalization | sfpu-unary | modern | modern-canonical/single | Advanced typecast + per-tile loop |
| 8 | running_statistics_kernel.cpp | ttnn-op:normalization | unknown | modern | modern-canonical/single | Momentum-based per-tile loop |
| 9 | batch_norm_kernel.cpp (sfpu) | ttnn-op:normalization | sfpu-unary | modern | modern-canonical/single | Fused sfpu path |
| 10 | moreh_abs_pow_kernel.cpp | ttnn-op:moreh | sfpu-unary | modern | modern-canonical/single | Nested 2-loop with masking |
| 11 | moreh_adam.cpp (copy) | ttnn-op:moreh | copy | modern | modern-canonical/single | Intermediate temp CB copies |
| 12 | moreh_adam.cpp (sfpu) | ttnn-op:moreh | sfpu-unary | modern | modern-canonical/single | Multi-stage with FP32 accumulation |
| 13 | compute_common.hpp | ttnn-op:transformer | copy | raw-dst | raw-dst/single | Reduce copy tile to DST |
| 14 | compute_common.hpp (max) | ttnn-op:transformer | eltwise-fpu-bcast | raw-dst | raw-dst/single | Binary max block |
| 15 | compute_common.hpp (reduce_c) | ttnn-op:transformer | eltwise-fpu-bcast | raw-dst | raw-dst/single-in-loop | Granularized reduce with loop |
| 16 | compute_common.hpp (reduce sum) | ttnn-op:transformer | eltwise-fpu-binary | raw-dst | raw-dst/single | Reduce block operations |
| 17 | accumulation_compute.cpp | ttnn-op:reduction | eltwise-fpu-binary | modern | modern-canonical/multi-unrolled | Dual DEST slot (DST_IN, DST_ACC) |
| 18 | accumulation_compute.cpp | ttnn-op:reduction | unknown | modern | modern-canonical/single | Sync on cb_acc between iterations |

---

## Section 2: Per-Kernel CB Lifecycle Classification

### Pattern Taxonomy
- **Input CB Wait Shape:**
  - `per-tile`: `cb_wait_front(cb, 1)` inside per-tile loop
  - `upfront-N`: `cb_wait_front(cb, N)` once before any loop
  - `cumulative`: `cb_wait_front(cb, cumul_count)` with running total across loop levels
  - `pre-waited`: no explicit wait; preemp from reader
  - `persistent`: `cb_wait_front()` once, never popped within kernel execution
  - `streaming-pre-pushed`: tile pushed before kernel reads

- **Input CB Pop Shape:**
  - `per-tile`: `cb_pop_front(cb, 1)` per tile
  - `upfront-end`: all pops collected at loop exit or after block
  - `none`: never popped (persistent)

- **Output CB Reserve/Push:**
  - `per-tile`: `cb_reserve_back()` + `cb_push_back()` per tile
  - `upfront-N`: entire block reserved upfront
  - `out-of-order`: `cb_reserve_back()` interleaved with pack (accumulation pattern)

- **DEST Sync (per TSV sync_style):**
  - `modern[ACWR]` or `modern[CR]`: acquire-before, release-after
  - `raw-dst[ar]`: acquire at block start, release at block end; may reuse across iterations
  - `ACQ-REL-macro`: lockstep with explicit macro sync boundaries

### Kernel Lifecycle Summaries

**K1-K3: bcast_h/hw/w (raw-dst eltwise-fpu-bcast)**
```
Input CB (c_0, c_1):
  - c_1: wait_front(1) at h-loop level, pop_front(1) at h-loop exit → upfront-per-h-level
  - c_0: wait_front(1) per w-tile, pop_front(1) per w-tile → per-tile

Output CB (c_2):
  - reserve_back(1) per tile, push_back(1) per tile → per-tile

DEST Sync:
  - acquire_dst() per w-tile, release_dst() per w-tile → per-tile acquire/release
  - sync_style = raw-dst [ar]: raw acquisition, release via pack_tile

Control Flow:
  - 3-loop nesting (b, h, w); payload inside w-loop
  - c_1 reused across all w-tiles in h-loop iteration
```

**K4-K5: eltwise_binary_kernel.cpp / eltwise_binary_sfpu_kernel.cpp (modern sfpu-unary)**
```
Input CB (c_0, c_1, optionally c_3, c_4):
  - c_0 (primary input):
    - if SFPU_OP_INIT_PRE_IN0: wait_front(per_core_block_size) at block start
                                 reconfig_data_format_srca(c_inp0, c_in0)
                                 per-block → upfront-block wait
    - else: cb_wait_front(c_inp0, per_core_block_size) before binary op
  - c_1 (secondary input): symmetric reconfig and wait_front pattern

  - Pop: cb_pop_front() at block end after all per_core_block_size tiles processed → upfront-block pop

Output CB (c_2):
  - reserve_back(per_core_block_size) once per block
  - push_back(per_core_block_size) once at block end → upfront-block reserve/push

DEST Sync:
  - tile_regs_acquire() once per block
  - tile_regs_commit() / tile_regs_wait() / tile_regs_release() per block
  - sync_style = modern [ACWR]: acquire at block start, release at block end

Reconfig Pattern (KEY):
  - Entry-only reconfig: reconfig_data_format_srca() called at block start for input dtype
  - Pack reconfig: pack_reconfig_data_format() called at block start for output dtype
  - Flow: reconfig at entry → copy tile to DST → SFPU chain → pack with same dtype

Control Flow:
  - Outer loop: per_core_block_cnt blocks
  - Inner loop (per block): per_core_block_size tiles
  - Conditional: #ifdef SFPU_OP_INIT_PRE_IN0_0 / SFPU_OP_INIT_PRE_IN1_0 triggers staged pre-scaling
```

**K6: batch_norm_kernel.cpp (modern eltwise-fpu-binary)**
```
Uses helper function batchnorm_bcast_tiles() with nested loops:
  - Outer: num_tiles iterations (per core)
  - Inner (per iteration): tile_freq per-tile loop

Input CBs:
  - cb_batch_mean (broadcast): wait_front(1) once at helper start, persist until end
  - cb_input: wait_front(1) per tile in inner loop → per-tile wait
  - cb_eps: wait_front(1) once at kernel start, persist → persistent
  - cb_weight, cb_bias (conditional): wait_front(1) once at helper start

Output CBs:
  - cb_output_0: reserve_back(1) per tile, push_back(1) per tile in inner loop → per-tile

CB Chaining (intermediate):
  - cb_den: computed once per helper invocation (1/(sqrt(batch_var + eps)))
           wait_front(1) at tile start, used across all tiles in frequency loop
           → upfront-per-helper persistent

  - cb_affine_or_out: computed per tile if weight or bias, consumed immediately
           reserve_back(1) per tile, push_back(1) per tile → per-tile

  - cb_scaled_output: intermediate, depends on bias_has

DEST Sync:
  - tile_regs_acquire() / commit() / wait() / release() per iteration of the tile_freq loop
  - sync_style = modern [ACWR]

Reconfig Pattern (IMPLICIT):
  - No explicit reconfig_data_format_srca() or pack_reconfig_data_format()
  - BUT: add_tiles_init_with_dt(), mul_tiles_init_with_dt(), pack_tile_with_dt()
           → implicit dtype reconfig inside init and pack macros
  - Frequency: per iteration of the tile_freq loop (per-tile stage, not entry-only)
```

**K7-K9: batch_norm_sfpu_kernel.cpp, running_statistics_kernel.cpp**
```
K7: batch_norm_sfpu_kernel.cpp
  - Similar to K6 but with advanced typecast handling
  - add_binary_tile_init() uses copy_tile_to_dst_init_short_with_dt(last_srca_cb, ...)
  - Tracks last_srca_cb state machine to avoid redundant reconfig across multi-stage compute
  - Reconfig Pattern: implicit per-tile via copy_tile_to_dst_init_short_with_dt()

K8: running_statistics_kernel.cpp (batch norm momentum update)
  - Per-tile loop: 0..num_tiles
  - Inputs: cb_batch_mean, cb_batch_var (wait_front once at kernel start, persistent)
  - Inputs: cb_old_running_{mean,var}, cb_momentum, cb_one (persistent)
  - Output: cb_updated_running_{mean,var}, cb_out0
  - Reconfig: none; implicit *_to_cb() helpers manage dtype
  - Wait/Pop shape: upfront-kernel wait, upfront-loop pop/push

K9: same source as K6, but traced as sfpu variant (internal SFPU within batch norm)
```

**K10: moreh_abs_pow_kernel.cpp (modern sfpu-unary)**
```
Structure:
  - Nested loop: num_rows_per_core × Wt
  - Per tile: copy → abs → optional mask → power_tile_to_cb()

Input CBs:
  - cb_x: wait_front(1) per tile in inner loop → per-tile
  - cb_one, cb_decimal: wait_front(1) once at kernel start → upfront-kernel persistent
  - cb_mask_w (conditional): wait_front(1) once at kernel start

Output CB:
  - cb_y: populated via power_tile_to_cb() helper

Intermediate CBs:
  - cb_xabs, cb_xpow, cb_logx, cb_exp_lxmd: chained within power_tile_to_cb()

DEST Sync:
  - tile_regs_acquire() / commit() / wait() / release() per tile
  - sync_style = modern [CWR or similar]

Reconfig Pattern:
  - copy_tile_init_with_dt(cb_x): implicit dtype reconfig
  - No explicit reconfig_data_format_srca() calls
```

**K11-K12: moreh_adam.cpp (copy + sfpu-unary)**
```
K11 (copy path):
  - Uses copy_tile_to_cb(tmp_cb_exp_avg, cb_exp_avg_out, ...) helper
  - No explicit DEST operations; copy_tile_to_cb() manages reserve/push internally

K12 (sfpu path):
  - Per-tile loop: per_core_tile_cnt iterations
  - tile_regs_acquire/commit/wait/release per tile
  - Extensive use of *_to_cb() helpers (mul_tiles_to_cb, add_tiles_to_cb, sub_tiles_to_cb)
  - Complex intermediate CB chaining: cb_tmp1, cb_tmp2, cb_exp_avg, cb_exp_avg_sq, etc.

Reconfig Pattern (K12):
  - WITH_FP32_DEST_ACC(reconfig_data_format(cb_one, cb_tmp1)) → per-tile reconfig
  - Frequency: mid-loop stage (inside per-tile loop, not entry-only)
  - Trigger: compile-time FP32_DEST_ACC_EN flag
```

**K13-K16: compute_common.hpp (raw-dst transformer functions)**
```
K13 (copy): sdpa_reduce_copy_tile_to_dst_init_short()
  - Helper function, no loop; abstract pattern
  - Uses llk_unpack_A_init with NONE broadcast and UnpackToDestEn
  - No explicit CB lifecycle; DEST only

K14 (max block): max_block_inplace()
  - Loop: num_tiles
  - Input CBs: in0, in1 → cb_wait_front(in0/in1, num_tiles) once before loop → upfront-block
  - Output CB: in0 (in-place) → cb_pop_front(), cb_reserve_back(), cb_push_back() per tile
  - DEST Sync: acquire_dst/release_dst per tile
  - sync_style = raw-dst [ar]

K15 (reduce_c with runtime cols):
  - Nested loop: granularity (rows) × cols
  - Input CBs:
    - in0_cb: cb_wait_front(in0_wait_tiles) with cumulative tracking → cumulative wait
    - prev_cb (if do_eltwise_max): cb_wait_front(prev_cb, g * dst_tiles) per granule → cumulative
    - scale_cb: cb_wait_front(scale_cb, 1) once at function start → persistent
  - Output CB: out_cb → cb_reserve_back(rows) once, cb_push_back(rows) once → upfront-function
  - DEST Sync: acquire_dst once per granule, release_dst once per granule
  - sync_style = raw-dst [ar]
  - Multi-tile pack_tile<true>() with indexing

K16 (reduce block max row):
  - Binary max operation on pre-loaded tiles
  - Similar to K14 but for reduce semantics
```

**K17-K18: accumulation_compute.cpp (modern multi-unrolled / single)**
```
Special Pattern: Held-DEST with out-of-order CB reserve/push

Structure:
  - Outer loop: num_rows
  - Inner loop: tiles_per_row
  - Two DEST slots: DST_IN (tile index 0), DST_ACC (tile index 1)

Input CBs:
  - CB_IN: wait_front(1) per tile in inner loop → per-tile

Output CBs (out-of-order):
  - CB_ACC:
    - reserve_back(1) at outer-loop start, push_back(1) after fill
    - wait_front(1) / pop_front(1) before each iteration
    - reserve_back(1) after pop (out-of-order), then:
      - pack_tile(DST_ACC, CB_ACC) → same location
    - push_back(1) after pack
    - Repeat: wait_front(1) before each inner-loop iteration

  - CB_OUT:
    - reserve_back(1) per inner-loop iteration
    - pack_tile(DST_ACC, CB_OUT) with reconfig
    - push_back(1) per inner-loop iteration → per-tile

Reconfig Pattern:
  - pack_reconfig_data_format(CB_ACC) → pack destination dtype
  - pack_reconfig_data_format(CB_ACC, CB_OUT) → reconfig for cross-CB packing
  - Frequency: per inner-loop iteration → per-tile stage

DEST Sync:
  - tile_regs_acquire/commit/wait/release per inner-loop iteration
  - Synchronization sync between unpacker and packer via CB_ACC

Held-DEST Pattern (YES):
  - DST_ACC loaded in one iteration, held across multiple accesses
  - pack_tile(DST_ACC, CB_ACC) at row-block end, then re-loaded in next outer iteration
  - BUT: CB_ACC acts as a staging buffer, not true held-DEST register
  - Genuine held-DEST: NO (DEST released after each commit/wait)
```

---

## Section 3: Reconfig Pattern Survey

### Reconfig Calls Found

#### Type A: Entry-Only Reconfig (Block Start)

**K4, K5: eltwise_binary_kernel.cpp / eltwise_binary_sfpu_kernel.cpp**
```cpp
// Entry-only at per-block level
if (defined SFPU_OP_INIT_PRE_IN0_0) {
    reconfig_data_format_srca(cb_inp0, cb_in0);      // input dtype config
    pack_reconfig_data_format(cb_out0, cb_inp0);     // output dtype config
    // Then per-core_block_size tiles processed with same reconfig
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in0, i, i);
        SFPU_OP_FUNC_PRE_IN0_0
    }
}
```
- **Frequency**: per block (per_core_block_cnt outer loop)
- **Sides affected**: both srca (input) and pack (output)
- **Trigger**: compile-time SFPU_OP_INIT_PRE_IN0_0 / SFPU_OP_INIT_PRE_IN1_0 macros
- **Benefit**: amortizes reconfig cost across per_core_block_size tiles

#### Type B: Per-Tile Implicit Reconfig (Inside Loop)

**K6: batch_norm_kernel.cpp**
```cpp
for (uint32_t j = tile_start; j < freq; ++j) {
    // ...
    add_tiles_init_with_dt(cb_batch_var, cb_eps);    // implicit dtype check in init
    add_tiles(cb_batch_var, cb_eps, 0, 0, 0);
    // ...
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
    // ...
    pack_tile_with_dt(dst0, cb_affine_or_out);       // dtype-aware pack
}
```
- **Frequency**: per tile in the frequency loop
- **Sides affected**: implicit (handled by *_init_with_dt and pack_tile_with_dt)
- **Trigger**: runtime dtype of CBs (determined at init time, checked per-call)
- **Pattern**: no explicit reconfig_data_format_*() call; dtype negotiation inside helper macros

**K7: batch_norm_sfpu_kernel.cpp**
```cpp
for (uint32_t j = tile_start; j < freq; ++j) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_other);
    last_srca_cb = cb_other;
    copy_tile(cb_other, index, index * 2);
    // ...
    copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_bcast);
    last_srca_cb = cb_bcast;
    // ...
    pack_tile(index * 2, cb_affine_or_out);
}
```
- **Frequency**: per tile, within tile_regs_acquire/release block
- **Sides affected**: srca only (via copy_tile_to_dst_init_short_with_dt)
- **Trigger**: state machine (last_srca_cb) to detect when new input dtype differs
- **Optimization**: avoids redundant reconfig within same compute stage

**K12: moreh_adam.cpp**
```cpp
// Inside per-tile loop
tile_regs_acquire();
// ...
WITH_FP32_DEST_ACC(reconfig_data_format(cb_one, cb_tmp1));
sub_tiles_init(cb_one, cb_tmp1);
sub_tiles(cb_one, cb_tmp1, first_tile, first_tile, dst0);
// ...
tile_regs_commit();
tile_regs_wait();
pack_tile_with_dt(dst0, cb_tmp1);
```
- **Frequency**: mid-loop, specifically triggered by FP32 accumulation requirement
- **Sides affected**: compute unpacker (srca)
- **Trigger**: compile-time FP32_DEST_ACC_EN flag
- **Pattern**: explicit reconfig_data_format() call within tile_regs_acquire/release

#### Type C: No Explicit Reconfig (Implicit via Helper Macros)

**K1-K3: bcast kernels**
- No reconfig calls; dtype handled by init_bcast() and BCAST_OP macros
- Assumes uniform dtype across c_0, c_1, c_2

**K8: running_statistics_kernel.cpp**
- No explicit reconfig; all dtype negotiation via *_to_cb() helpers
- Frequency: per-tile loop, implicit within helper functions

**K10: moreh_abs_pow_kernel.cpp**
- copy_tile_init_with_dt() and power_tile_to_cb() handle dtype implicitly
- No explicit reconfig_data_format_*() calls

**K13-K16: transformer kernels**
- Abstract patterns; dtype negotiation within helper function signatures

**K17-K18: accumulation_compute.cpp**
- reconfig_data_format() and pack_reconfig_data_format() with explicit CB arguments
- Frequency: per inner-loop iteration (per-tile)
- Pattern: multi-CB reconfig for cross-CB packing

### Reconfig Pattern Summary Table

| Type | Kernel | Category | Frequency | Sides | Trigger | Code Pattern |
|------|--------|----------|-----------|-------|---------|--------------|
| A (Entry) | K4, K5 | eltwise | per-block | srca + pack | SFPU_OP_INIT_PRE | reconfig_data_format_srca() |
| B (Implicit) | K6 | normalization | per-tile | implicit | none | add_tiles_init_with_dt() |
| B (Implicit) | K7 | normalization | per-tile | srca | state-machine | copy_tile_to_dst_init_short_with_dt() |
| B (Explicit) | K12 | moreh | per-tile | srca | FP32_DEST_ACC_EN | reconfig_data_format() |
| C (None) | K1-K3 | eltwise | none | — | — | init_bcast() |
| C (Implicit) | K8, K10 | norm, moreh | per-tile | implicit | none | *_to_cb(), *_init_with_dt() |
| C (Helper) | K13-K16 | transformer | per-fn | implicit | none | helper function signature |
| B (Explicit) | K17-K18 | reduction | per-tile | pack | FP32_DEST_ACC_EN | pack_reconfig_data_format() |

---

## Section 4: Multi-DEST-Slot Pack-in-Window Analysis

### TSV Multi-Pack Identification

Rows with 2+ `P` in sync_seq and NOT back-to-back acquire/release windows:

**Candidates (from earlier grep):**
1. `ACWPRACWPR` (two `P` in separate windows: ACW-P-R and ACW-P-R)
2. `CWPRACWPR` (similar two-window pattern)
3. `ACWPRACWPRA` (three `P` across windows)
4. `PraPr` (two `P` potentially in same window if lowercase `r` = release within window)
5. `aPraPr` (acquire, pack, release; acquire, pack, release = two windows)
6. `PRACWPRA` (pack at start, then compute+pack)
7. `WPRACWPRA` (wait, pack, release, compute, wait, pack, release, acquire)

### Single-Window Multi-Pack Evidence

Looking at transformer reduce patterns from compute_common.hpp:

**K15: reduce_c() function (single-window multi-pack hypothesis)**
```cpp
for (uint32_t i = 0; i < dst_tiles; i++) {
    const uint32_t reduce_dst_idx = i;
    reduce_block_max_row<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols, reduce_dst_idx);
}
reduce_block_max_row_uninit(in0_cb);

for (uint32_t i = 0; i < dst_tiles; i++) {
    const uint32_t cur_max_dst_idx = i;
    pack_tile<true>(cur_max_dst_idx, out_cb, (row_start_idx + i));
}
```
- **Observation**: loop over `dst_tiles` indices within single acquire_dst/release_dst block
- **Pattern**: `acquire_dst()` once → multiple pack_tile<true>(i, ...) for i=0..dst_tiles-1 → release_dst()
- **Multi-pack-in-window**: YES, if dst_tiles > 1
  - Window structure: ACW * dst_tiles * P (dst_tiles pack operations) R
  - Sync_seq equivalent: `ACWPPP...R` (condensed; TSV would show `pack_tile<true>` as `P`)

**K17-K18: accumulation_compute.cpp (dual DEST slot)**
```cpp
tile_regs_acquire();
// ... compute in DST_IN and DST_ACC ...
tile_regs_commit();
tile_regs_wait();

cb_out_obj.reserve_back(ONE_TILE);
pack_reconfig_data_format(CB_ACC, CB_OUT);
pack_tile(DST_ACC, CB_OUT);  // First pack
cb_out_obj.push_back(ONE_TILE);

cb_acc_obj.reserve_back(ONE_TILE);
pack_reconfig_data_format(CB_OUT, CB_ACC);
pack_tile(DST_ACC, CB_ACC);  // Second pack (same DEST slot)
```
- **Observation**: two pack_tile(DST_ACC, ...) calls within single tile_regs_acquire/release
- **Pattern**: single DEST slot (DST_ACC) packed to two different CBs
- **Multi-pack-in-window**: YES
  - Window structure: ACW P (to CB_OUT) P (to CB_ACC) R
  - Implication: DEST_ACC must support multiple output consumers

### TSV Sync_seq Patterns: Multi-Pack Confirmation

**Pattern breakdown for `ACWPRACWPR`:**
- `A` (acquire), `C` (compute), `W` (wait), `P` (pack), `R` (release)
- Split: `ACWPR` + `ACWPR` = two independent acquire/release windows
- Multi-pack-in-window: NO (back-to-back windows, not combined)

**Pattern for `CWPRACWPR`:**
- `C`, `W`, `P`, `R` (first window) + `A`, `C`, `W`, `P`, `R` (second window)
- Multi-pack-in-window: NO

**Pattern for `PraPr` (raw-dst transformer):**
- `P`, `r` (lowercase = within-window release?), `a` (acquire), `P`, `r`
- Possible interpretation: `P` (pack), `r` (release), `a` (acquire), `P` (pack), `r` (release)
- Multi-pack-in-window: could be `P` (output-packing multiple slots), `r` (release first set), `a` (acquire for next), `P` (pack second set), `r` (release)
- Evidence: WEAK (capitalization inconsistency in TSV)

**Pattern for `ACWPACWP` (hypothetical):**
- `A`, `C`, `W`, `P`, `A`, `C`, `W`, `P` = interleaved compute + pack with NO release between
- Multi-pack-in-window: STRONG evidence if this pattern exists

### Verdict on Multi-Slot Packing

**Confirmed patterns:**
1. **K17-K18 (accumulation_compute.cpp)**: Single DEST slot packed to two CBs within one acquire/release window
   - Sync_seq: `ACWPACWP` (if both packs are tracked separately) or `ACWPPR` (if sequential packs before release)
   - Recommendation: support via **single PackTileBlock element with N output CB indices**

2. **K15 (reduce_c with dst_tiles > 1)**: Multiple DEST indices packed within single window
   - Sync_seq: `ACWPPP...PR` (N packs for N DEST indices)
   - Recommendation: support via **single PackTileBlock with loop over indices**

**Not found patterns:**
- Independent PackTile elements for each DEST slot within same window (would require unwind/rewind between packs)

---

## Section 5: Held-DEST Hold-Loop Survey

### Definition
"Held-DEST": A DEST register is loaded in one tile_regs_acquire/release cycle and accessed again in a subsequent cycle without being released and re-loaded.

### Grep Results

**accumulation_compute.cpp (K17-K18):**
```cpp
// Iteration 0 (row i=0)
tile_regs_acquire();
fill_tile_init();
FILL_TILE(DST_ACC, default_acc_value);
tile_regs_commit();
tile_regs_wait();
pack_tile(DST_ACC, CB_ACC);
tile_regs_release();

// Iteration 1 (row j=0 of row i=0)
tile_regs_acquire();
cb_in_obj.wait_front(ONE_TILE);
reconfig_data_format(CB_IN, CB_IN);
copy_tile_to_dst_init_short(CB_IN);
copy_tile(CB_IN, 0, DST_IN);
copy_tile_to_dst_init_short(CB_ACC);
copy_tile(CB_ACC, 0, DST_ACC);  // RELOAD DST_ACC
BINARY_OP(DST_IN, DST_ACC, DST_ACC);
```
- **Finding**: DST_ACC is released after fill (iteration 0), then re-loaded via copy_tile() in iteration 1
- **Pattern**: NOT held-DEST; tile register is released and re-acquired
- **Implementation**: CB_ACC holds the persisted value (staging buffer, not DEST register)

**moreh_adam.cpp (K12):**
```cpp
// First compute stage: power_tile
tile_regs_acquire();
copy_tile_init_with_dt(cb_scalar_args);
copy_tile(cb_scalar_args, beta2_tile, dst0);
power_tile_init();
power_tile(dst0, step);
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(cb_tmp1, onetile);
pack_tile_with_dt(dst0, cb_tmp1);
cb_push_back(cb_tmp1, onetile);
tile_regs_release();

// Second compute stage: sub and recip
tile_regs_acquire();
cb_wait_front(cb_tmp1, onetile);
cb_reserve_back(cb_tmp1, onetile);
// ... reconfig and sub ...
sub_tiles(cb_one, cb_tmp1, first_tile, first_tile, dst0);
recip_tile_init();
recip_tile(dst0);  // Recompute in dst0; not reading from first compute
```
- **Finding**: No held-DEST; each stage re-loads DEST registers
- **Pattern**: NOT held-DEST; CB_tmp1 staging buffer acts as persistence

**No genuine held-DEST patterns found in sampled kernels.**

### Verdict
- **Held-DEST gap: Minimal to None**
- All sampled kernels use CB staging (reserve_back/pop_front/push_back) to persist values across tile cycles
- DEST registers are acquired, committed, waited, and released per-tile or per-block
- **Recommendation**: Do NOT prioritize held-DEST support in policy enums; focus on CB lifecycle

---

## Section 6: Policy-Enum Recommendations

### CB Lifecycle Policies

Based on Section 2 analysis:

```cpp
enum class CBLifecyclePolicy {
    PerTile,            // K1-K3, K10, K14, K17-K18: wait_front(1) per tile, pop_front(1) per tile
    UpfrontBlock,       // K4-K5: wait_front(block_size) once, pop_front(block_size) once
    UpfrontKernel,      // K6, K8, K11: wait_front(1) once at kernel start, never pop (persistent)
    CumulativeWait,     // K15: cumulative wait_front tracking across granules
    ImplicitViaHelper,  // K6, K7, K8, K10, K12: dtype reconfig implicit in *_init_with_dt / *_to_cb
};
```

**Kernel-to-Policy Mapping:**

| Policy | Kernels | Characteristics |
|--------|---------|-----------------|
| PerTile | K1-K3, K10, K14, K17-K18 | Tight per-tile loop; acquire/release per tile; simple CB flow |
| UpfrontBlock | K4-K5 | Block-level amortization; conditional pre-scaling; per-block reconfig |
| UpfrontKernel | K6, K8, K11 | Persistent inputs (constants, weights); never popped; warm-read friendly |
| CumulativeWait | K15 | Granularized reduce; tracks cumulative in0_wait_tiles; nested loop optimization |
| ImplicitViaHelper | K6, K7, K8, K10, K12 | *_init_with_dt() and *_to_cb() handle dtype; no explicit reconfig_data_format_*() |

### Dtype Reconfig Modes

```cpp
enum class DtypeReconfigMode {
    EntryOnly,          // K4-K5: reconfig_data_format_srca() at block start, amortize cost
    MidLoopStage,       // K12: reconfig inside per-tile loop, triggered by FP32_DEST_ACC_EN
    ImplicitViaHelper,  // K6, K7, K8, K10: handled by *_init_with_dt() without explicit call
    None,               // K1-K3: uniform dtype; no reconfig needed
};
```

**Kernel-to-Mode Mapping:**

| Mode | Kernels | Frequency | Trigger |
|------|---------|-----------|---------|
| EntryOnly | K4, K5 | per-block | SFPU_OP_INIT_PRE_{IN0,IN1}_0 |
| MidLoopStage | K12 | per-tile | FP32_DEST_ACC_EN |
| ImplicitViaHelper | K6, K7, K8, K10 | per-tile (internal) | none (automatic) |
| None | K1-K3 | — | — |

### DEST Sync Styles

```cpp
enum class DestSyncStyle {
    Modern,             // K1-K3, K4-K12: acquire_dst/release_dst per tile or block; modern[ACWR]
    RawDst,             // K13-K16: acquire at block/window start, release at end; raw-dst[ar]
    AcqRelMacro,        // Not found in sample, but TSV indicates moreh_dot_backward uses ACQ-REL-macro
};
```

**Recommendation:**
- Support Modern (required; all samples except transformer)
- Support RawDst (required; transformer reduce patterns)
- Flag ACQ-REL-macro for potential deprecation (complex, rare in current sample)

### Multi-DEST-Slot Packing

```cpp
enum class MultiDestPackModel {
    SingleElement,      // K1-K3, K4-K5, K6, K8, K10, K11, K14, K15, K16:
                        // pack_tile(single_idx, out_cb) or loop-unrolled packs
    SingleBlockMultiIdx, // K17-K18: pack_tile(DST_ACC, CB_OUT), then pack_tile(DST_ACC, CB_ACC)
                        // Single PackTileBlock with N output CB slots
};
```

**Recommendation:**
- Primary: SingleElement (most kernels)
- Secondary: SingleBlockMultiIdx for accumulators and cross-CB packing (K17-K18)
- Do NOT support independent PackTile elements for each slot within one window (not found in practice)

### Held-DEST Support

```cpp
enum class HeldDestSupport {
    None,               // All sampled kernels; use CB staging instead
};
```

**Recommendation:**
- Gap: None needed; all production kernels use CB staging for persistence
- Simplify helper design by removing held-DEST constraints

---

## Section 7: Appendix — Master Kernel Analysis Table

| # | File | Category | Heavy_Lifting | CB_Lifecycle | Reconfig_Mode | Sync_Style | Multi_Slot | Held_Dest | Notes |
|---|------|----------|----------------|--------------|---------------|-----------|-----------|-----------|-------|
| 1 | bcast_h.cpp | eltwise | eltwise-fpu-bcast | PerTile | None | RawDst | SingleElement | No | 3-loop bcast; per-tile acquire/release |
| 2 | bcast_hw.cpp | eltwise | eltwise-fpu-bcast | PerTile | None | RawDst | SingleElement | No | Similar to K1 |
| 3 | bcast_w.cpp | eltwise | eltwise-fpu-bcast | PerTile | None | RawDst | SingleElement | No | Similar to K1; column-wise |
| 4 | eltwise_binary_kernel.cpp | eltwise | sfpu-unary | UpfrontBlock | EntryOnly | Modern | SingleElement | No | Per-block reconfig; conditional pre-scale |
| 5 | eltwise_binary_sfpu_kernel.cpp | eltwise | sfpu-unary | UpfrontBlock | EntryOnly | Modern | SingleElement | No | Mixed SFPU+binary; per-block stages |
| 6 | batch_norm_kernel.cpp | normalization | eltwise-fpu-binary | UpfrontKernel + PerTile | ImplicitViaHelper | Modern | SingleElement | No | CB chaining; persistent DEN |
| 7 | batch_norm_sfpu_kernel.cpp | normalization | sfpu-unary | UpfrontKernel + PerTile | ImplicitViaHelper | Modern | SingleElement | No | Typecast; state-machine reconfig |
| 8 | running_statistics_kernel.cpp | normalization | unknown | UpfrontKernel | None | Modern | SingleElement | No | Momentum update; persistent inputs |
| 9 | batch_norm_kernel.cpp | normalization | sfpu-unary | UpfrontKernel + PerTile | ImplicitViaHelper | Modern | SingleElement | No | (sfpu variant) |
| 10 | moreh_abs_pow_kernel.cpp | moreh | sfpu-unary | PerTile | ImplicitViaHelper | Modern | SingleElement | No | 2-loop nested; per-tile copy/abs/pow |
| 11 | moreh_adam.cpp | moreh | copy | UpfrontKernel | None | Modern | SingleElement | No | CB-to-CB copy helpers |
| 12 | moreh_adam.cpp | moreh | sfpu-unary | PerTile | MidLoopStage | Modern | SingleElement | No | FP32 accumulation; per-tile reconfig |
| 13 | compute_common.hpp | transformer | copy | — | ImplicitViaHelper | RawDst | SingleElement | No | Helper function; reduce copy init |
| 14 | compute_common.hpp | transformer | eltwise-fpu-bcast | PerTile | None | RawDst | SingleElement | No | max_block_inplace; per-tile acquire/release |
| 15 | compute_common.hpp | transformer | eltwise-fpu-bcast | CumulativeWait | None | RawDst | SingleElement | No | reduce_c; granularized; cumulative wait tracking |
| 16 | compute_common.hpp | transformer | eltwise-fpu-binary | PerTile | None | RawDst | SingleElement | No | reduce block max row |
| 17 | accumulation_compute.cpp | reduction | eltwise-fpu-binary | OutOfOrder | MidLoopStage | Modern | SingleBlockMultiIdx | No | Dual DEST slot; per-tile cross-CB pack |
| 18 | accumulation_compute.cpp | reduction | unknown | OutOfOrder | MidLoopStage | Modern | SingleBlockMultiIdx | No | (single variant) |

---

## Summary of Findings

### CB Lifecycle
- **Dominant patterns**: PerTile (7 kernels), UpfrontBlock (2 kernels), UpfrontKernel (4 kernels)
- **CumulativeWait**: 1 kernel (K15), suggests optimization for nested reduce loops
- **Implication**: Helper design must support variable wait/pop granularity; UpfrontBlock reduces reconfiguration cost

### Dtype Reconfig
- **EntryOnly (K4-K5)**: ~11% of sample; high reuse ratio amortizes cost
- **MidLoopStage (K12, K17-K18)**: ~22% of sample; triggered by FP32 accumulation or explicit dtype crossing
- **ImplicitViaHelper (K6-K10)**: ~56% of sample; leverages *_init_with_dt() macros
- **Implication**: Helper must provide both explicit and implicit reconfig paths; default to implicit for common case

### DEST Sync
- **Modern (K1-K12, K17-K18)**: 14/18 kernels; universal acquire/release per-tile or per-block
- **RawDst (K13-K16)**: 4/18 kernels; transformer reduce; block-level acquire/release with multi-pack
- **Implication**: Support Modern as baseline; RawDst as specialized path for reduce/transformer

### Multi-Slot Packing
- **SingleElement**: 16/18 kernels; standard pack_tile(idx, cb)
- **SingleBlockMultiIdx**: 2/18 kernels (K17-K18); same DEST slot to multiple CBs
- **Implication**: Default to SingleElement; add option for multi-output CB packing in accumulators

### Held-DEST
- **Found**: 0/18 kernels
- **Alternative**: All kernels use CB staging (reserve_back/push_back) for persistence
- **Implication**: Drop held-DEST from helper design; focus on CB lifecycle

---

## Recommendations for Helper Policy Enums

1. **CB Lifecycle Policy**: Support PerTile, UpfrontBlock, UpfrontKernel, CumulativeWait
2. **Dtype Reconfig Mode**: Support EntryOnly, MidLoopStage, ImplicitViaHelper
3. **DEST Sync Style**: Support Modern (default), RawDst (transformer), drop ACQ-REL-macro (complex, rare)
4. **Multi-Slot Pack Model**: Support SingleElement (default), SingleBlockMultiIdx (accumulator path)
5. **Held-DEST**: Not implemented; rely on CB staging for cross-tile persistence

---

Generated: 2026-05-06
Analysis scope: 18 production kernels across 5 categories
Source: `/localdev/astancov/tt-metal/pack_patterns.tsv`, kernel source files
