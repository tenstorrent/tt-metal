# Eltwise-chain migration audit — groupnorm compute kernels

Read-only audit. For each remaining **raw LLK eltwise** stage (not already on
`eltwise_chain`/convenience): name, file:line-range, current LLK op, verdict, and
proposed chain shape if migratable.

OUT-OF-SCOPE families present in these kernels and intentionally skipped:
`compute_kernel_lib::reduce<>` (reduce helper), `compute_kernel_lib::tilize<>` /
`untilize<>` (tilize helpers), `transpose_wh_*` (transpose), `welford_*` (welford LLK).
Stages already migrated to the chain/convenience wrappers are not re-listed.

---

## 1. groupnorm.cpp
`ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp`

### 1.1 Mask input (Average Calc / Local Reduce)  — lines 315–347
- LLK: `mul_tiles(cb_in0|cb_in, cb_input_mask, index, index_mask, w)` per subblock → `pack_tile(.., cb_x)`.
- Verdict: **MIGRATABLE**
- Shape: `mul<cb_inX, cb_input_mask, cb_x, BroadcastDim::None, ...>` over
  `EltwiseShape::grid(out_block_h_actual, block_w)`.
  - A operand `cb_in0`(or `cb_in` under TILIZE_IN): `OperandKind::Block` (absolute walk
    `ht*block_w+wt` = original `w+index_subblock_w_offset+index_h_offset`), lifecycle
    `InputLifecycle::Bulk` (caller `wait_front(out_block_hw_normal)` upfront, pop at end —
    matches `cb_in0.wait_front`/`cb_in0.pop_front(out_block_hw_normal)` at 312/345).
  - B operand `cb_input_mask`: `OperandKind::Row` (idx=wt=`w+index_subblock_w_offset`),
    `InputLifecycle::HeldBulk` (mask waited at 302, popped at 827 outside this stage).
  - Output `cb_x`: `OutputLifecycle::Bulk` (reserve(out_block_hw_normal)@318, push@347).
  - Reconfig: explicit `reconfig_data_format_srcb(cb_in0,cb_input_mask)`@315 +
    `mul_tiles_init` → `BinaryDataFormatReconfig::Input`; plain `pack_tile` →
    `PackTileReconfig::None`.
  - Note: the extra-out-block padding (actual<normal) is handled by the original via the
    upfront `_normal` reserve/wait; Bulk sizing must use `out_block_hw_normal`. Block walk
    over `out_block_h_actual` rows handles the short tail naturally.

### 1.2 Re-mask residual (Variance Calc / Local Reduce) — lines 425–451
- LLK: `mul_tiles(cb_xmm, cb_input_mask, index, index_mask, w)` → `pack_tile(.., cb_x)`.
- Verdict: **MIGRATABLE**
- Shape: `mul<cb_xmm, cb_input_mask, cb_x, BroadcastDim::None, ...>` over
  `grid(out_block_h_actual, block_w)`.
  - A `cb_xmm`: `OperandKind::Scalar` front-read with `InputLifecycle::Streaming` — original
    pops `block_w` per row (`cb_xmm.pop_front(block_w)`@446) i.e. front-relative rotation;
    per-tile streaming pop over the grid is the net-identical drain. (Same idiom already
    migrated in groupnorm_sharded_v2.cpp x-E[x].)
  - B `cb_input_mask`: `OperandKind::Row`, `InputLifecycle::HeldBulk`.
  - Output `cb_x`: `OutputLifecycle::Bulk` (reserve@427, push@451).
  - Extra-out-block slack pop (`cb_xmm.pop_front(normal-last)`@449) stays as caller cleanup,
    or fold via a Bulk A on `out_block_hw_normal`; simplest faithful mapping = Streaming A +
    keep the slack pop.
  - Reconfig: `reconfig_data_format_srcb`@425 + `mul_tiles_init` → `Input`; plain pack → `None`.

### 1.3 (x − E[x])² (Variance Calc) — lines 456–478
- LLK: `mul_tiles(cb_x, cb_x, index, index, w)` same-CB → `pack_tile(.., cb_xmm)`.
- Verdict: **MIGRATABLE**
- Shape: `square<cb_x, cb_xmm, InputLifecycle::Bulk, OutputLifecycle::Bulk, BinaryDataFormatReconfig::Input, PackTileReconfig::None, OperandKind::Block>`
  over `grid(out_block_h_actual, block_w)`.
  - cb_x: Block absolute walk (`w+index_subblock_w_offset+index_h_offset`), Bulk
    (wait@458, pop@477 of `out_block_hw_normal`).
  - cb_xmm: Bulk (reserve@457, push@478).
  - Reconfig: `mul_tiles_init(cb_x,cb_x)` (srca/srcb) preceded by
    `reconfig_data_format_srcb(cb_input_mask,cb_x)`@453 → `Input`; plain pack → `None`.

### 1.4 Re-mask residual (Final Val Calc) — lines 595–621
- LLK: identical to 1.2 (`mul_tiles(cb_xmm, cb_input_mask)` → cb_x).
- Verdict: **MIGRATABLE** — same shape as 1.2.

### 1.5 (x − Ex) · 1/√(Var+eps) (Final Val Calc) — lines 626–649
- LLK: `mul_tiles_bcast_scalar(cb_x, cb_ex2pe, index, 0, w)` → `pack_tile(.., cb_xmm)`.
- Verdict: **MIGRATABLE**
- Shape: `mul<cb_x, cb_ex2pe, cb_xmm, BroadcastDim::Scalar, ...>` over
  `grid(out_block_h_actual, block_w)`.
  - cb_x: `OperandKind::Block` (abs walk `w+index_subblock_w_offset+index_h_offset`),
    `InputLifecycle::Bulk` (wait@629, pop@648).
  - cb_ex2pe: `OperandKind::Scalar`, `InputLifecycle::CallerManaged` (held by external
    `wait_front(1)`@628 / `pop_front(1)`@826) — mirrors the already-migrated x−E[x] chain.
  - Output cb_xmm: `OutputLifecycle::Bulk` (reserve@627, push@649). NOTE the trailing
    `cb_xmm.wait_front(out_block_hw_normal)`@650 stays as the consumer barrier for stage 1.6.
  - Reconfig: `mul_tiles_bcast_scalar_init_short`@626 → `Input`; plain pack → `None`.

### 1.6 Add-or-copy with previous output — lines 659–714
- LLK: runtime `copy_or_add` bool selects `copy_tile(cb_xmm)` vs `add_tiles(cb_reread_out, cb_xmm)`
  per `w`-column → `pack_tile<true>(.., cb_reread_write_out, index)`.
- Verdict: **BLOCKED: runtime per-tile OP selection** — `copy_or_add` is a runtime variable
  (toggled by the per-column group-reset state machine, 691–706); the op (copy vs FPU add)
  is chosen inside the column loop. The chain bakes the op into the element type at compile
  time, so this cannot be expressed as one chain. (Also writes with a runtime
  destination tile index via `pack_tile<true>(.., index)`.)

### 1.7 Optional Gamma — lines 717–747
- LLK: runtime `apply_gamma_beta[j]` selects `mul_tiles_bcast_rows(cb_reread_write_out, cb_gamma)`
  vs `copy_tile(cb_reread_write_out)` per tile → `pack_tile(.., cb_outgamma)`.
- Verdict: **BLOCKED: runtime per-tile OP selection** — `apply_gamma_beta[j]` is a runtime
  bool array (computed at 708–710); FPU-mul-bcast vs copy is picked per inner tile. Not
  expressible as a single fixed-op chain.

### 1.8 Optional Beta — lines 751–781
- LLK: runtime `apply_gamma_beta[j]` selects `add_tiles_bcast_rows(cb_inbeta, cb_beta)` vs
  `copy_tile(cb_inbeta)` per tile → `pack_tile(.., cb_outbeta)`.
- Verdict: **BLOCKED: runtime per-tile OP selection** — same structure as 1.7.

---

## 2. groupnorm_sharded_v2.cpp
`ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp`

### 2.1 Mask input — lines 211–245
- LLK: `mul_tiles(cb_in0|cb_in, cb_input_mask, index, index_mask, w)` → `pack_tile(.., cb_x)`,
  with runtime in-bounds index clamp (`if (index >= per_core_MN) index = per_core_MN-1;`).
- Verdict: **MIGRATABLE**
- Shape: `mul<cb_inX, cb_input_mask, cb_x, BroadcastDim::None, ...>` over
  `grid(block_h, block_w)`.
  - A `cb_in0`/`cb_in`: `OperandKind::Block` with `TileOffset::Set` carrying
    `index_h_offset = index_b_offset + index_g_offset` (runtime base), lifecycle
    `InputLifecycle::CallerManaged` (cb_in is held by the outer `cb_in.wait_front(per_core_MN)`
    @200/`pop_front`@594; cb_in0 sharded so no per-stage pop). Block+Set+CallerManaged is a
    legal combo (chain.hpp `is_legal_input_lifecycle_with_base`).
  - B `cb_input_mask`: `OperandKind::Row`, `InputLifecycle::HeldBulk` (waited@215, no pop here).
  - Output cb_x: `OutputLifecycle::Bulk` (reserve@214, push@245).
  - Reconfig: `reconfig_data_format_srcb`@212 + `mul_tiles_init` → `Input`; plain pack → `None`.
  - CAVEAT (verify before applying): the original clamps any `index >= per_core_MN` to the
    last tile (relying on the mask zeroing it). A plain Block walk computes `base + ht*Wt+wt`
    with no clamp. For the last short group this could read out of the CB window. The chain
    has no per-index clamp hook, so a faithful migration must guarantee `base + block_h*block_w
    <= per_core_MN` for every group, OR this stage is **BLOCKED: per-tile runtime index clamp**.
    Classify conservatively as MIGRATABLE *only if* the host guarantees no clamp ever fires;
    otherwise BLOCKED on the clamp. (sharded_v1 groupnorm.cpp has no clamp, so its 1.1 is clean.)

### 2.2 Partial-E[x] accumulate — lines 249–270
- LLK: `mul_tiles(cb_x, cb_ones, index, 0, dst0)` looped over block_h·block_w, **all writing
  into the single DEST slot dst0** (FPU multiply-by-ones accumulation), one `pack_tile(dst0, cb_ex2pe)`.
- Verdict: **OUT-OF-SCOPE: reduce** — this is a hand-rolled SUM reduction into one DEST tile
  (×1 accumulate), not a per-tile eltwise map. The chain emits one packed output per input
  tile; it cannot accumulate N input tiles into a single DEST slot. (The kernel comment at
  256–257 confirms it is an alternative to `reduce_tile`.) Belongs to the reduce helper family.

### 2.3 (x − E[x])² accumulate — lines 354–376
- LLK: `mul_tiles(cb_x, cb_x, index, index, dst0)` looped over block_h·block_w **into single
  DEST slot dst0**, one `pack_tile(dst0, cb_ex2pe)`.
- Verdict: **OUT-OF-SCOPE: reduce** — same as 2.2: squared-residual sum reduced into one
  DEST tile. Not a per-tile eltwise map; in-DEST cross-iteration accumulation a `square`/`mul`
  chain cannot express (DestReuseBinary accumulates DEST↔src per tile, not N→1 over the walk).

### 2.4 Add-or-copy with previous output (non-negative-mask) — lines 456–502
- LLK: runtime `copy_or_add` selects `copy_tile(cb_x)` vs `add_tiles(cb_out, cb_x)` per
  `w`-column → `pack_tile<true>(.., cb_out, index)`.
- Verdict: **BLOCKED: runtime per-tile OP selection** — identical pattern to groupnorm.cpp 1.6;
  `copy_or_add` runtime-toggled per column, plus runtime destination index in `pack_tile<true>`.

### 2.5 Negative-mask zero-out (FUSE_NEGATIVE_MASK) — lines 509–527
- LLK: `mul_tiles(cb_in, cb_in_negative_mask, index_in, index_mask, dst0)` → in-place
  `pack_tile<true>(dst0, cb_in, index_in)`.
- Verdict: **MIGRATABLE**
- Shape: `mul<cb_in, cb_in_negative_mask, cb_in, BroadcastDim::None, ...>` over
  `grid(block_h, block_w_curr)`.
  - A `cb_in`: `OperandKind::Block` + `TileOffset::Set`(base `index_b_offset+index_g_offset`),
    `InputLifecycle::CallerManaged` (cb_in held by outer wait). Output back into cb_in via
    `OutputLifecycle::CallerManaged`/`HeldReserve` with `TileOffset::Set` (same base) — the
    original does in-place `pack_tile<true>(.., index_in)` with no reserve/push here.
  - B `cb_in_negative_mask`: `OperandKind::Row`, `InputLifecycle::HeldBulk` (waited@505,
    popped@553 outside). idx = `w`.
  - Reconfig: `reconfig_data_format_srcb(cb_x,cb_in_negative_mask)`@506 + `mul_tiles_init` →
    `Input`. NOTE: the original walks columns-outer / rows-inner with stride `per_core_N`
    (column-major within the per_core grid); the chain's Block walk is row-major. The *set of
    tiles* touched is the same and each is an independent in-place map, so order does not
    affect correctness — but the in-place absolute-index write needs TileOffset::Set on the
    PackTile to hit `index_in`, and output-CB sizing must cover `base + grid`. Confirm the
    in-place output mapping (HeldReserve / CallerManaged + Set) before applying.

### 2.6 Negative-mask add-back — lines 534–552
- LLK: `add_tiles(cb_in, cb_x, index, index_x, dst0)` → in-place `pack_tile<true>(dst0, cb_in, index)`.
- Verdict: **MIGRATABLE**
- Shape: `add<cb_in, cb_x, cb_in, BroadcastDim::None, ...>` over `grid(block_h, block_w_curr)`.
  - A `cb_in`: `OperandKind::Block` + `TileOffset::Set`(base `index_b_offset+index_g_offset`),
    `InputLifecycle::CallerManaged`.
  - B `cb_x`: `OperandKind::Block` (abs idx `w+index_h1_offset`, stride block_w),
    `InputLifecycle::CallerManaged` (cb_x popped once at 556). Two distinct Block walks with
    different strides — the chain derives both from the same (Ht,Wt) grid, so the strides must
    coincide; here A strides `per_core_N` and B strides `block_w`. These differ, so a single
    grid cannot drive both Block indices. Re-map: cb_x as Block over its own (block_h,block_w)
    grid, cb_in via TileOffset::Set with per-row stride per_core_N — NOT expressible as one
    EltwiseShape (chain uses one grid for all Block operands).
  - REVISED Verdict: **BLOCKED: two Block operands with mismatched row strides**
    (cb_in stride per_core_N vs cb_x stride block_w). The chain's Block index derives every
    Block operand from the single (Ht,Wt) grid; it cannot give two operands different row
    pitches. (TileOffset only adds a constant base, not a per-row stride.)
  - (Stage 2.5 above is single-operand-Block so it is unaffected; only the mask is Row.)

### 2.7 do_gamma / do_beta / final write — lines 601–724
- Already migrated to `compute_kernel_lib::mul`/`add`/`untilize`; the `do_beta` non-neg-mask
  branch (680–692) carries an in-source comment that a chain there *deadlocks* the
  UNTILIZE_OUT tile_layout pipeline, but it IS in fact expressed via `add<>` (it is migrated).
  Not a raw stage — excluded.

---

## 3. welford_groupnorm.cpp
`ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp`

### 3.1 Welford partial-tile updates — lines 254–322
- LLK: `transpose_wh_tile` + `welford_update_rows` / `welford_save/restore_state`.
- Verdict: **OUT-OF-SCOPE: welford** (and transpose). Not eltwise.

### 3.2 Statistics finalize — lines 325–335
- LLK: `welford_finalize_to_face` + `pack_tile_block`.
- Verdict: **OUT-OF-SCOPE: welford**.

### 3.3 (Var + eps) → 1/√ — lines 343–356
- LLK: `add_tiles(cb_ex_global, cb_eps)` + `rsqrt_tile<true>` → `pack_tile(.., cb_ex2pe)`,
  per group with strided read `1+(g<<1)`.
- Verdict: **MIGRATABLE**
- Shape: per-group `eltwise_chain(1, BinaryFpu<cb_ex_global, cb_eps, Add, None,
  InputLifecycle::HeldBulk, InputLifecycle::CallerManaged, BinaryDataFormatReconfig::Input,
  Dst::D0, OperandKind::Scalar, OperandKind::Scalar, TileOffset::Set>{1+(g<<1), 0},
  Rsqrt<Approx::Exact, Legacy::On, Dst::D0>{},
  PackTile<cb_ex2pe, OutputLifecycle::Bulk, PackTileReconfig::None>{})`.
  - This is **the exact shape already used in welford_groupnorm_sharded_v2.cpp lines 246–269**
    (TileOffset::Set on the strided cb_ex_global read; cb_ex2pe Bulk per-call replacing the
    upfront reserve@341 + push@357). cb_ex_global waited upfront (340) → HeldBulk; cb_eps held
    by outer wait (227) → CallerManaged.
  - Reconfig: `add_tiles_init` + `reconfig_data_format_srcb(cb_eps)`@343-344 → `Input`;
    plain pack → `None`; `rsqrt_tile_init<true>` → `Legacy::On`.

### 3.4 Final normalization a/b/c/d + gamma/beta/write — lines 391–552
- Sub-steps a (x−u), b (1/√·mask), c (a·b) are RAW; d (accumulate), gamma, beta, write are
  already migrated (`copy`/`add`/chain).
- **3.4a x − u** (396–404): `sub_tiles_bcast_scalar(cb_in0, cb_ex_global, 0, g<<1, dst0)`
  → `pack_tile(.., cb_xmm)`. Verdict: **MIGRATABLE**.
  - `sub<cb_in0, cb_ex_global, cb_xmm, BroadcastDim::Scalar, InputLifecycle::Streaming,
    InputLifecycle::HeldBulk, OutputLifecycle::HeldReserve, BinaryDataFormatReconfig::Input,
    PackTileReconfig::None, OperandKind::Scalar, OperandKind::Scalar + TileOffset::Set{g<<1}>(1u)`.
    cb_in0 front-read (idx 0, per-tile streaming — popped at 487 once per nt outside this g-loop;
    use CallerManaged for cb_in0 since pop happens after the g-loop). cb_ex_global strided read
    `0+(g<<1)` via TileOffset::Set, HeldBulk. cb_xmm reserved as part of the 2-tile
    `reserve_back(2)`@392 / `push_back(2)`@418 window → output here is the first of the pair
    (HeldReserve / CallerManaged, the push covers both a and b).
  - Reconfig: `sub_tiles_bcast_scalar_init_short` + `reconfig_data_format_srcb(cb_eps,
    cb_ex_global)`@396-397 → `Input`; plain pack → `None`.
- **3.4b 1/√·mask** (407–417): `mul_tiles_bcast_scalar(cb_input_mask, cb_ex2pe, mask_index, g, dst0)`
  → `pack_tile(.., cb_xmm)` (2nd of the pair). Verdict: **MIGRATABLE**.
  - `mul<cb_input_mask, cb_ex2pe, cb_xmm, BroadcastDim::Scalar, ..., OperandKind::Scalar +
    TileOffset::Set{mask_index} / TileOffset::Set{g}>`. cb_input_mask strided runtime read
    `mask_index` (HeldBulk + Set), cb_ex2pe strided read `g` (HeldBulk + Set). Output cb_xmm
    second tile of the held 2-tile window (CallerManaged — the `push_back(2)`@418 is shared).
  - Reconfig: `mul_tiles_bcast_scalar_init_short` + `reconfig_data_format_srcb(cb_ex_global,
    cb_ex2pe)`@410-411 → `Input`; plain pack → `None`.
- **3.4c a · b** (421–432): `mul_tiles(cb_xmm, cb_xmm, 0, 1, dst0)` → `pack_tile(.., cb_xmm)`.
  Verdict: **MIGRATABLE**.
  - `mul<cb_xmm, cb_xmm, cb_xmm, BroadcastDim::None, ...>` with the two operands reading
    tile 0 and tile 1 of cb_xmm (A: Scalar idx0; B: Scalar + TileOffset::Set{1}). Input
    `wait_front(2)`@421 / `pop_front(2)`@427 → CallerManaged (caller-bracketed). Output the
    1-tile result via the `reserve_back(1)`@428 / `push_back(1)`@432 window → HeldReserve /
    CallerManaged. Same-CB read+write needs the chain's same-buffer path; the explicit
    wait/pop/reserve/push bracket is caller-owned so CallerManaged on both ends is faithful.
  - Reconfig: `mul_tiles_init` + `reconfig_data_format_srcb(cb_ex2pe, cb_xmm)`@422-423 → `Input`.
  - NOTE: 3.4a/b/c share one `cb_xmm.reserve_back(2)`/`wait_front(2)`/`pop_front(2)` bracket
    spanning all three; migrating them individually requires routing those edges through
    CallerManaged/HeldReserve lifecycles exactly as annotated — fiddly but expressible. They
    could alternatively fuse into ONE chain (CopyTile×2 + sub/mul + SFPU) but that changes the
    intermediate cb_xmm round-trip the original performs; the faithful mapping keeps three calls.

---

## 4. welford_groupnorm_sharded_v2.cpp
`ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp`

### 4.1 Welford partial-tile updates — lines 176–222
- LLK: `transpose_wh_tile` + `welford_update_rows` / save/restore.
- Verdict: **OUT-OF-SCOPE: welford** (+ transpose).

### 4.2 Statistics finalize — lines 224–234
- LLK: `welford_finalize_to_face` + `pack_tile_block`.
- Verdict: **OUT-OF-SCOPE: welford**.

### 4.3 (Var + eps) → 1/√ — lines 246–269
- Already migrated (`eltwise_chain` BinaryFpu(Add)+Rsqrt+PackTile, TileOffset::Set). Excluded.

### 4.4 Final Val Calc a/b/c — lines 296–341 (d/gamma/beta/write already migrated)
- **4.4a x − u** (301–313): `sub_tiles_bcast_scalar(cb_in0|cb_in, cb_ex_global, tile_id, g<<1, dst0)`
  → `pack_tile(.., cb_xmm)`. Verdict: **MIGRATABLE**.
  - Same shape as welford_groupnorm.cpp 3.4a, but the cb_in read uses a **runtime `tile_id`**
    (sharded absolute tile index, incremented per nt) → `OperandKind::Scalar` +
    `TileOffset::Set{tile_id}`, `InputLifecycle::CallerManaged` (cb_in sharded, no per-stage
    pop). cb_ex_global Scalar + TileOffset::Set{g<<1}, HeldBulk. Output cb_xmm first of the
    held 2-tile window (reserve@297 / push@327).
  - Reconfig: `reconfig_data_format(cb_in0,cb_ex_global)` + `sub_tiles_bcast_scalar_init_short`
    @301-302 → `Input`; plain pack → `None`.
- **4.4b 1/√·mask** (316–326): `mul_tiles_bcast_scalar(cb_input_mask, cb_ex2pe, mask_index, g, dst0)`
  → cb_xmm. Verdict: **MIGRATABLE** — same shape as welford_groupnorm.cpp 3.4b
  (cb_input_mask Scalar+Set{mask_index} HeldBulk, cb_ex2pe Scalar+Set{g} HeldBulk, cb_xmm 2nd
  of held pair). Reconfig: `reconfig_data_format(cb_in0,cb_input_mask,cb_ex_global,cb_ex2pe)`
  4-arg@319 + `mul_tiles_bcast_scalar_init_short` → `Input`; pack → `None`.
- **4.4c a · b** (330–341): `mul_tiles(cb_xmm, cb_xmm, 0, 1, dst0)` → cb_xmm. Verdict:
  **MIGRATABLE** — same shape as welford_groupnorm.cpp 3.4c (A Scalar idx0, B Scalar+Set{1},
  CallerManaged in, HeldReserve/CallerManaged out over the shared 2→1 cb_xmm bracket).
  Reconfig: `reconfig_data_format(cb_input_mask,cb_xmm,cb_ex2pe,cb_xmm)`@331 +
  `mul_tiles_init` → `Input`; pack → `None`.
  - Same caveat as 3.4: a/b/c share one `cb_xmm.reserve_back(2)`/`wait_front(2)`/`pop_front(2)`
    + `reserve_back(1)`/`push_back(1)` bracket; faithful migration routes those edges via
    CallerManaged/HeldReserve.

---

## Summary counts

| Kernel | MIGRATABLE | BLOCKED | OUT-OF-SCOPE |
|---|---|---|---|
| groupnorm.cpp | 5 | 3 | 0 |
| groupnorm_sharded_v2.cpp | 2 | 3* | 2 |
| welford_groupnorm.cpp | 4 | 0 | 2 |
| welford_groupnorm_sharded_v2.cpp | 3 | 0 | 2 |

\* groupnorm_sharded_v2 stage 2.1 is MIGRATABLE only if the host guarantees the index clamp
never fires; otherwise it is BLOCKED on the per-tile runtime clamp. Counted as MIGRATABLE
above (conservative note inline).
