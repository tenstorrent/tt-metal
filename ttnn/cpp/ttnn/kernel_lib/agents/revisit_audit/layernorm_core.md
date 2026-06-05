# Revisit audit — layernorm core compute kernels (eltwise_chain missed-migration sweep)

Scope: find every remaining **raw LLK elementwise stage** (manual cb wait/pop/reserve/push +
FPU/SFPU/copy/bcast LLK calls inside an explicit `tile_regs_*` window) that the
`compute_kernel_lib::eltwise_chain` family could own, and classify it.

Capability recap (verified against eltwise_chain.hpp / eltwise_convenience.hpp / eltwise_math.hpp):
- Elements: CopyTile, BinaryFpu(Add/Sub/Mul), DestReuseBinary, UnaryBcast, PackTile; SFPU op
  structs (Exp/Rsqrt/Sqrt/…) run DEST-only between CopyTile and PackTile.
- OperandKind Block/Row/Col/Scalar; BroadcastDim None/Col/Row/Scalar; TileOffset::Set base.
- Lifecycles absorb cb wait/pop/reserve/push incl. `CircularBuffer` wrapper methods.
- NOT blockers: tile_regs_*, subblock/DEST-batch loops, bulk-vs-streaming, broadcasts,
  bulk-wait + per-tile-drain.
- Legitimate blockers: runtime per-tile OP selection; runtime CB id; not-eltwise
  (matmul/reduce/tilize/untilize/transpose/welford LLK → OUT-OF-SCOPE); in-DEST cross-iter
  accumulation DestReuseBinary can't express; **raw host-injected SFPU activation macro**
  (`SFPU_OP_INIT_ACTIVATION` / `SFPU_OP_FUNC_ACTIVATION`) interleaved between FPU op and pack —
  the chain runs only typed SFPU op structs, not arbitrary text-substituted activation
  passthrough (see MEMORY: eltwise_chain_migration_gotchas "no raw-SFPU-macro passthrough").

---

## 1. layernorm.cpp

Already migrated: FUSE_PRE_ADD add (167), x-E[x] sub (223), (x-E[x])^2 square (267),
Var+eps→rsqrt chain (290). Remaining raw stages are in the gamma/beta scale loop (305-401).

### Stage L.1 — x·rstd (mul bcast cols)  — `layernorm.cpp:318-335`
- LLK: `mul_bcast_cols_init_short(cb_xmm, cb_ex2pe)` + `mul_tiles_bcast_cols` + `pack_tile(cb_im_or_out)`;
  manual `cb_im_or_out_obj.reserve_back`/`push_back`, `reconfig_data_format(cb_xmm,cb_ex2pe)`,
  conditional `pack_reconfig_data_format(cb_out|cb_fusion)`, ACQ/REL.
- **Verdict: BLOCKED: raw SFPU activation macro.** The `SFPU_OP_INIT_ACTIVATION` /
  `SFPU_OP_FUNC_ACTIVATION` block (322-330) is injected between the FPU mul and the
  `pack_tile`, applied only in the no-gamma/no-beta case. That activation is an
  arbitrary host-substituted SFPU op the chain cannot express. When `SFPU_OP_INIT_ACTIVATION`
  is undefined this stage would be a clean `mul<cb_xmm,cb_ex2pe,cb_im_or_out, BroadcastDim::Col,
  HeldBulk(xmm)/CallerManaged(ex2pe), Bulk, …, Block, Scalar>` per block — but the macro is a
  compile-time variant of the SAME kernel, so the migration must hold for all variants. Blocked.

### Stage L.2 — ·gamma (mul bcast rows)  — `layernorm.cpp:343-374`
- LLK: `mul_bcast_rows_init_short(cb_fusion, cb_gamma)` + `mul_tiles_bcast_rows` + pack to
  `cb_outg`; manual wait/pop/reserve/push, `reconfig_data_format_srcb`, `pack_reconfig`.
- **Verdict: BLOCKED: raw SFPU activation macro.** Same `SFPU_OP_*` block (358-366) interleaved
  before pack (applied when do_beta==0). Otherwise this is `mul<cb_fusion,cb_gamma,cb_outg,
  BroadcastDim::Row, Bulk/HeldBulk+TileOffset, Bulk, …, Block, Block>`. Blocked.

### Stage L.3 — +beta (add bcast rows)  — `layernorm.cpp:375-400`
- LLK: `add_bcast_rows_init_short(cb_fusion, cb_beta)` + `add_tiles_bcast_rows` + pack to cb_out.
- **Verdict: BLOCKED: raw SFPU activation macro.** `SFPU_OP_*` block (390-393) unconditionally
  interleaved before pack. Otherwise `add<cb_fusion,cb_beta,cb_out, BroadcastDim::Row, …>`. Blocked.

layernorm.cpp counts: MIGRATABLE 0 | BLOCKED 3 (all raw-activation-macro) | OUT-OF-SCOPE 0.
(Reduces row_wise_mean — kernel_util reduce helper, already a helper; not a raw eltwise stage.)

---

## 2. layernorm_large_tensor.cpp

Already migrated: Var+eps→rsqrt chain (229), unary_bcast COL on cb_ex2pe (256).
This variant has NO chains in its two big blocked loops. Many raw stages.

### Pass-1 variance loop (126-211)

### Stage LT.1 — RMSNORM copy / x-E[x] sub  — `layernorm_large_tensor.cpp:133-147`
- LLK: RMSNORM path `copy_tile_init(cb_in)` + `copy_tile`; non-RMS path
  `sub_bcast_cols_init_short(cb_in,cb_ex)` + `sub_tiles_bcast_cols`; manual cb_in wait/pop, reconfig.
- **Verdict: MIGRATABLE.**
  - RMS: `copy<cb_in, <dest>, InputLifecycle::Bulk, …, OperandKind::Block>` — but note this copy
    feeds the SAME DEST that the square below consumes (it's the first half of a fused
    copy→(dest-reuse-add)→square→reduce DEST sequence). See LT.4 note: the whole 133-211 body is
    one DEST window. As an isolated x-E[x] producer-into-cb it is `sub<cb_in,cb_ex,?, Col, Bulk,
    CallerManaged, …, Block, Scalar>`; but here it does NOT pack to a CB — it leaves the result in
    DEST for the in-place square. So the eltwise result is consumed DEST-internally.
  - **Reclassify: BLOCKED: in-DEST cross-stage fusion.** The sub/copy result stays in DEST and is
    squared in place (159-162) then reduced (185-190) within one `tile_regs_acquire`…
    `tile_regs_commit` window — no CB pack between. The chain always packs (or dest-reuses to a
    CB); it cannot hand the live DEST tile to a following `square_tile`+`reduce_tile` in the same
    DEST window. Blocked.

### Stage LT.2 — FUSE_PRE_ADD dest-reuse add  — `layernorm_large_tensor.cpp:148-157`
- LLK: `binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCB>(cb_inb)` + `binary_dest_reuse_tiles`.
- **Verdict: BLOCKED: in-DEST cross-stage fusion.** Adds cb_inb into the live DEST tile (output of
  LT.1) with no pack; result continues into the in-place square. DestReuseBinary as a chain element
  must pack/own its CB output; here DEST is consumed by the next non-chain LLK (square+reduce) in
  the same window. Blocked.

### Stage LT.3 — (x-E[x])^2 square + pack  — `layernorm_large_tensor.cpp:159-171`
- LLK: `square_tile_init()` + `square_tile(i)` (DEST in place) then `pack_tile(cb_xmm2)`.
- **Verdict: BLOCKED: in-DEST cross-stage fusion.** `square_tile` operates on the DEST tile
  produced by LT.1/LT.2 (no CopyTile load) — it is the tail of the fused DEST sequence. The chain's
  `square` reads its operand from a CB (BinaryFpu Mul of CbIn×CbIn); it cannot square a tile that
  is already live in DEST from a prior non-chain stage. Blocked.

### Stage LT.4 — accumulate copy + REDUCE_ROW + scalar mul-unary  — `layernorm_large_tensor.cpp:173-210`
- LLK: `copy_tile(cb_accumulate)` (carry prev partial), `reduce_init/reduce_tile<SUM,REDUCE_ROW>`,
  `mul_unary_tile(dst0, 1/W)`, pack to cb_accumulate|cb_ex2.
- **Verdict: OUT-OF-SCOPE: reduce.** Core op is `reduce_tile` (REDUCE_ROW); the surrounding
  copy/mul_unary are in-DEST glue around the reduce within one DEST window. Not eltwise.

### Pass-2 final-value loop (270-424)

### Stage LT.5 — RMSNORM copy / x-E[x] sub  — `layernorm_large_tensor.cpp:280-295`
- LLK: same as LT.1 (copy / sub_bcast_cols) but here it PACKS to cb_xmm at 313-318.
- **Verdict: BLOCKED: in-DEST cross-stage fusion** *unless FUSE_PRE_ADD off*. Non-FUSE path:
  sub→(commit)→pack cb_xmm is a clean `sub<cb_in,cb_ex,cb_xmm, Col, Bulk, CallerManaged, Bulk,
  Input, Output, Block, Scalar>`. But the FUSE_PRE_ADD path (296-304) inserts a dest-reuse add
  into the live DEST before the pack, so the sub result cannot pack directly. Since FUSE_PRE_ADD is
  a compile-time variant of the same kernel, the migration must cover it → blocked by LT.6 fusion.
  *(If the host op guaranteed non-fuse, LT.5 alone is MIGRATABLE as the sub above.)*

### Stage LT.6 — FUSE_PRE_ADD dest-reuse add  — `layernorm_large_tensor.cpp:296-305`
- LLK: `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(cb_inb)`.
- **Verdict: BLOCKED: in-DEST cross-stage fusion.** Adds into the live sub-result DEST tile, then
  the combined DEST is packed to cb_xmm (313). DestReuseBinary chain element would need the sub
  result already in DEST from a prior chain element in the SAME chain — but the producer is LT.5,
  a separate stage. Could only migrate if LT.5+LT.6 were fused into one chain (CopyTile/sub →
  DestReuseBinary(add) → PackTile). That IS expressible as a single chain (sub bcast result in DEST
  → DestReuseBinary add cb_inb → PackTile cb_xmm). **Reclassify LT.5+LT.6 jointly: MIGRATABLE**
  as one fused chain — but only because no SFPU activation sits between them and the pack here.
  See "joint" entry below.

### Stage LT.5+LT.6 (joint) — sub bcast cols → dest-reuse add → pack  — `layernorm_large_tensor.cpp:278-318`
- **Verdict: MIGRATABLE** (single fused chain).
  - Shape: `EltwiseShape::tiles(full_block_size, blk)`.
  - Elements:
    - non-RMS: `BinaryFpu<cb_in, cb_ex, Mul→Sub, BroadcastDim::Col, Bulk, CallerManaged, Input,
      D0, Block, Scalar>` (the x-E[x]) ;  RMS: `CopyTile<cb_in, D0, Bulk, Input, Block>`.
    - if FUSE_PRE_ADD: `DestReuseBinary<cb_inb, BinaryFpuOp::Add, DEST_TO_SRCB, Bulk,
      Input, D0,D0, Block>`.
    - `PackTile<cb_xmm, OutputLifecycle::Bulk, Output, D0>`.
  - Reconfig: matches the explicit `reconfig_data_format(cb_in,cb_ex)` + `reconfig_data_format_srca
    (cb_inb)` + `pack_reconfig_data_format(cb_xmm)` in the original.
  - This is the cleanest unlocked migration in this file.

### Stage LT.7 — xmm·rstd (mul, col-vec ex2pe)  — `layernorm_large_tensor.cpp:321-350`
- LLK: `mul_tiles_init(cb_xmm,cb_ex2pe)` + `mul_tiles(...,0,...)` (ex2pe is a single col tile,
  index 0) + pack to cb_fusion|cb_out. (ex2pe was COL-broadcast to a full tile by the earlier
  unary_bcast, so this is a plain elementwise mul against tile 0 held — effectively Scalar B.)
- **Verdict: BLOCKED: raw SFPU activation macro.** `SFPU_OP_*` block (328-336) interleaved between
  mul and commit (no-gamma/no-beta case). Otherwise `mul<cb_xmm,cb_ex2pe,cb_fusion, None,
  Bulk, CallerManaged(ex2pe tile0), Bulk, …, Block, Scalar>`. Blocked by the activation macro.

### Stage LT.8 — ·gamma (mul bcast rows)  — `layernorm_large_tensor.cpp:353-393`
- LLK: `mul_bcast_rows_init_short(cb_fusion,cb_gamma)` + `mul_tiles_bcast_rows`, double-pack
  (cb_out vs cb_fusion) chosen at compile time.
- **Verdict: BLOCKED: raw SFPU activation macro.** `SFPU_OP_*` (365-373) before commit (do_beta==0
  case). Otherwise `mul<cb_fusion,cb_gamma,...,Row,…,Block,Block>` with output CB compile-selected.
  Blocked.

### Stage LT.9 — +beta (add bcast rows)  — `layernorm_large_tensor.cpp:394-418`
- LLK: `add_bcast_rows_init_short(cb_fusion,cb_beta)` + `add_tiles_bcast_rows` + pack cb_out.
- **Verdict: BLOCKED: raw SFPU activation macro.** `SFPU_OP_*` (404-407) before commit. Otherwise
  `add<cb_fusion,cb_beta,cb_out, Row, …, Block, Block>`. Blocked.

layernorm_large_tensor.cpp counts: MIGRATABLE 1 (LT.5+LT.6 joint sub→dest-reuse-add→pack)
| BLOCKED 7 (LT.1,LT.2,LT.3 in-DEST fusion; LT.5,LT.6 individually-but-jointly-migratable;
LT.7,LT.8,LT.9 raw-activation-macro) | OUT-OF-SCOPE 1 (LT.4 reduce).
Net unique unlocked: 1 fused chain (LT.5+LT.6).

---

## 3. layernorm_welford.cpp

Already migrated (FULLY, this is the model): FUSE_PRE_ADD add (115), x-E[x] sub (208),
Var+eps→rsqrt chain (235), and ALL THREE gamma/beta scale stages as chains (268, 290, 310).
The author note at line 256 confirms "No SFPU_OP_INIT_ACTIVATION macros in welford variant ->
all three stages migrate cleanly into chains."

Remaining raw stages are exclusively the Welford / transpose machinery:

### Stage W.1 — Welford E[x]/Var[x] (transpose + welford LLK)  — `layernorm_welford.cpp:131-156`
- LLK: `transpose_wh_init_short` / `transpose_wh_tile`, `welford_init` / `welford_update` /
  `welford_update_rows` / `welford_finalize_to_row`.
- **Verdict: OUT-OF-SCOPE: welford+transpose.** Not eltwise.

### Stage W.2 — pack mean/var, transpose back, re-pack  — `layernorm_welford.cpp:158-193`
- LLK: `pack_tile(mean_dst,cb_ex)` / `pack_tile(var_dst,cb_ex2)`, `transpose_wh_tile`, re-pack.
- **Verdict: OUT-OF-SCOPE: transpose.** The packs are the output side of the welford/transpose
  DEST window (mean & var already live in DEST from W.1); no CB-load eltwise op. Not a migratable
  eltwise stage (no CopyTile/BinaryFpu/SFPU producing these DEST tiles).

layernorm_welford.cpp counts: MIGRATABLE 0 | BLOCKED 0 | OUT-OF-SCOPE 2 (welford, transpose).

---

## 4. layernorm_large_tensor_welford.cpp

Already migrated: Var+eps→rsqrt chain (359), unary_bcast COL on cb_ex2pe (384).
The two welford helper functions + the second-pass final-value loop hold the raw stages.

### welford_fuse_pre_add() (41-160)

### Stage LTW.1 — fused pre-add (add_tiles → pack interm)  — `layernorm_large_tensor_welford.cpp:73-95`
- LLK: `add_tiles_init(cb_in,cb_inb)` + `add_tiles` + pack to cb_interm_pre_add.
- **Verdict: MIGRATABLE.** `add<cb_in, cb_inb, cb_interm_pre_add, BroadcastDim::None,
  InputLifecycle::Bulk, InputLifecycle::Bulk, OutputLifecycle::Bulk, Input, Output,
  OperandKind::Block>` over `EltwiseShape::tiles(full_block_size, blk)`. Plain per-block bulk
  add → pack; no DEST carryover to a later stage (the pack closes the window before welford reads
  cb_interm_pre_add). Identical pattern to the already-migrated add in layernorm_welford.cpp:115.

### Stage LTW.2 — restore-state copies (cb_ex/cb_ex2 → DEST)  — `layernorm_large_tensor_welford.cpp:98-130`
- LLK: `copy_tile(cb_ex, mean_dst)`, `copy_tile_to_dst_init_short_with_dt`+`copy_tile(cb_ex2,
  var_dst)`, then `welford_restore_state`, `transpose_wh_tile`, `welford_*`.
- **Verdict: OUT-OF-SCOPE: welford+transpose.** The two copies load mean/var into specific DEST
  slots (mean_dst=1, var_dst=2) purely to seed `welford_restore_state`; they are welford-state
  glue inside the welford DEST window, not a copy→pack eltwise stage. Not migratable.

### Stage LTW.3 — final restore-state copies  — `layernorm_large_tensor_welford.cpp:147-159`
- LLK: `copy_tile(cb_ex,mean_dst)` + `copy_tile(cb_ex2,var_dst)` → `welford_restore_state` →
  `welford_finalize_to_row`.
- **Verdict: OUT-OF-SCOPE: welford.** Same as LTW.2 — copies feed welford state, consumed in DEST.

### welford_no_fuse_pre_add() (179-226)

### Stage LTW.4 — transpose + welford  — `layernorm_large_tensor_welford.cpp:190-225`
- LLK: `transpose_wh_init_short`/`transpose_wh_tile`, `welford_*`.
- **Verdict: OUT-OF-SCOPE: welford+transpose.**

### kernel_main pack/transpose (314-347)

### Stage LTW.5 — pack mean/var + transpose back + re-pack  — `layernorm_large_tensor_welford.cpp:314-347`
- LLK: pack_tile(mean/var), `transpose_wh_tile`, re-pack. Same shape as W.2.
- **Verdict: OUT-OF-SCOPE: transpose.** Output/transpose side of the welford DEST window.

### Second-pass final-value loop (396-504)

### Stage LTW.6 — x-E[x] sub bcast cols  — `layernorm_large_tensor_welford.cpp:401-408`
- LLK: `sub_bcast_cols_init_short(cb_in,cb_ex)` + `sub_tiles_bcast_cols`; result stays in DEST.
- **Verdict: BLOCKED: in-DEST cross-stage fusion.** The sub result is NOT packed — it stays live
  in DEST and is consumed by the dest-reuse add (LTW.7) and dest-reuse mul (LTW.8) before any pack
  at 441. Migratable only as part of the joint fused chain below.

### Stage LTW.7 — FUSE_PRE_ADD dest-reuse add  — `layernorm_large_tensor_welford.cpp:410-422`
- LLK: `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(cb_inb)`.
- **Verdict: BLOCKED: in-DEST cross-stage fusion** (joint-migratable, see below).

### Stage LTW.8 — ·rstd dest-reuse mul (cb_ex2pe)  — `layernorm_large_tensor_welford.cpp:424-444`
- LLK: `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCB>(cb_ex2pe, 0, i)` (ex2pe tile0, col-vec
  broadcast already materialized) + pack to cb_xmm.
- **Verdict: BLOCKED: in-DEST cross-stage fusion** (joint-migratable, see below).

### Stage LTW.6+7+8 (joint) — sub → [dest-reuse add] → dest-reuse mul → pack  — `…welford.cpp:401-444`
- **Verdict: MIGRATABLE** (single fused chain — NO activation macros in this variant).
  - Shape: `EltwiseShape::tiles(full_block_size, blk)`.
  - Elements:
    - `BinaryFpu<cb_in, cb_ex, Sub, BroadcastDim::Col, Bulk, CallerManaged, Input, D0, Block,
      Scalar>` (x-E[x]).
    - if FUSE_PRE_ADD: `DestReuseBinary<cb_inb, Add, DEST_TO_SRCB, Bulk, Input, D0,D0, Block>`.
    - `DestReuseBinary<cb_ex2pe, Mul, DEST_TO_SRCB, CallerManaged, Input, D0,D0, Scalar>`
      (ex2pe is a single held col tile, index 0 → Scalar/CallerManaged).
    - `PackTile<cb_xmm, OutputLifecycle::Bulk, Output, D0>`.
  - Reconfig matches the explicit `reconfig_data_format(cb_in,cb_ex)`, `reconfig_data_format_srca`
    chain (412/421/425) and `pack_reconfig_data_format(cb_xmm)` (437).
  - DestReuseBinary IS the chain element for in-DEST accumulate, so the whole sub→add→mul→pack
    fuses into ONE chain. This is the highest-value unlock in this file.

### Stage LTW.9 — ·gamma (mul bcast rows)  — `layernorm_large_tensor_welford.cpp:447-479`
- LLK: `mul_bcast_rows_init_short(cb_xmm,cb_gamma)` + `mul_tiles_bcast_rows`, compile-time double
  pack (cb_out vs cb_xmm). NO activation macro in this variant.
- **Verdict: MIGRATABLE.** `mul<cb_xmm, cb_gamma, cb_outg, BroadcastDim::Row,
  InputLifecycle::Bulk, InputLifecycle::Bulk, OutputLifecycle::Bulk, Input, Output,
  OperandKind::Block, OperandKind::Block>` over `tiles(full_block_size, blk)`, where
  `cb_outg = do_beta ? cb_xmm : cb_out` (compile-time). Mirrors the already-migrated welford
  gamma stage (layernorm_welford.cpp:290).

### Stage LTW.10 — +beta (add bcast rows)  — `layernorm_large_tensor_welford.cpp:481-503`
- LLK: `add_bcast_rows_init_short(cb_xmm,cb_beta)` + `add_tiles_bcast_rows` + pack cb_out.
  NO activation macro.
- **Verdict: MIGRATABLE.** `add<cb_xmm, cb_beta, cb_out, BroadcastDim::Row, Bulk, Bulk, Bulk,
  Input, Output, Block, Block>` over `tiles(full_block_size, blk)`. Mirrors
  layernorm_welford.cpp:310.

layernorm_large_tensor_welford.cpp counts: MIGRATABLE 4
(LTW.1 fused pre-add; LTW.6+7+8 joint sub→dest-reuse→pack; LTW.9 gamma mul; LTW.10 beta add)
| BLOCKED 3 (LTW.6,LTW.7,LTW.8 individually — but jointly migratable as the one chain counted above)
| OUT-OF-SCOPE 4 (LTW.2,LTW.3,LTW.4,LTW.5 welford/transpose).
Net unique unlocked: 4 chains.

---

## Grand totals (unique unlocked chains)

| Kernel | MIGRATABLE | BLOCKED | OUT-OF-SCOPE |
|---|---|---|---|
| layernorm.cpp | 0 | 3 | 0 |
| layernorm_large_tensor.cpp | 1 | 7* | 1 |
| layernorm_welford.cpp | 0 | 0 | 2 |
| layernorm_large_tensor_welford.cpp | 4 | 3* | 4 |

*BLOCKED counts include the individual in-DEST-fusion stages that are jointly migratable; the
MIGRATABLE column counts the resulting fused chain once.

Dominant blocker = raw host-injected SFPU activation macro (`SFPU_OP_INIT_ACTIVATION` /
`SFPU_OP_FUNC_ACTIVATION`) in the non-welford gamma/beta scale loops — blocks all 3 stages in
layernorm.cpp and 3 stages in layernorm_large_tensor.cpp. The welford variants have no such macro,
so their scale stages migrate cleanly (already done in layernorm_welford.cpp; outstanding in
layernorm_large_tensor_welford.cpp).
