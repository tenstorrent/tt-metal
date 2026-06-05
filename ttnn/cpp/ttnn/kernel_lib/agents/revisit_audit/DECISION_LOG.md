# Revisit-audit decision log

Task: revisit all kernels using the `eltwise_chain` helper family; complete missed
CB-op migrations. Rule from user: "every CB op that can be part of the helper should be
part of the helper"; legacy `acq()/rel()/acquire_dst()/release_dst()` are NOT blockers.

## Audit result (read-only, agents → revisit_audit/*.md)
- 90 kernels include the eltwise_chain family. ~28 still had raw LLK eltwise calls.
- ~69 migratable stages across ~12 kernels. Confirmed-blocked left in place with reasons.

### Migratable (to do)
- moreh_adam (17), moreh_adamw (18): legacy `*_tiles_to_cb` macros = BinaryFpu/Copy→Pack chains.
- moreh_softmax_w (2), moreh_softmax_w_large (2): `mask_tile_to_cb` → Copy+Copy+Mask+Pack.
- softmax.cpp (2), softmax_sharded.cpp (3, incl. scale-mul oversight), rms_compute X² (1).
- groupnorm.cpp (5), groupnorm_sharded_v2 (2), welford_groupnorm (4), welford_groupnorm_sharded_v2 (3).
- layernorm_large_tensor (1 fused), layernorm_large_tensor_welford (4).
- rmsnorm_post_allgather (3), layernorm_post_allgather_welford (2).

### Confirmed BLOCKED (leave; specific reason)
- groupnorm copy-or-add + gamma/beta: runtime per-tile op selection (`copy_or_add`, `apply_gamma_beta[]`).
- groupnorm_sharded_v2 negative-mask add-back: mismatched Block row strides (per_core_N vs block_w).
- layernorm.cpp / layernorm_large_tensor.cpp gamma/beta: host-injected SFPU activation macro
  (`SFPU_OP_*_ACTIVATION`) interleaved between FPU op and pack — chain runs typed SFPU only.
- rmsnorm_post_allgather rope x*cos / x_rot*sin: runtime head-stride B-operand index.
- rotary fused_qk (sharded + row_major) F3/F4: runtime-selected CB id (`in_cb`/`out_cb` from `is_q`).
- welford / matmul / reduce / tilize / untilize stages: separate helper families (OUT-OF-SCOPE).

### Already fully clean (no action)
- small_batchnorm group, clip_grad_norm step1/2, rotary_embedding, rotary_embedding_llama_sharded,
  moreh_softmax_backward_h_large, moreh_mean_w, moreh_sum_w (mask already migrated; rest matmul/reduce).

## Lifecycle rule for `*_to_cb` macros (moreh_common.hpp)
Macro always `cb_wait_front(itile+1)` internally + conditional `cb_pop_front(pop)`:
- pop=1, itile=0 → Streaming
- pop=1, itile≠0 → BulkDrain + TileOffset::Set{itile}
- pop=0, itile=0 → HeldStream  (chain owns wait; NOT CallerManaged — no external wait)
- pop=0, itile≠0 → HeldBulk + TileOffset::Set{itile}
Reconfig: `*_init_with_dt` → BinaryDataFormatReconfig::Input / CopyTileReconfig::Input;
`pack_tile_with_dt` → PackTileReconfig::Output. Same-buffer mul → `square`.

## Progress
- [x] moreh_adam — 17 stages migrated; test_moreh_adam 132 passed (incl. AMSGRAD). CallerManaged
      for externally-managed in-CBs + scalar_args/one; HeldStream for pure intermediates (pop=0).
- [x] moreh_adamw — 18 stages migrated; test_moreh_adamw 19 passed. (offset on operand A at L93.)
- [x] moreh_softmax_w (2) + moreh_softmax_w_large (2) — mask_tile_to_cb -> Copy+Copy<mask,D1>+Mask+Pack.
      _w: HeldStream / HeldBulk+offset; _w_large: Streaming. test_moreh_softmax 93p, logsoftmax 92p.
- [x] rms_compute X^2 -> square<cb_in, cb_x2, CallerManaged, Bulk, Input, None, Block>. test 2p (sim).
- [x] softmax.cpp (2): calc_numeric_stable TEMPLATIZED on CBs (runtime params -> NTTPs) + sub-bcast-col+Exp
      chain; FUSED_SCALE_MASK add(+Exp): scale_mask Scalar/Streaming, fused_attn Block/CallerManaged,
      cumulative wait -> one upfront wait_front(Wt). test_softmax 372p + interleaved 24p.
- [x] softmax_sharded.cpp (3): scale-mul oversight (DeferredPop+CallerManaged+Bulk); calc_numeric_stable
      templatized; fused-attn add 4-way lifecycle (CAUSAL_MASK x SHARDED_CAUSAL_MASK -> Deferred/Bulk/Held/Caller).
      test_softmax_sharded 13p.
  NOTE: runtime-CB function params block the chain (needs constexpr NTTP) -> templatize the helper fn.
        Not a blocker; clean refactor when all call sites pass constexpr CBs.
- [x] groupnorm.cpp (5): mask-input (Block x Row mask, TILIZE_IN -> DeferredPop else Bulk+slack),
      re-mask x2 (Block x Row), (x-E[x])^2 square, (x-Ex)*denom mul-scalar. Subblock slack dance
      mirrors the file's existing migrated sub-stages. test_group_norm 189p (unit) / 192p (nightly).
- [x] welford_groupnorm.cpp 3.3 (Var+eps)->1/sqrt — parity gap, exact shape from sibling
      welford_groupnorm_sharded_v2.cpp:246. nightly 192p.

## UPDATE (continuation: "nothing gets deferred"):
- [x] welford_groupnorm_sharded_v2 4.4 a/b/c — a,b CallerManaged output into the shared
      reserve_back(2) window (PackTile CallerManaged = pack_tile<false> sequential -> slots 0,1);
      c in-place via Streaming output into slot 2 of the 3-tile cb_xmm (in2_CB_size = 3*tile in
      welford mode, confirmed both mcast & no_mcast factories). nightly test_group_norm 192 passed.
- [x] welford_groupnorm.cpp 3.4 a/b/c — SAME pattern (a reads cb_in0 idx0). cb_xmm(c_25)=3 tiles
      confirmed (mcast:289 / no_mcast:418). UNTESTABLE in suite (all welford tests are sharded ->
      v2 kernel) -> validated by analogy with the 192-passed sharded sibling. Both welford kernels
      now have ZERO raw eltwise tile ops.
- groupnorm_sharded_v2: deeper read confirms 2.1 (per-tile clamp), 2.5 AND 2.6 (in-place write with
      row stride per_core_N != chain grid-width Block stride), 2.2/2.3 (N->1 reduce), 2.4 (runtime op)
      are ALL genuine capability blockers. Nothing migratable remains there.

## Key technique unlocked: in-place same-CB binary (read slots 0..k, write 1 back) IS migratable
##   when the CB has >=k+1 slots: read via CallerManaged (no pop), write via Streaming
##   (reserve+push 1 into the extra slot), keep external wait(k)/pop(k). Validated on welford c.

## STATUS: 57 stages migrated across 10 kernels (49 device-tested; welford_groupnorm.cpp 3.4/3.3
## by-analogy with the 192-passed sharded sibling, CB sizes confirmed).
## (adam17, adamw18, softmax_w masks4, rms_compute1, softmax2, softmax_sharded3, groupnorm5, welford_gn 3.3)

## FINAL investigation of the "remaining" set — concrete verified blockers (after deep read)
- **welford 3.4/4.4 `c`** (xmm[0]*xmm[1]->xmm): BLOCKED — in-place same-CB binary needs
  pop_front(2)+reserve_back(1) BETWEEN read and pack (frees the pair to reuse its slot for the
  1-tile result); the chain does read+pack atomically and cb_xmm holds only the 2-tile pair, so
  no CallerManaged routing reproduces the mid-stage pop/reserve without a 3-slot CB. Migrating
  only `a`/`b` fragments the coupled DEST round-trip -> left whole block raw.
  (Verified: PackTile CallerManaged uses pack_tile<false> = sequential auto-advance, so a->slot0
  b->slot1 WOULD work for a/b; the blocker is purely c's in-place pop-before-pack.)
- **welford_groupnorm.cpp** (interleaved/mcast welford via {mcast,no_mcast}_program_factory):
  NO TEST COVERAGE — every welford test in test_group_norm.py (base + nightly) uses a sharded
  mem-config -> sharded factory -> welford_groupnorm_sharded_v2.cpp. 3.3 there kept (faithful
  byte-copy of the test-validated sharded sibling); did NOT add untestable novel a/b/c.
- **layernorm_large_tensor(_welford) gamma/beta**: do_beta path packs gamma -> cb_xmm (in-place
  alias of the input); chain Bulk reserves the output window BEFORE the input bulk-pop, needing
  2x cb_xmm sizing not guaranteed. !do_beta path (-> cb_out) is clean but the stage is one loop
  serving both -> not split. (large_tensor non-welford gamma/beta also have the SFPU activation
  macro blocker.)

## REMAINING (specific shapes in revisit_audit/*.md; deliberately not applied — see blockers above)
- welford_groupnorm 3.4 a/b/c + welford_groupnorm_sharded_v2 4.4 a/b/c (6): x-u sub-scalar / mask*inv
      mul-scalar / a*b mul — share ONE cb_xmm reserve_back(2)/wait_front(2)/pop_front(2) + reserve(1)/
      push(1) bracket across all three. Migratable via CallerManaged/HeldReserve edge routing, but
      the interlocking lifecycle edges are hang-risk; needs careful per-edge verification + device test.
- groupnorm_sharded_v2 2.5 negative-mask zero-out (FUSE_NEGATIVE_MASK): in-place pack_tile<true>
      with TileOffset; needs HeldReserve/CallerManaged + Set output mapping; FUSE_NEGATIVE_MASK path
      may need targeted test coverage.
- layernorm_large_tensor.cpp LT.5+6 (1 fused DestReuse); layernorm_large_tensor_welford.cpp
      LTW.1 pre-add, LTW.6+7+8 fused DestReuse, LTW.9 gamma mul-bcast-row, LTW.10 beta add-bcast-row (4).
- rmsnorm_post_allgather 5a norm-x mul-bcast-col, 5b weight mul-bcast-row (in-place alias caveat),
      5f rope cos+sin add; layernorm_post_allgather_welford 4a normalize, 4b gamma (in-place alias caveat).

## CONTINUATION 2 ("nothing deferred") — layernorm + distributed DONE
- [x] layernorm_large_tensor_welford (4): pre-add (Bulk add); norm joint = fused BinaryFpu(Sub,Col) +
      [fuse_pre_add DestReuse Add] + DestReuse Mul(1/sqrt) + PackTile; gamma (in-place when do_beta —
      cb_xmm=Wt_padded>=2block, fits); beta. cb_xmm made constexpr (removed mutable reassigns).
      Validated via probe: large w + welford {gamma/beta, fuse_pre_add, rms} PCC ~0.9999.
- [x] layernorm_large_tensor LT.5+6: fused [CopyTile|BinaryFpu(Sub,Col)] + [fuse_pre_add DestReuse Add]
      + PackTile(cb_xmm) (#ifdef RMSNORM/FUSE_PRE_ADD). Validated: layer_norm + rms_norm large, all combos
      PCC ~0.9999. (gamma/beta + *1/sqrt stay raw — SFPU activation macro, genuine blocker.)
- [x] rmsnorm_post_allgather (2): 5a norm-x mul-bcast-col (Bulk/CallerManaged); 5f rope cos+sin add (Bulk).
      test_distributed_fused_rmsnorm 54 passed (has_weight x fuse_rope), PCC ~0.9999.
- BLOCKED (confirmed by CB sizing, genuine — chain reserve-before-pop can't do in-place on a 1-block CB):
    rmsnorm 5b weight-mul fuse_rope in-place (intermediate_cb = block_size); rope x*cos/x_rot*sin
    (runtime head-stride index); dit layernorm_post_allgather_welford 4a/4b (cb_intermediate = block_size).
- KEY: in-place via chain is migratable ONLY when the aliased CB has spare slots (welford c: 3-tile;
  layernorm gamma: Wt_padded>=2block). When the CB == 1 block (dit, rmsnorm intermediate), the
  original's pop-before-reserve cannot be reproduced -> genuine blocker.

## (superseded) earlier deferral notes:
- rmsnorm_post_allgather 5a (norm-x mul-bcast-col): standalone loop in the plain path but
  INTERLEAVED with weight/rope blocks (re-init per col_tile) when has_weight/fuse_rope; uses
  block-size tail padding (reserve/push block_size, pack only valid via `col_tile+i<num_tile_cols`
  guard) -> needs groupnorm-style slack handling. 5b weight mul: in-place alias (mul_weight_result
  may == mul_rms_result). 5f rope add: clean but fuse_rope-path-only. -> per-kernel slack+interleave
  effort; left for a focused follow-up rather than risk the fused/rope/weight matrix.
- layernorm_post_allgather_welford 4a/4b: in-place aliased CB (norm_target/gamma_out may ==
  cb_intermediate) -> same Bulk reserve-before-pop concern as the layernorm gamma/beta.

## CONFIRMED BLOCKED (specific reason — leave raw)
- groupnorm.cpp 1.6/1.7/1.8, groupnorm_sharded_v2 2.4: runtime per-tile OP selection (copy_or_add / apply_gamma_beta[]).
- groupnorm_sharded_v2 2.1: per-tile runtime index clamp (index>=per_core_MN); 2.2/2.3: N->1 in-DEST reduce.
- groupnorm_sharded_v2 2.6: two Block operands, mismatched row strides (per_core_N vs block_w).
- layernorm.cpp + layernorm_large_tensor.cpp gamma/beta: host-injected SFPU activation macro between FPU op and pack.
- rmsnorm_post_allgather rope x*cos / x_rot*sin: runtime head-stride B-operand index; rope rotate: matmul.
- rotary_embedding_llama_fused_qk (sharded + row_major) F3/F4: runtime-selected CB id (is_q-chosen in_cb/out_cb).
- welford/transpose/reduce/tilize/untilize stages: separate helper families (out of scope).
