# Eltwise-chain MISSED-migration audit — softmax / distributed-norm kernels

Scope: find every remaining **raw LLK eltwise stage** in the 5 assigned compute
kernels and classify it MIGRATABLE / BLOCKED / OUT-OF-SCOPE against the
`compute_kernel_lib::eltwise_chain` family. Read-only audit — no files edited.

Helper capabilities used as the bar (verified from `eltwise_chain.hpp` /
`eltwise_convenience.hpp` / `eltwise_math.hpp`):
- Elements: CopyTile, BinaryFpu(Add/Sub/Mul, BroadcastDim None/Col/Row/Scalar),
  DestReuseBinary, UnaryBcast, PackTile; SFPU op structs (Exp/Rsqrt/Recip/…) DEST-only
  between CopyTile and PackTile. **Fused BinaryFpu + SFPU + PackTile is supported**
  (e.g. `sub_bcast + exp + pack`, `add + rsqrt + pack`).
- OperandKind Block/Row/Col/Scalar; TileOffset::Set base offset (Bulk-family/CallerManaged).
- Input lifecycles incl. Bulk/HeldBulk/CallerManaged/DeferredPop/Streaming; output
  Bulk/BulkReservePerChunk/Streaming/etc. Chain absorbs cb wait/pop/reserve/push.
- subblock/DEST-batch loops → `EltwiseShape::tiles(n, block_size)`.

NOT blockers (treated MIGRATABLE): tile_regs_*, acquire/release, subblock loops,
bulk-vs-streaming lifecycle, broadcasts, bulk-wait+per-tile-drain, held broadcast
operands.

---

## 1. softmax.cpp (attention/compute/softmax.cpp)

Already heavily migrated (mul/unary/eltwise_chain for scale, exp, recip-mul). Two raw
LLK eltwise stages remain.

### 1a. `calc_numeric_stable` sub-bcast + exp + pack — softmax.cpp:43-60
- Current LLK: `sub_bcast_cols_init_short`; loop `sub_tiles_bcast_cols(cb_in, cb_max)`
  → `exp_tile<EXP_APPROX>` → `pack_tile(cb_out)`. ndst DEST-batch loop.
  cb_max waited(1)/popped(1) outside the loop (lines 42/62); cb_in popped Wt at line 61;
  cb_out reserved/pushed per ndst.
- **Verdict: MIGRATABLE.** This is the canonical fused sub-bcast-col + exp pattern the
  chain supports (cf. the already-migrated `mul` recip step in the same file).
- Chain shape:
  ```
  eltwise_chain(EltwiseShape::tiles(Wt /*, ndst as block_size if made CT*/),
      BinaryFpu<cb_in, cb_max, BinaryFpuOp::Sub, BroadcastDim::Col,
                InputLifecycle::DeferredPop /*cb_in: no wait, bulk-pop Wt at end*/,
                InputLifecycle::CallerManaged /*cb_max: held, wait/pop kept outside*/,
                BinaryDataFormatReconfig::Input, Dst::D0,
                OperandKind::Block, OperandKind::Scalar>{},
      Exp<Approx(EXP_APPROX), Approx::Exact, Dst::D0>{},
      PackTile<cb_out, OutputLifecycle::Streaming>{});
  ```
  Note: original `cb_in` has no per-tile wait_front (reader pre-pushed) and a single
  trailing `pop_front(Wt)` → DeferredPop. cb_max is the held per-row max → CallerManaged
  (keep the surrounding wait_front(1)/pop_front(1)), or HeldBulk to fold the wait. ndst is
  a runtime arg so block_size stays 1 unless promoted (same trade-off already accepted in
  this file's other migrated loops).

### 1b. FUSED_SCALE_MASK add(+exp) + pack — softmax.cpp:148-179
- Current LLK: CAUSAL `add_tiles(cb_scale_mask, cb_fused_attn)`; non-causal
  `add_tiles_bcast_rows(...)`; both then `[exp_tile if !NUMERIC_STABLE]` → `pack_tile(cb_x)`.
  cb_scale_mask waited/popped per ndst; cb_fused_attn cumulative-waited (gated by `wait_mask`
  in the non-causal case so the mask row is read once and reused across Ht).
- **Verdict: MIGRATABLE.** Fixed op (Add) + optional Exp + pack — exactly the BinaryFpu+SFPU
  chain. The `wait_mask` gating is a held-broadcast operand pattern: caller keeps the
  conditional `cb_fused_attn` wait/pop, chain runs CallerManaged on that operand.
- Chain shape (non-causal, !NUMERIC_STABLE):
  ```
  eltwise_chain(EltwiseShape::tiles(Wt),
      BinaryFpu<cb_scale_mask, cb_fused_attn, BinaryFpuOp::Add, BroadcastDim::Row,
                InputLifecycle::Streaming   /*cb_scale_mask: wait/pop ndst→per-tile*/,
                InputLifecycle::CallerManaged/*cb_fused_attn: wait_mask-gated, held*/,
                ..., OperandKind::Scalar, OperandKind::Row>{},
      Exp<...>{},
      PackTile<cb_x, OutputLifecycle::Streaming>{});
  ```
  CAUSAL variant: `BroadcastDim::None`, both operands `OperandKind::Block`, cb_fused_attn
  cumulative-waited then popped Wt → DeferredPop/Bulk. The `#ifdef NUMERIC_STABLE` simply
  drops the Exp element from the chain (compile-time). The `wait_mask`/`ht` bookkeeping and
  the final conditional `pop_front` stay in the caller (CallerManaged side).

softmax.cpp count: **2 migratable**, 0 blocked, 0 out-of-scope.
(reduce<MAX>/reduce<SUM> calls already use reduce_helpers — not eltwise, out of scope.)

---

## 2. softmax_sharded.cpp (attention/compute/softmax_sharded.cpp)

Same two-stage structure as softmax.cpp; the non-fused exp path and the recip-mul are
already migrated. Two raw LLK eltwise stages remain (mirrors of softmax.cpp).

### 2a. `calc_numeric_stable` sub-bcast + exp + pack — softmax_sharded.cpp:34-53
- Current LLK: `sub_bcast_cols_init_short`; `sub_tiles_bcast_cols(cb_in, cb_max)` →
  `exp_tile` → `pack_tile(cb_out)`, subblock_w DEST-batch loop over num_subblocks_w.
  cb_max waited(1)/popped(1) outside; cb_in popped block_w at end; cb_out reserved/pushed
  per subblock. (Note: there is a redundant double `reserve_back(subblock_w)` at lines
  38 & 43 in the original — a pre-existing quirk, not a migration blocker.)
- **Verdict: MIGRATABLE.** Same fused sub-bcast-col + exp shape as 1a; subblock_w is a
  compile-time arg here, so it maps cleanly to `block_size`.
- Chain shape:
  ```
  eltwise_chain(EltwiseShape::tiles(block_w, subblock_w),
      BinaryFpu<cb_in, cb_max, BinaryFpuOp::Sub, BroadcastDim::Col,
                InputLifecycle::DeferredPop, InputLifecycle::CallerManaged,
                BinaryDataFormatReconfig::Input, Dst::D0,
                OperandKind::Block, OperandKind::Scalar>{},
      Exp<...>{},
      PackTile<cb_out, OutputLifecycle::Bulk>{});
  ```

### 2b. FUSED_SCALE_MASK add(+exp) + pack — softmax_sharded.cpp:147-174
- Current LLK: CAUSAL `add_tiles`; non-causal `add_tiles_bcast_rows`; `[exp]`;
  `pack_tile(cb_x)`. subblock_w DEST-batch loop. cb_scale_mask waited(block_w)/popped(block_w)
  bracketing the loop; cb_fused_attn waited(block_w) once (non-causal) / popped(block_w) (causal).
- **Verdict: MIGRATABLE.** Fixed Add + optional Exp + pack. subblock_w → block_size.
- Chain shape (non-causal):
  ```
  eltwise_chain(EltwiseShape::tiles(block_w, subblock_w),
      BinaryFpu<cb_scale_mask, cb_fused_attn, BinaryFpuOp::Add, BroadcastDim::Row,
                InputLifecycle::Bulk    /*cb_scale_mask: wait/pop block_w*/,
                InputLifecycle::HeldBulk/*cb_fused_attn: wait block_w, no pop (held)*/,
                ..., OperandKind::Block, OperandKind::Row>{},
      Exp<...>{},
      PackTile<cb_x, OutputLifecycle::Bulk>{});
  ```
  CAUSAL: BroadcastDim::None, both Block, cb_fused_attn Bulk (pops block_w). `#ifdef
  NUMERIC_STABLE` drops the Exp element. The `cb_scale_mask` scale stage above it
  (lines 110-125, `mul_tiles_bcast_scalar`) is ALSO raw LLK — see 2c.

### 2c. FUSED_SCALE_MASK scale mul-bcast-scalar + pack — softmax_sharded.cpp:108-125
- Current LLK: `mul_tiles_bcast_scalar_init_short`; `mul_tiles_bcast_scalar(cb_in0,
  cb_fused_scale)` → `pack_tile(cb_scale_mask)`, subblock_w loop. cb_fused_scale held
  (waited(1) at line 107, not popped); cb_in0 popped block_w at line 126.
- **Verdict: MIGRATABLE.** Identical shape to softmax.cpp's already-migrated scale step
  (`mul<cb_in0, cb_fused_scale, cb_scale_mask, BroadcastDim::Scalar, …, CallerManaged>`),
  which softmax_sharded left as raw LLK. Pure oversight.
- Chain shape:
  ```
  mul<cb_in0, cb_fused_scale, cb_scale_mask,
      BroadcastDim::Scalar,
      InputLifecycle::Bulk         /*cb_in0: wait/pop block_w*/,
      InputLifecycle::CallerManaged/*cb_fused_scale: held*/,
      OutputLifecycle::Bulk,
      BinaryDataFormatReconfig::Input, PackTileReconfig::Output,
      OperandKind::Block, OperandKind::Scalar>(EltwiseShape::tiles(block_w, subblock_w));
  ```

softmax_sharded.cpp count: **3 migratable**, 0 blocked, 0 out-of-scope.

---

## 3. rms_compute.cpp (experimental/ccl/rms_allgather/.../rms_compute.cpp)

Heavily migrated already (pre-add `add`, two `mul` bcast steps, rsqrt eltwise_chain, three
`reduce<>` calls). One raw LLK eltwise stage remains.

### 3a. X^2 mul-square + pack — rms_compute.cpp:107-126
- Current LLK: `mul_tiles_init(cb_in, cb_in)`; `mul_tiles(cb_in, cb_in, index, index, w)` →
  `pack_tile(cb_x2)`, subblock_w DEST-batch loop over num_subblocks_w. cb_in is the resident
  /preadded input (NOT waited or popped here — preceding code owns it); cb_x2 reserved
  (num_tiles_per_block) at line 109 and pushed once at line 126.
- **Verdict: MIGRATABLE.** This is x*x — the `square<>` convenience (BinaryFpu reading the
  same buffer for both operands). num_tiles_per_block = num_subblocks_w*subblock_w so
  `tiles(n, subblock_w)` covers the subblock loop.
- Chain shape:
  ```
  square<cb_in, cb_x2,
      InputLifecycle::CallerManaged /*cb_in: resident, no wait/pop here*/,
      OutputLifecycle::Bulk         /*cb_x2: reserve+push num_tiles_per_block*/,
      BinaryDataFormatReconfig::Input, PackTileReconfig::Output,
      OperandKind::Block>(EltwiseShape::tiles(num_tiles_per_block, subblock_w));
  ```
  (Equivalent to `mul<cb_in, cb_in, cb_x2, …, OperandKind::Block>`. The downstream
  reconfig at line 100-101 / 129-130 stays caller-side as today.)

### 3b. E(x^2) reduce — rms_compute.cpp:137-153  — OUT-OF-SCOPE: reduce
  `reduce_init/reduce_tile<AVG, REDUCE_ROW>` — a reduction, not eltwise. (The `tensix_sync()`
  workaround at line 142 is reduce-internal.) Not chain-eligible.

rms_compute.cpp count: **1 migratable** (X^2), 0 blocked, 1 out-of-scope (reduce).

---

## 4. layernorm_post_allgather_welford.cpp (experimental/transformer/dit_layernorm_post_all_gather/...)

Migrated: rsqrt eltwise_chain, x-mean `sub`, optional-beta eltwise_chain. Welford combine is
its own kernel-util helper. One raw LLK eltwise stage remains.

### 4a. normalize (x-mean)*inv_std mul-bcast-col + pack — welford.cpp:120-143
- Current LLK: `mul_bcast_cols_init_short(cb_intermediate, cb_recip_sqrt_var)`;
  `mul_tiles_bcast_cols(cb_intermediate, cb_recip_sqrt_var, i, 0, i)` → `pack_tile(norm_target_cb)`,
  block_size DEST-batch loop. cb_intermediate waited(block_size)/popped(block_size);
  cb_recip_sqrt_var waited(1) (held per-row, popped at line 205); **norm_target_cb may equal
  cb_intermediate** (in-place when no gamma/beta) — hence the deliberate compute/pack split
  with pop-before-reserve.
- **Verdict: MIGRATABLE.** mul-bcast-col with a held scalar broadcast operand — same shape
  as the already-migrated x-mean `sub` two stages above it. The in-place case
  (norm_target_cb == cb_intermediate) is handled by the chain's own pop-then-reserve
  ordering (it pops the input before reserving the output, same as the original split).
- Chain shape:
  ```
  mul<cb_intermediate, cb_recip_sqrt_var, norm_target_cb,
      BroadcastDim::Col,
      InputLifecycle::Bulk          /*cb_intermediate: wait/pop block_size*/,
      InputLifecycle::CallerManaged /*cb_recip_sqrt_var: held per-row*/,
      OutputLifecycle::Bulk,
      BinaryDataFormatReconfig::Input, PackTileReconfig::Output,
      OperandKind::Block, OperandKind::Scalar>(EltwiseShape::tiles(block_size, block_size));
  ```
  CAUTION (correctness note, not a blocker): verify the chain's input-pop happens before the
  output-reserve so the in-place norm_target_cb==cb_intermediate case stays valid. Bulk input
  (pop AtEnd) + Bulk output (reserve Upfront) reserves before the end-pop — for the in-place
  case prefer `OutputLifecycle::HeldReserve`/`DeferredReserve` or keep block_size DEST
  staging so the read completes before the reserve, matching the original's explicit
  pop_front(cb_intermediate) → reserve_back(norm_target_cb) ordering. If the chain cannot
  guarantee pop-before-reserve for the aliased CB, this stage is **BLOCKED: in-place aliased
  in/out CB requires pop-before-reserve ordering the Bulk output policy does not emit** —
  flag for verification before migrating.

### 4b. optional gamma mul-bcast-row + pack — welford.cpp:146-170
- Current LLK: `mul_bcast_rows_init_short(norm_target_cb, cb_gamma)`;
  `mul_tiles_bcast_rows(norm_target_cb, cb_gamma, i, col_tile+i, i)` → `pack_tile(gamma_out_cb)`.
  cb_gamma cumulative-waited (col_tile+block_size, held across the col loop, popped at end);
  norm_target_cb waited/popped block_size; **gamma_out_cb may equal cb_intermediate** (in-place
  when do_beta) — again the pop-before-reserve split.
- **Verdict: MIGRATABLE** (same caveat as 4a re in-place ordering). This is the exact twin of
  rms_compute.cpp's already-migrated gamma `mul<…BroadcastDim::Row…>`. TileOffset::Set(col_tile)
  on the held cb_gamma operand.
- Chain shape:
  ```
  mul<norm_target_cb, cb_gamma, gamma_out_cb,
      BroadcastDim::Row,
      InputLifecycle::Bulk          /*norm_target_cb: wait/pop block_size*/,
      InputLifecycle::CallerManaged /*cb_gamma: cumulative-held, caller owns wait/pop*/,
      OutputLifecycle::Bulk, ...,
      OperandKind::Block, OperandKind::Block /* +TileOffset::Set{col_tile} on B */>(
      EltwiseShape::tiles(block_size, block_size));
  ```
  (cb_gamma column offset = col_tile → TileOffset::Set on the B operand, CallerManaged so the
  cumulative wait_front(col_tile+block_size) and end pop stay caller-side.)

welford.cpp count: **2 migratable** (normalize, gamma) — both with an in-place pop-before-reserve
ordering caveat to verify; 0 hard-blocked; combine_welford = kernel-util helper, not raw eltwise.

---

## 5. rmsnorm_post_allgather.cpp (experimental/transformer/fused_distributed_rmsnorm/...)

Migrated: reduce<AVG>, rsqrt eltwise_chain. The norm-x mul and the gamma mul are raw LLK; the
ROPE block is matmul + mul + add (mostly out-of-scope).

### 5a. norm-x mul-bcast-col + pack — rmsnorm_post_allgather.cpp:116-134
- Current LLK: `mul_bcast_cols_init_short(input_cb, reduce_result_cb)`;
  `mul_tiles_bcast_cols(input_cb, reduce_result_cb, i, 0, i)` → `pack_tile(mul_rms_result_cb)`,
  block_size loop with `col_tile+i < num_tile_cols` tail guard. reduce_result_cb held (wait(1)
  at line 119, pop at 285); input_cb waited/popped block_size.
- **Verdict: MIGRATABLE.** mul-bcast-col with held scalar — identical to welford 4a / rms_compute
  mul. The `col_tile+i < num_tile_cols` guard is a tail-clamp; since `num_tile_cols % block_size`
  handling is the only wrinkle, use `EltwiseShape::tiles(num_tile_cols, block_size)` and let the
  shape walk the exact tile count (chain iterates the true count, no manual tail guard needed).
- Chain shape:
  ```
  mul<input_cb, reduce_result_cb, mul_rms_result_cb,
      BroadcastDim::Col,
      InputLifecycle::Bulk          /*input_cb: wait/pop block_size→num_tile_cols*/,
      InputLifecycle::CallerManaged /*reduce_result_cb: held per-row*/,
      OutputLifecycle::Bulk,
      BinaryDataFormatReconfig::Input, PackTileReconfig::Output,
      OperandKind::Block, OperandKind::Scalar>(EltwiseShape::tiles(num_tile_cols, block_size));
  ```
  Caveat: when `has_weight`/`fuse_rope`, this mul is interleaved INSIDE the col_tile loop with
  the weight/rope blocks and re-inits the mul_bcast_col each iteration (lines 170-172, 280-282).
  Pulling it into a single whole-row chain only works in the **plain path** (no weight, no rope)
  where the col loop body is just this one mul. With weight/rope the per-iter interleave means
  the mul stays a per-block_size chain call inside the loop (still migratable, block_size shape,
  but not hoistable to a single call). Either way the stage itself is MIGRATABLE.

### 5b. weight (gamma) mul-bcast-row + pack — rmsnorm_post_allgather.cpp:139-167
- Current LLK: `mul_bcast_rows_init_short(mul_rms_result_cb, weight_cb)`;
  `mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile+i, i)` →
  `pack_tile(mul_weight_result_cb)`. weight_cb cumulative-waited (col_tile+block_size, popped
  at line 296); mul_rms_result_cb waited/popped block_size; **mul_weight_result_cb may alias
  mul_rms_result_cb** (in-place when fuse_rope) → pop-before-reserve split.
- **Verdict: MIGRATABLE** (in-place ordering caveat as 4a). Twin of welford 4b / rms_compute
  gamma. TileOffset::Set(col_tile) on held weight_cb.
- Chain shape:
  ```
  mul<mul_rms_result_cb, weight_cb, mul_weight_result_cb,
      BroadcastDim::Row,
      InputLifecycle::Bulk          /*mul_rms_result_cb: wait/pop block_size*/,
      InputLifecycle::CallerManaged /*weight_cb: cumulative-held*/,
      OutputLifecycle::Bulk, ...,
      OperandKind::Block, OperandKind::Block /* +TileOffset::Set{col_tile} on B */>(
      EltwiseShape::tiles(block_size, block_size));
  ```

### 5c. ROPE rotate matmul — rmsnorm_post_allgather.cpp:182-197 — OUT-OF-SCOPE: matmul
  `mm_init_short` + `matmul_tiles(intermediate_cb, transformation_mat_cb)`. Not eltwise.

### 5d. ROPE x*cos mul + pack — rmsnorm_post_allgather.cpp:202-225
- Current LLK: `mul_tiles_init(intermediate_cb, rope_cos_cb)`; `mul_tiles(intermediate_cb,
  rope_cos_cb, i, rope_cos_tile_in_head, i)` → `pack_tile(intermediate_cb)` (in-place).
  rope_cos indexed by a **runtime per-iter head-stride counter** (`rope_cos_tile_in_head`,
  incremented and wrapped at head_dim_tiles inside the loop).
- **Verdict: BLOCKED: per-tile runtime B-operand tile index (`rope_cos_tile_in_head`
  head-stride wrap).** The chain's OperandKind/TileOffset express a fixed base + kind-derived
  index, not a caller-mutated per-iter counter that wraps mid-loop. The wrap
  (`if (==head_dim_tiles) reset 0`) is a runtime modulo over the iteration that no
  OperandKind/TileOffset combination reproduces. Also in-place pack into the read CB. Not
  migratable without a head-stride index mode the chain does not have.

### 5e. ROPE x_rotated*sin mul + pack — rmsnorm_post_allgather.cpp:230-254
- **Verdict: BLOCKED: per-tile runtime B-operand tile index (`rope_sin_tile_in_head`
  head-stride wrap).** Same blocker as 5d (rope_sin counter, in-place pack).

### 5f. ROPE cos_interim + sin_interim add + pack — rmsnorm_post_allgather.cpp:259-274
- Current LLK: `add_tiles_init(intermediate_cb, rotated_input_cb)`;
  `add_tiles(intermediate_cb, rotated_input_cb, i, i, i)` → `pack_tile(output_cb)`, block_size
  loop with tail guard. Both inputs waited(block_size)/popped(block_size); output reserved/pushed.
- **Verdict: MIGRATABLE.** Plain streaming binary add, both Block index, no broadcast, no
  in-place alias, no runtime index. Straight `add<>`.
- Chain shape:
  ```
  add<intermediate_cb, rotated_input_cb, output_cb,
      BroadcastDim::None,
      InputLifecycle::Bulk, InputLifecycle::Bulk, OutputLifecycle::Bulk,
      BinaryDataFormatReconfig::Input, PackTileReconfig::Output,
      OperandKind::Block, OperandKind::Block>(EltwiseShape::tiles(block_size, block_size));
  ```
  (block_size tail-guarded by `col_tile+i < num_tile_cols`; use a tiles() count that matches
  the live tile span as in 5a.)

rmsnorm_post_allgather.cpp count: **3 migratable** (norm-x mul, weight mul, rope add),
**2 blocked** (rope x*cos, rope x_rotated*sin — runtime head-stride index),
**1 out-of-scope** (rope rotate matmul).

---

## TOTALS

| Kernel | Migratable | Blocked | Out-of-scope |
|---|---|---|---|
| softmax.cpp | 2 | 0 | 0 |
| softmax_sharded.cpp | 3 | 0 | 0 |
| rms_compute.cpp | 1 | 0 | 1 (reduce) |
| layernorm_post_allgather_welford.cpp | 2* | 0 | 0 |
| rmsnorm_post_allgather.cpp | 3 | 2 | 1 (matmul) |
| **TOTAL** | **11** | **2** | **2** |

\* welford normalize+gamma carry an in-place-aliased-CB pop-before-reserve ordering caveat to
verify before migration (could become BLOCKED if the chain's Bulk output reserves before the
input bulk-pop for the aliased case).

### Migratable stage names
- softmax.cpp: `calc_numeric_stable` sub-bcast+exp (43-60); FUSED_SCALE_MASK add(+exp) (148-179)
- softmax_sharded.cpp: `calc_numeric_stable` sub-bcast+exp (34-53); FUSED_SCALE_MASK add(+exp)
  (147-174); FUSED_SCALE_MASK scale mul-bcast-scalar (108-125)
- rms_compute.cpp: X^2 square (107-126)
- welford.cpp: normalize mul-bcast-col (120-143); gamma mul-bcast-row (146-170)
- rmsnorm_post_allgather.cpp: norm-x mul-bcast-col (116-134); weight mul-bcast-row (139-167);
  rope cos+sin add (259-274)

### Blocked
- rmsnorm_post_allgather.cpp rope x*cos (202-225) — runtime head-stride B index
- rmsnorm_post_allgather.cpp rope x_rotated*sin (230-254) — runtime head-stride B index

### Out-of-scope
- rms_compute.cpp E(x^2) reduce (137-153) — reduction
- rmsnorm_post_allgather.cpp rope rotate (182-197) — matmul
