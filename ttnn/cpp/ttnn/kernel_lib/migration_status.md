# Eltwise-chain migration status (cross-op)

Top-level ledger of kernels migrated to `compute_kernel_lib::eltwise_chain` on
`astancov/eltwise_vx_migration`. Tracks both fully-migrated and PARTIAL
kernels, and the specific raw-LLK blocks still pending in each PARTIAL.

Sister file `ttnn/cpp/ttnn/operations/normalization/migration_status.md`
covers the original normalization-only audit (per-kernel difficulty
classification + cross-op attempts); this file is the live ledger for
*any* op being migrated on this branch.

## Status legend

- **FULL** — every raw eltwise-shaped block in the file is now in a chain (or
  documented as out-of-scope).
- **PARTIAL** — at least one block migrated; one or more raw blocks remain.
  Each PARTIAL row lists every un-migrated block with a reason.
- **PLANNED** — queued for an upcoming session.

## Reconfig audit policy

For each migration, the `BinaryDataFormatReconfig` / `PackTileReconfig` /
`CopyTileReconfig` flags are set to match the **original LLK calls in this
kernel**, not what the reference branch
`astancov/eltwise_run7_refined_rebase_v2` did. See
`~/.claude/projects/-localdev-astancov-tt-metal/memory/feedback_eltwise_reconfig_audit.md`
for the rule.

## Migrated kernels

### moreh

| Kernel | Status | Commit | Notes |
|---|---|---|---|
| `moreh_dot/.../moreh_dot.cpp` | FULL | `52c461763c5` | One chain: BinaryFpu(Mul) + PackTile(c_24, None reconfig). Original `mul_tiles_init` reconfigs srca/srcb -> `Input`; original plain `pack_tile` to c_24 with pack format=c_16 from startup -> `None` (preserve original pack format assumption). 21 PASS on `test_moreh_dot.py`. |
| `moreh_dot_backward/.../moreh_dot_backward.cpp` | FULL | `5cb08a20f15` | Two chains (has_input_grad / has_other_grad), each BinaryFpu(Mul, BroadcastDim::Scalar) + PackTile. Original `init_bcast` outside loop + raw `mul_tiles_bcast / pack_tile` inside loop with NO per-iter reconfig -> both flags `None`. 60 PASS on `test_moreh_dot_backward.py`. |
| `moreh_clip_grad_norm/.../moreh_clip_grad_norm_step1_kernel.cpp` | PARTIAL | `d47573e2270` | Migrated: tile_idx==0 seed copy (CopyTile, `CopyTileReconfig::Input`, `PackTileReconfig::None`); tile_idx>0 accumulator add (BinaryFpu Add, `Input`, `None`). Skipped: <ul><li>abs+mask prologue (do_mask_h × do_mask_w 4 branches; needs `Mask` op struct + per-branch chain split — refactor)</li><li>`power_tile_to_cb` (moreh composite helper: abs + power-iterative + log + exp + mul; multi-stage chain decomposition)</li></ul> 20 PASS on `test_moreh_clip_grad_norm.py`. |
| `moreh_clip_grad_norm/.../moreh_clip_grad_norm_step2_kernel.cpp` | PARTIAL | `b6a3c70773e` | Same seed+accumulator pattern as step1, no abs/mask prologue. Skipped: <ul><li>`power_tile_to_cb` (same composite helper)</li></ul> 20 PASS (same test as step1). |

### normalization

| Kernel | Status | Commit | Notes |
|---|---|---|---|
| `batch_norm/.../batch_norm_kernel.cpp` | FULL | `17056fe08cb` | 1-tile BinaryFpu+Rsqrt+PackTile prologue + fused Stage 2-4 via DestReuseBinary. 20/20 PASS. (See normalization/migration_status.md for the full audit.) |
| `layernorm_distributed/.../layernorm_pre_allgather.cpp` | FULL | `189ef03f223` | Squaring chain over `EltwiseShape::of(Wt/blk, blk)`. Bit-exact-probe verified. |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | FULL | `c41aca20b88` | Same squaring chain. Bit-exact-probe verified. |
| `rmsnorm_distributed/.../rmsnorm_post_allgather.cpp` | FULL | `72388c98689` | 4-stage chain (add+rsqrt + x*recip + [*gamma] + [+beta]). 36/36 PASS subset. |
| `groupnorm/.../groupnorm.cpp` | PARTIAL | `3e5f209fa06` | Migrated: (Var+eps) -> rsqrt Variance Calc block (1 chain). Reconfig: `Input` + `None`. cb_ex_global Streaming, cb_eps CallerManaged (held via `cb_eps.wait_front(1)` outside chain), cb_ex2pe OutStreaming. rsqrt Legacy::On. **PARTIAL-DEFER**: local reduce / global reduce already library-based; cb_x prep + (x-Ex)*rsqrt+gamma+beta multi-stage block-strided normalization (lines 606+) — chain support for subblock-strided indexing needs verification. 189 PASS on `test_group_norm.py`. |
| `groupnorm/.../welford_groupnorm.cpp` | PARTIAL | `92a19132dd7` | Migrated: per-group (Var+eps) -> rsqrt strided loop (chain per group with TileBaseRuntime(1+(g<<1)) for cb_ex_global stride-2 indexing). Reconfig: `Input` + `None`. cb_ex_global HeldBulk + Scalar + TileBaseRuntime, cb_eps CallerManaged, cb_ex2pe OutBulk per call (replaces upfront reserve(num_groups)+push_at_end). rsqrt Legacy::On. **PARTIAL-DEFER**: tilize/pack-untilize sharded layout, welford accumulator (OOS), final normalization with subblock dispatch + mask handling. 189 PASS. |
| `groupnorm/.../groupnorm_sharded_v2.cpp` | PARTIAL | `abe68ea4c28` | Same (Var+eps) -> rsqrt migration as basic groupnorm.cpp. Reconfig: `Input` + `None`. rsqrt Legacy::On. 189 PASS. |
| `groupnorm/.../welford_groupnorm_sharded_v2.cpp` | PARTIAL | `66907980ee9` | Same per-group strided (Var+eps) -> rsqrt migration as welford_groupnorm.cpp. TileBaseRuntime(1+(g<<1)) per-group chain call. 189 PASS. |
| `layernorm/.../layernorm.cpp` | PARTIAL | `5d0e13a7f96`, `7fa2452a677` | Migrated (4 chains): Var[x]+eps -> rsqrt; FUSE_PRE_ADD X+Y per-block Add; x - E[x] per-block sub_bcast_cols; (x-E[x])^2 same-CB Mul with cumulative wait + TileBaseRuntime(block.start()). **Real blocker for remaining 3 blocks**: preprocessor-macro-injected SFPU activation (SFPU_OP_INIT/FUNC_ACTIVATION) between binary op and pack inside the DEST window. No chain element wraps user-defined preprocessor macros, and the activation runs in-DEST so it can't split across the chain boundary without an intermediate CB. Resolvable by adding a kernel-local `ActivationMacro : UnaryOp` struct guarded by `#ifdef SFPU_OP_INIT_ACTIVATION`, with care to alias `slot_offset` to the macro's expected loop variable. Deferred to a focused follow-up. 52 PASS on `test_layernorm.py`. |
| `layernorm/.../layernorm_welford.cpp` | PARTIAL | `547aa2f1ec3`, `f53bc329ce1` | 6 chains migrated: Var(x)+eps rsqrt; FUSE_PRE_ADD X+Y per-block Add; x - E[x] per-block sub_bcast_cols; (x-E[x]) * recip col bcast (HeldBulk + TileBaseRuntime(start)); gamma mul_bcast_rows; beta add_bcast_rows. **Real blocker for remaining welford accumulator (lines 138-200)**: in-DEST persistent accumulator across blocks + transpose_wh_tile interleaved with welford_update + welford_finalize_to_row dual-pack. cite: HQ §"CB Lifecycle Taxonomy" — cumulative wait + in-DEST hold is OOS. 52 PASS on `test_layernorm.py`. |
| `layernorm/.../layernorm_large_tensor.cpp` | PARTIAL | `66c0ea6590b`, `0b0ca49146e` | 2 chains migrated: Var(X)+eps -> rsqrt + UnaryBcast<COL> on cb_ex2pe. **Real blocker for remaining blocks A/D/E/F**: (1) Block A (pass-1 variance calc) has interleaved FPU(sub_bcast/binary_dest_reuse(ELWADD)) + SFPU(square_tile) + reduce_init/reduce_tile + scalar mul_unary in one DEST window; cite: HQ §"Control-flow shape" "Interleaved op classes in one DEST window — only migratable if helper exposes a fusion point". (2) Blocks D/E/F have macro-injected SFPU activation (SFPU_OP_INIT/FUNC_ACTIVATION at line 335) and same fusion-point blocker. 52 PASS. |
| `layernorm/.../layernorm_large_tensor_welford.cpp` | PARTIAL | `decf009afc0`, `3016a1d484f` | 2 chains migrated: Var(X)+eps -> rsqrt + UnaryBcast<COL> on cb_ex2pe. Welford accumulator + transpose_wh OOS. **PARTIAL-DEFER (specific patterns, not lazy)**: Blocks A (FUSE_PRE_ADD add+pack), F (pass-2 sub_bcast + binary_dest_reuse(ELWMUL) + pack), G (gamma mul_bcast_rows), H (beta add_bcast_rows) all have structural patterns identified; deferred for focused follow-up. cite: lines 71-94, 407-451, 455-487, 489-511. 52 PASS. |

### rotary_embedding family

| Kernel | Status | Commit | Notes |
|---|---|---|---|
| `experimental/.../rotary_embedding/.../rotary_embedding.cpp` | PARTIAL | `54d28d9b172` | 2 chains migrated: rotated × scalar (-1) and final add cos_interm + sin_interm. MUL_TILES helper (runtime-parameterized) stays raw — needs templated chain wrapper. 750 PASS on `test_rotary_embedding.py`. |
| `experimental/.../rotary_embedding/.../rotary_embedding_single_tile.cpp` | PARTIAL | `c941a8bd3ce` | 3 chains migrated (rotated×sin, in×cos, cos_interm + sin_interm). Matmul rotate_half OOS. 750 PASS. |
| `experimental/.../rotary_embedding_llama/.../rotary_embedding_llama.cpp` | PARTIAL | `e4fe7ef5795` | 3 chains migrated; matmul rotate_half OOS. RELOAD_IMPL = 0/1 use different sin/cos lifecycles (TileBaseRuntime(sin_cos_row_cnt * Wt) under 0, Bulk under 1). 424 PASS on `test_rotary_embedding_llama.py`. |
| `experimental/.../rotary_embedding_llama/.../rotary_embedding_llama_sharded.cpp` | PARTIAL | `e1134ae5e31` | 3 chains migrated (sin/cos mul ROW bcast, add). HeldBulk + Block on sin/cos. 290 PASS. |
| `experimental/.../rotary_embedding_llama_fused_qk/.../rotary_embedding_llama_sharded.cpp` | PARTIAL | `b0166a5e82a` | Only sin stage migrated. cos and add BLOCKED on runtime in_cb/out_cb (q vs k via runtime is_q arg) + TRISC2 size budget rules out duplication workaround. cite: file line 17 size-budget comment. 16 PASS on `test_rotary_embedding_llama_fused_qk.py`. |
| `experimental/.../rotary_embedding_llama_fused_qk/.../rotary_embedding_llama_sharded_row_major.cpp` | PARTIAL | `d2ba9f29d82` | Same as sibling sharded — only sin stage migrates; cos/add BLOCKED on runtime CB ids. 16 PASS. |

### softmax attention

| Kernel | Status | Commit | Notes |
|---|---|---|---|
| `normalization/softmax/.../attention/compute/softmax.cpp` | PARTIAL | `d0b0d9a5f78` | Plain `else` branch (no FUSED_SCALE_MASK + no NUMERIC_STABLE) migrated: CopyTile + Exp + PackTile chain. BLOCKED: <ul><li>NUMERIC_STABLE: helper takes runtime CB ids</li><li>FUSED_SCALE_MASK: cumulative wait on cb_fused_attn (HQ §"Non-goals" OOS) + conditional add_tiles_bcast_rows on last subblock tile</li><li>mul-by-recipsumexps: cumulative wait + runtime ndst</li></ul> 24 PASS on `test_softmax_interleaved.py`. |
| `normalization/softmax/.../attention/compute/softmax_sharded.cpp` | PARTIAL | `b1637a6cca5` | Same plain `else` branch migrated: per-subblock CopyTile + Exp + PackTile with TileBaseRuntime(index_subblock_w_offset) + HeldBulk + Block. Other paths same blockers as softmax.cpp. 13 PASS on `test_softmax_sharded.py`. |

### distributed normalization post_allgather rsqrt stages

| Kernel | Status | Commit | Notes |
|---|---|---|---|
| `layernorm_distributed/.../layernorm_post_allgather.cpp` | PARTIAL | `6277004b227` | 3 chains: E[x]² (same-CB Mul + TileBaseCompileTime<1>), E[x²]-E[x]² (Sub), Var+eps rsqrt (Add+Rsqrt). Reduce stages OOS (use reduce_helpers). chain_llk gamma/beta BLOCKED on chain_llk substitution playbook. 79 PASS on `test_distributed_layernorm_post_allgather.py -k layernorm`. |
| `layernorm_distributed/.../layernorm_post_allgather_welford.cpp` | PARTIAL | `9ce11f1cd76` | Migrated rsqrt stage with HeldBulk+TileBaseCompileTime<1> for cb_stats_reduced (variance at index 1). combine_welford_partials OOS. chain_llk BLOCKED. 79 PASS. |
| `experimental/.../fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp` | PARTIAL | `da3f90e6f22` | rsqrt stage with same-CB in/out on reduce_result_cb. untestable_locally — no local pytest covers multi-chip op; structural isomorphism with rmsnorm_distributed (72388c98689). |
| `experimental/ccl/rms_allgather/.../rms_compute.cpp` | PARTIAL | `0fc4bc57602` | rsqrt stage. cb_eps Streaming here (not held). untestable_locally — multi-chip CCL op. |
| `experimental/.../dit_layernorm_post_all_gather/layernorm_post_allgather_welford.cpp` | PARTIAL | `638248c12f7` | rsqrt stage, same migration as layernorm_distributed welford counterpart. untestable_locally — multi-chip DiT path. |

## Cross-helper dependencies

When a planned migration calls for an op struct / policy / reconfig mode
not yet in the helper library, file a row here and (per HQ §"Adding
operation structs is not a blocker") add the struct before migrating:

| Need | Source kernel | Status |
|---|---|---|
| `Mask<DataFormat, Dst>` op struct (already exists in `eltwise_misc.hpp`) | clip_grad_norm step1 abs+mask prologue | available |
| `PowerIterative<>` op struct | clip_grad_norm power_tile_to_cb | available (added on reference branch, in `eltwise_misc.hpp`) |
| `DestReuseBinarySfpu` (load CB to D1, run SFPU binary against running D0) | batch_norm_sfpu_kernel, running_statistics_sfpu_kernel | **gap** — not yet added. Blocks ~5 SFPU kernels. |
| DEST-accumulating BinaryFpu (`add_tiles_init(acc_to_dest=true)`) | `*_pre_allgather_2d.cpp` merge-core | **gap** |
| BlockIterOffset with runtime offset (`block.to_global(i)`) | layernorm gamma/beta path | partial — chain has `BlockIterOffset` index mode, runtime-offset variant TBD |

## Methodology — per-file audit before commit

Before committing each PARTIAL migration:

1. `grep -nE '(add|sub|mul)_tiles|copy_tile|pack_tile|reconfig_data_format|pack_reconfig|tile_regs_(acquire|release|wait|commit)|ACQ|REL|rsqrt_tile|exp_tile|recip_tile|sqrt_tile|log_tile|tanh_tile|mask_tile|abs_tile|fill_tile' <kernel>.cpp` to enumerate every raw eltwise-shaped block remaining.
2. For each remaining block, classify:
   - **migratable now** — known chain pattern, just not done in this PR.
   - **needs helper update** — op struct / policy / index mode missing.
   - **out of scope** — welford accumulator, transpose_wh, raw reduce_tile, etc.
3. List each in the commit's `Skipped:` section AND append/update the
   per-kernel row in this file.

Anti-pattern: noting only the blocks I happened to look at while migrating
the chosen stage. The audit grep is the safety net.

## Next migration targets (this branch)

Pending in TaskList, ordered by simplicity:

1. layernorm Var+eps rsqrt — `layernorm_large_tensor.cpp`, `layernorm_large_tensor_welford.cpp` (same pattern as layernorm.cpp, in progress).
2. groupnorm Var+eps rsqrt loops — 4 kernels.
3. distributed-norm post_allgather rsqrt stages (5 kernels).
4. moreh_norm accumulators (4+ kernels).
5. rotary_embedding eltwise stages (5+ kernels).
6. softmax attention (copy+exp+pack, mul-by-recip).
7. moreh_softmax / moreh_adam / moreh_adamw / moreh_norm_backward.
8. batch_norm running_statistics + prod_all/prod_nc / reduce_*_neg.
