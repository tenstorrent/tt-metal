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
| `layernorm/.../layernorm.cpp` | PARTIAL | `5d0e13a7f96`, `7fa2452a677` | Migrated (4 chains): Var[x]+eps -> rsqrt; FUSE_PRE_ADD X+Y per-block Add; x - E[x] per-block sub_bcast_cols; (x-E[x])^2 same-CB Mul with cumulative wait + TileBaseRuntime(block.start()). **Real blocker for remaining 3 blocks**: preprocessor-macro-injected SFPU activation (SFPU_OP_INIT/FUNC_ACTIVATION) between binary op and pack inside the DEST window. No chain element wraps user-defined preprocessor macros, and the activation runs in-DEST so it can't split across the chain boundary without an intermediate CB. Resolvable by adding a kernel-local `ActivationMacro : UnaryOp` struct guarded by `#ifdef SFPU_OP_INIT_ACTIVATION`, with care to alias `slot_offset` to the macro's expected loop variable. Deferred to a focused follow-up. 52 PASS on `test_layernorm.py`. |
| `layernorm/.../layernorm_welford.cpp` | PARTIAL | `547aa2f1ec3` | Migrated: Var(x)+eps -> rsqrt prologue (1 chain, `Input` + `None` reconfig — original lacks explicit pack_reconfig). Skipped: <ul><li>cb_x prep loop (add_tiles per block + pack)</li><li>welford accumulator (lines 142-197) — OOS per HQ doc (in-DEST persistent accumulator)</li><li>cb_xmm subtraction (lines 205-220)</li><li>gamma/beta normalization (lines 266+) — block-stream with BlockIter offsets, same blocker as layernorm.cpp</li></ul> 52 PASS on `test_layernorm.py`. |
| `layernorm/.../layernorm_large_tensor.cpp` | PARTIAL | `66c0ea6590b` | Migrated: Var(X)+eps -> rsqrt prologue. Reconfig: `Input` + `Output` (explicit reconfig_data_format + pack_reconfig in original). Output lifecycle: `OutDeferredReserve` (original lacks cb_ex2pe.reserve_back, only push_back). Skipped: cb_x prep loop, cb_xmm subtraction, cb_xmm2 squaring, gamma/beta BlockIter, SFPU_OP_FUNC_ACTIVATION macro stage. 52 PASS. |
| `layernorm/.../layernorm_large_tensor_welford.cpp` | PARTIAL | `decf009afc0` | Migrated: Var(X)+eps -> rsqrt. Reconfig: `Input` + `None` (no explicit pack_reconfig; pack stays at cb_ex2 from preceding pack). Output: `OutStreaming` (cb_ex2pe.reserve_back IS called). Non-LEGACY rsqrt. Skipped: cb_x prep, welford accumulator (OOS), separate mean/var pack stages, cb_xmm subtraction, cb_ex2pe unary_bcast<COL>, gamma/beta BlockIter. 52 PASS. |

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
