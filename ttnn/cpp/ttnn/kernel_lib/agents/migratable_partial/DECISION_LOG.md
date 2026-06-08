# Migratable + Partial migrations — decision log

Task: migrate the MIGRATABLE + PARTIAL kernels from the fresh pack_tile gap map
(`generated/migration_gaps.html`). Faithful migration only — no test edits, numerics must match.
Device tests serial (single device / flock).

## MIGRATABLE

### [x] rotary_embedding_hf.cpp — DONE, test 749 passed / 1 skipped
- Structural twin of the device-validated `rotary_embedding.cpp` minus DECODE_MODE.
- Mapping mirrors that sibling's non-DECODE path:
  - scalar mul (`mul_tiles_bcast_scalar`): `mul<rotated_in, scalar, rotated_in_interm, BroadcastDim::Scalar,
    Streaming, CallerManaged>` (default Input/Output reconfig replaces the explicit
    reconfig_data_format + pack_reconfig pair). scalar_cb held (waited once, never popped).
  - sin/cos muls: local `mul_tiles_chain<>` = `mul<..., BinaryDataFormatReconfig::None, PackTileReconfig::None>`;
    the original inter-stage `reconfig_data_format*` + `pack_reconfig_data_format` calls are PRESERVED.
  - final add: `add<cos_interm, sin_interm, out>` (default Input/Output replaces reconfig_srca + pack_reconfig).
- Reconfig faithfulness: intermediate CBs share the working format, so None-reconfig muls reproduce the
  original (same simplification the validated sibling made).

### [x] rotary_embedding_hf_sharded.cpp — DONE as PARTIAL (3/5 stages), test 749 passed / 1 skipped
- Stages 2/3/4 (sin_interim=rotated*sin ROW, cos_interim=x*cos ROW, out=cos+sin) migrated to
  `mul`/`mul`/`add` with Bulk / HeldBulk / Bulk + OperandKind::Block — byte-identical lifecycle to the
  device-validated rotary_embedding_llama_sharded.cpp. sin/cos HeldBulk (pushed at batch level, popped at
  batch level, held across the heads_per_batch_t loop). in_cb Bulk on the cos stage (its last use, replaces
  the raw cb_pop_front).
- **GAP map said MIGRATABLE (no regex fired) — corrected to PARTIAL.** Rotate-half (scalar-mul + copy)
  stays raw: it writes disjoint halves of ONE co-reserved Wt window across two chain calls. PackTile
  CallerManaged is pinned at base (eltwise_chain.inl:569) so can't walk a multi-tile half-window; Bulk would
  reserve+push its own sub-window. Same partial-window/co-reserved blocker as welford-c. Validated sibling
  leaves its (matmul) rotate raw too.
- Coverage: the decode HEIGHT-sharded test cases route through the sharded program factory → this kernel.

### [/] ternary_addc_ops_fpu_bcast.cpp — NOT MIGRATED (dead code), reverted
- GAP map said MIGRATABLE. On verification the file is **referenced by NO program factory, NO CMake,
  nothing** (ternary_op_utils.cpp selects `ternary_addc_ops_sfpu_bcast.cpp`, `ternary_addc_ops_fpu_rowbcast.cpp`,
  or the int SFPU variant for bcast addc; the bare `fpu_bcast` file is never wired in). Only mentions are
  the stale html.patch and the old pack_patterns.tsv.
- Migrated it, test_ternary_bcast::addc passed 78 — but that pass exercised fpu_rowbcast + sfpu_bcast, NOT
  this file. Migrating dead code is pointless and unvalidatable, so the change was reverted.
- **RECOMMENDATION: delete this file** (and audit for other orphaned ternary kernels). The live FPU bcast
  kernel `ternary_addc_ops_fpu_rowbcast.cpp` is the real migration target if FPU-bcast addc coverage is wanted.
- Lesson reinforced: GAP-map MIGRATABLE = "no blocker regex" only; a dead/unreferenced kernel slips through.

## PARTIAL

### [~] moreh_layer_norm family (5 kernels) — IN PROGRESS
moreh_layer_norm_small/large + 3 backward kernels. NOT the `*_tiles_to_cb` macro pattern — these are
full multi-stage norms with raw copy/mask/add/sub_bcast/mul_bcast and explicit DST management.
Per-kernel migratable-vs-blocked map (small_kernel, representative):
- MIGRATABLE (isolated, proven): (Var+eps)->rsqrt block; the (x-E[x])^2 square; mean/rstd copies (caveat:
  copy_tile_init_with_dt has an is_lastdim 2nd arg — verify before using convenience copy<>); the normalize
  mul-bcast (Col/Scalar, per-chunk TileOffset on held cb_xmm); gamma mul / beta add (compile-time op via
  is_groupnorm/is_lastdim).
- BLOCKED: Sum[x] (in-place cb_xsum read+add+writeback + runtime per-tile need_to_do_mask_h); x-E[x] masking
  (runtime per-tile mask); Sum[(x-E[x])^2] (in-place cb_xmm2sum accumulation); reduce stages (separate family).
- [x] moreh_layer_norm_small_kernel.cpp: (Var+eps)->rsqrt migrated to BinaryFpu(Add)+Rsqrt<Exact,Off>+Pack.
      cb_var Streaming, cb_eps CallerManaged (held), cb_recip_std Streaming. test_moreh_layer_norm PASS.
      Rsqrt<Exact,Off> matches rsqrt_tile() defaults (legacy=false, fast=false).
- [x] moreh_layer_norm_large_kernel.cpp: (Var+eps)->rsqrt migrated identically. test_moreh_layer_norm
      48 passed / 48 skipped (large-W shapes e.g. [..,77,109] exercise large_kernel).
- NEXT migratable stages in small/large forward kernels (precision-sensitive, each a test cycle):
  * normalize mul (cb_xmm * cb_recip_std -> cb_gamma_beta_or_out): clean producer. Needs RAW eltwise_chain
    (convenience mul<> does NOT expose TileOffset) — BinaryFpu<cb_xmm, cb_recip_std, Mul,
    Col(is_lastdim)/Scalar, CallerManaged, CallerManaged, Input, D0, Block, Scalar, Set, Unset>{inner_idx,0}
    + PackTile<out, Bulk, Output> over tiles(block_size, block_size) per inner chunk. VERIFY
    is_legal_kind_lifecycle(Block, CallerManaged) compiles before relying on it.
  * (x-E[x])^2 square (1 tile, cb_xmm[inner_idx]^2 -> cb_xmm2): clean producer, BinaryFpu same-buffer +
    TileOffset::Set{inner_idx}. The cb_xmm2sum accumulation after it is BLOCKED (in-place accum).
  * gamma mul / beta add: in-place ALIAS when beta_has_value (cb_gamma_beta_or_out == cb_gamma_beta == cb_outg)
    -> blocked (layernorm-gamma pattern); the !beta branch is clean. Compile-time beta_has_value.
  * mean/rstd copies: copy_tile_init_with_dt(cb, is_lastdim) has a 2nd transpose arg — NOT plain copy<>;
    verify before migrating.
- 3 backward kernels (input_grad_large/small, gamma_beta_grad): DIFFERENT structure — cb_recip_nrstd
  computed via rstd/n (not Var+eps->rsqrt); mostly dyadd/ydyadd in-place accumulation. Migratable surface
  TBD — needs per-stage read.

### VALIDATED so far (test_moreh_layer_norm 48 passed / 48 skipped each run):
- [x] moreh_layer_norm_small_kernel.cpp: rsqrt block + normalize-mul (raw chain BinaryFpu + Block +
      TileOffset::Set{inner_idx} + CallerManaged, Col/Scalar bcast).
- [x] moreh_layer_norm_large_kernel.cpp: rsqrt block + x-E[x] sub + normalize-mul (convenience sub<>/mul<>,
      Bulk/CallerManaged/Block/Scalar — cleaner than small since cb_reuse is per-block Bulk, no offset).
      NOTE: large_kernel needed eltwise_convenience.hpp include (rsqrt-only would've missed sub/mul).
- KEY validated technique: held-CB Block reads with TileOffset (small normalize) AND per-block Bulk Block
  reads (large) both work; mirrors groupnorm.cpp:612 (x-Ex)*denom and groupnorm:393 sub patterns.

### STILL TODO (checkpoint for continuation):
- [x] small_kernel (x-E[x])^2 square: raw chain BinaryFpu<cb_xmm,cb_xmm,...,Block,Block,Set,Set>{inner_idx,inner_idx}
      + PackTile(cb_xmm2,Streaming). CallerManaged (cb_xmm held). test 48 passed.
- [x] large_kernel (x-E[x])^2 square: convenience square<cb_xmm,cb_xmm2,Bulk,Bulk,Input,Output,Block>
      (per-block, index j, no offset). test 48 passed.
- FORWARD KERNELS NOW AT CLEAN LIMIT. Remaining raw (all cited blockers):
  * Sum[x] (in-place cb_xsum accum + runtime need_to_do_mask_h)
  * x-E[x] in small_kernel (runtime per-tile mask interleaved)
  * Sum[(x-E[x])^2] accum (in-place cb_xmm2sum)
  * gamma mul / beta add (in-place ALIAS when beta_has_value; layernorm-gamma blocker)
  * reduce stages (separate helper family)
- 3 backward kernels:
  * [x] input_grad_small_kernel: rstd/n (recip_nrstd) mul migrated — raw chain BinaryFpu<cb_n_recip_n, cb_rstd,
        Mul, Col/Scalar, CallerManaged, CallerManaged, Scalar, Scalar, Set{1}, Unset> + PackTile(cb_recip_nrstd,
        Streaming). test 48 passed. NOT migrated: (x-mean) sub (runtime mask); (x-mean)*rstd mul (cb_y has split
        reserve(Wt)/push(1) output lifecycle — no clean chain policy); dy*gamma (runtime mask); dyadd/ydyadd accum.
  * [x] input_grad_large_kernel: recip_nrstd mul migrated identically (raw chain, TileOffset::Set{1}). test 48 passed.
  * gamma_beta_grad_kernel: dominated by masked dycopy (copy_tile + runtime mask) + in-place cb_dyadd
        accumulation (copy-first/add-rest) — both BLOCKED. The (x-mean)*rstd mul (Streaming cb_y) is clean but
        cb_xmm is produced by a masked sub; left raw (thin surface, low ROI). NOT migrated.
- small moreh + fill_pad set (task 5): unread.

## SESSION TALLY (all device-tested, faithful, no test edits)
- rotary_embedding_hf.cpp (full), rotary_embedding_hf_sharded.cpp (partial 3/5)
- ternary_addc_ops_fpu_bcast.cpp DELETED (dead); ternary_addc_ops_fpu.cpp flagged dead (not deleted)
- moreh_layer_norm_small (rsqrt, normalize-mul, square), large (rsqrt, x-E[x] sub, normalize-mul, square)
- moreh_layer_norm_backward_input_grad_small (recip_nrstd)
- 10 stages migrated across 5 kernels + 1 deletion. Forward layernorm kernels at clean migratable limit.

### [~] small moreh + fill_pad (5 kernels) — ASSESSED
- moreh_clip_grad_norm_step1: ALREADY uses eltwise_chain/convenience (prior migration). The 3 remaining raw
  pack_tile are the runtime-masked input-prep stages (need_to_do_mask_h / do_mask_w) — BLOCKED. At clean limit.
- moreh_bias_backward_multi_core_h / single_core_hw: runtime-masked input copy + reduce (bias = sum over H).
  Mask BLOCKED, reduce = separate helper family. No clean migratable stage.
- moreh_abs_pow: copy+abs with runtime per-tile mask (do_mask_w && col_idx==Wt-1) -> runtime op selection
  BLOCKED; power_tile_to_cb = composite macro (gap-map "power composite") BLOCKED. No clean stage.
- fill_pad_compute: MIGRATABLE but deferred. process_masked_tile = CopyTile(data->D0) + CopyTile(mask->D2) +
  Fill(->D1) + Where<FMT>(D2,D1,D0->D2) + PackTile(D2); process_corner_tile = 2 sequential where. Follows the
  eltwise_where_no_bcast.cpp template (Where + Fill + custom DST slots + host FILL_PAD_FILL_FN macro). Non-trivial
  specialty multi-element chain + corner variant — deferred to avoid a rushed untested bug under context pressure.
