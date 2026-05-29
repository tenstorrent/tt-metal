# Eltwise-chain migration status (cross-op)

Live ledger of every TTNN compute kernel and its relationship to
`compute_kernel_lib::eltwise_chain` on branch `astancov/eltwise_vx_migration`.

**This file was regenerated from scratch on 2026-05-26** by cross-referencing:

1. `grep -r 'compute_kernel_lib::eltwise_chain'` across `ttnn/cpp/ttnn/operations/`
   and `ttnn/cpp/ttnn/kernel/` &mdash; finds every file actually wired to the
   chain helper today.
2. `generated/pack_tile_survey.tsv` (730 raw `pack_tile` callsites) &mdash;
   finds every file still emitting raw `pack_tile` / `pack_tile_with_dt`.
3. Per-kernel `grep -E '^[[:space:]]*pack_tile(_with_dt)?\('` &mdash; counts
   the real raw packs (excluding comments) to distinguish FULL from PARTIAL.
4. Per-kernel `grep` for blocker primitives (`matmul_tiles`,
   `transpose_wh_tile`, `reduce_tile`, `welford_*`, `reshuffle_rows_tile`,
   `rand_tile`) &mdash; classifies WHY a PARTIAL or UNTOUCHED kernel can't
   migrate cleanly. Note: `welford_*` is **not OOS** &mdash; the welford
   running mean/M2 live in SFPU LREG4/LREG5 (outside DEST), so each LLK
   primitive is wrappable as a chain element struct. Tagged as &sect;5
   helper gap, not OOS &sect;4.

A companion artifact `docs/eltwise_chain_migration_candidates.html` has a
per-kernel chain-shape proposal for the unmigrated queue; this file is the
state ledger.

## Live counts (2026-05-26)

| Category | Count | Definition |
|---|---:|---|
| FULL | 51 | File includes the chain header AND has zero remaining raw `pack_tile`. (+3 since 2026-05-26: batch_norm_sfpu, running_statistics_sfpu, running_statistics_kernel FPU). |
| PARTIAL | 23 | File includes the chain header AND still has at least one raw `pack_tile` &mdash; each row below names the specific stage(s) left raw with the root-cause #. (-1 since 2026-05-26: running_statistics_kernel FPU promoted to FULL). |
| UNTOUCHED | 137 | File still has raw `pack_tile` and does NOT include the chain header. Classified into MIGRATE NOW / blocked-by-helper-gap / OOS below. |

**2026-05-28 update:** `running_statistics_kernel` (FPU) promoted PARTIAL &rarr; FULL (commit `7fd2eedd157`); section 3 entries below for `moreh_int_sum_*`, `moreh_sum_nc`, the CCL accumulator cluster, and `moreh_nll_loss_step2` were re-audited &mdash; their classification has been corrected (see &sect;3a-corrections below).

## Reconfig audit policy

For each migration, the `BinaryDataFormatReconfig` / `PackTileReconfig` /
`CopyTileReconfig` flags are set to match the **original LLK calls in this
kernel**, not what the reference branch `astancov/eltwise_run7_refined_rebase_v2`
did. See
`~/.claude/projects/-localdev-astancov-tt-metal/memory/feedback_eltwise_reconfig_audit.md`.

---

## 1. FULL kernels (48)

### moreh

| Kernel | First chain commit | Notes |
|---|---|---|
| `moreh_dot/.../moreh_dot.cpp` | `52c461763c5` | One chain: BinaryFpu(Mul) + PackTile. |
| `moreh_dot_backward/.../moreh_dot_backward.cpp` | `5cb08a20f15` | Two chains, BinaryFpu(Mul, Scalar) + PackTile. |
| `moreh_clip_grad_norm_step2/.../moreh_clip_grad_norm_step2_kernel.cpp` | `b6a3c70773e` + `51cffeb6f03` | Seed+accumulator AND power_tile_to_cb composite migrated. |
| `moreh_adam/.../moreh_adam.cpp` | `d0dae745ea1` + `6142249dc57` + `d7edb1924ec` | All in-DEST stages, pow(beta, step), sqrt(exp_avg_sq * tmp1) migrated. |
| `moreh_adamw/.../moreh_adamw.cpp` | `d07becaaf42` + `6142249dc57` + `5c333a73630` | Same coverage. |
| `moreh_norm/{moreh_norm_h,moreh_norm_w,moreh_norm_other}_kernel.cpp` | `7743f35794f`, `51e435506bd`, `ff98c25acdf` | \|x\| with mask + power + accumulator + final \|x\|^(1/p). |
| `moreh_norm/ord_other/{h,w,nc}_kernel.cpp` | `7e61967482a` | IS_ZERO / MINUS_INF / default variants. |
| `moreh_norm_backward/.../moreh_norm_backward_kernel.cpp` | `01156fad878` | sign + \|x\|^(p-1) + 3 Mul stages with 4-branch bcast. |
| `moreh_mean/moreh_mean_h.cpp` | `5f4a6e59c73` | Mask_h stage + accumulator. |
| `moreh_mean/moreh_mean_nc.cpp` | `e14b8945adb` | Accumulator + final scalar-mul. |
| `moreh_mean_backward/.../moreh_mean_backward.cpp` | `7bb2652ae28` | Add-with-bcast + final scalar-mul. |
| `moreh_sum_backward/.../moreh_sum_backward.cpp` | `9b5247ea665` | Add-with-bcast stage. |
| `moreh_sum/.../moreh_sum_h.cpp` | `b4650549f85` | Mask_h stage. The matmul-based reduce path stays raw (OOS #8). |
| `moreh_softmax/moreh_softmax_c_large.cpp` | `c2cbd33e580` + `b83db8b05c0` | Max-accumulator + sub+exp + log/recip + final. No bcast. |
| `moreh_softmax/moreh_softmax_h.cpp` | `b5c5a694622` + `589d569fb4c` + `77ec97d932b` | All inner-loop stages migrated. |
| `moreh_softmax/moreh_softmax_h_large.cpp` | `2ca4acfe930` + `b83db8b05c0` | Per-tile sub+exp(+mask) + accumulator + final. |
| `moreh_softmax_backward/moreh_softmax_backward_{c,h,w}_large.cpp` | `b83db8b05c0`, `06a000ab302`, `b83db8b05c0` | LOG / non-LOG variants. |

### normalization

| Kernel | First chain commit | Notes |
|---|---|---|
| `batch_norm/.../batch_norm_kernel.cpp` | `17056fe08cb` + `6dd9dbcdece` | 1-tile BinaryFpu+Rsqrt prologue + fused Stage 2-4 via DestReuseBinary + 4-branch chain collapsed via OptionalChainElement. |
| `batch_norm/.../batch_norm_sfpu_kernel.cpp` | `c0507770694` + `50d82ae55a9` | Held-tile wait/pop absorbed into chain (Bulk+Scalar). |
| `batch_norm/.../running_statistics_kernel.cpp` (FPU) | `dd4d2b0bfdc` + `7fd2eedd157` | Stages 1-3 via `fpu_binary_to_cb_chain` wrapper; stage 4 = explicit chain `BinaryFpu<Add> + 2x PackTile` (primary Streaming to `cb_updated_running_*`, mirror OutCallerManaged to `cb_out0`). Constexpr-if `mean_packs_to_out0` for the conditional mirror pack. |
| `batch_norm/.../running_statistics_sfpu_kernel.cpp` | `b95524db659` | Fused 4-stage chain via `CopyTile + SubBinary/MulBinary/AddBinary`, 3 DEST slots, OptionalChainElement collapse. |
| `layernorm_distributed/.../layernorm_pre_allgather.cpp` | `189ef03f223` | Squaring chain over `EltwiseShape::of(Wt/blk, blk)`. |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | `c41aca20b88` | Same squaring chain. |
| `rmsnorm_distributed/.../rmsnorm_post_allgather.cpp` | `72388c98689` | 4-stage chain (add+rsqrt + x*recip + [*gamma] + [+beta]). |
| `layernorm_distributed/.../layernorm_post_allgather_welford.cpp` | `9ce11f1cd76` | Rsqrt stage; combine_welford partials covered by OOS-via-helpers boundary. |

### ttnn core eltwise / data_movement / experimental

| Kernel | First chain commit | Notes |
|---|---|---|
| `copy/typecast/.../eltwise_typecast.cpp` | `6589d8167d8` | Single CopyTile + Typecast<InDF,OutDF> + PackTile. |
| `eltwise/unary/.../eltwise_identity_kernel.cpp` | `25e03c15161` | Per-tile copy_tile -> pack_tile. |
| `eltwise/unary/.../hardswish_kernel.cpp` | `e0cb3c1cada` | x * hardsigmoid(x). |
| `eltwise/unary/.../logit_kernel.cpp` | `ef406b5ca83` | log(x/(1-x)). |
| `eltwise/unary/.../logsigmoid_kernel.cpp` | `ef60346171a` | CopyTile + CopyTile + Negative + Exp + Logsigmoid + PackTile. |
| `eltwise/unary/.../mish_kernel.cpp` | `ef406b5ca83` | x * tanh(softplus(x)). |
| `eltwise/unary/.../tanhshrink_kernel.cpp` | `27bcccc7fea` | x - tanh(x). |
| `eltwise/unary_backward/gelu_bw/.../eltwise_bw_gelu_poly.cpp` | `ef60346171a` | grad * GELU'(x). |
| `eltwise/unary_backward/tanh_bw/.../eltwise_bw_tanh_deriv.cpp` | `ef60346171a` | grad * sech²(x). |
| `experimental/unary_backward/gelu_backward/.../eltwise_bw_gelu_poly.cpp` | `ef60346171a` | Sibling of the gelu_bw kernel (separate copy). |
| `data_movement/clone/.../compute_kernel.cpp` | `96c844c6817` | Single CopyTile+PackTile. |
| `data_movement/bcast/.../bcast_{h,w,hw}.cpp` | `65b34f7740f` &rarr; 2D upgrade `48fa98659a9` &rarr; reverted to 1D + init_bcast `9dcb85feb59` | Per-tile BinaryFpu<CHAIN_BCAST_OP, dim> + PackTile. Driven by `bcast_op_utils::get_defines`. Original `init_bcast<BCAST_LLKOP, BCAST_DIM>` retained as the kernel-level big init (chain's BinaryFpu uses `CHAIN_BCAST_OP` / `CHAIN_BCAST_DIM` helper-lib types emitted alongside the LLK enum macros). `bcast_h` / `bcast_hw` are flat 1D chains over `B*Ht*Wt` total tiles (Streaming + Scalar). `bcast_w` keeps a flat outer `for (row 0..B*Ht)` with per-row HeldStream cb_rhs + explicit `cb_rhs_obj.pop_front(1)` at chain end (1 scalar tile/row). The 2D `EltwiseShape::of` upgrade was reverted &mdash; bcast is tile-by-tile, no benefit from a 2D shape. |
| `data_movement/sharded/.../eltwise_copy.cpp` | `3dfd9bb6b07` | Per-tile copy, runtime tile count. |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `3dfd9bb6b07` | Shared compile-time variant. |
| `experimental/bcast_to/.../compute_interleaved_{col,row,scalar}_bcast_to.cpp` | `a1fd70d3e88` | UnaryBcast<{Col,Row,Scalar}> + PackTile. |
| `experimental/dropout/.../dropout_kernel.cpp` | `6cf9b8a4d11` | dropout_kernel_init(seed) + Dropout + PackTile. |
| `reduction/prod/.../prod_all.cpp` | `96c844c6817` | Seed + accumulator + final copy. |
| `reduction/prod/.../prod_nc.cpp` | `96c844c6817` | Per output-tile pattern. |

---

## 2. PARTIAL kernels (24)

Each row gives the **real** count of raw `pack_tile(_with_dt)` calls remaining
(grep against the live file, comments excluded) and the specific stages they
belong to. Every "remaining stage" maps to a root cause in &sect;4 or a
helper gap in &sect;5.

### rotary_embedding family (6 PARTIAL)

| Kernel | Raw packs | First chain commit | Remaining raw stages (root-cause #) |
|---|---:|---|---|
| `experimental/rotary_embedding/rotary_embedding.cpp` | 2 | `54d28d9b172` | MUL_TILES runtime-parameterized helper. **Root cause:** OOS #7 runtime CB ids. |
| `experimental/rotary_embedding/rotary_embedding_single_tile.cpp` | 1 | `c941a8bd3ce` | Matmul-based rotate_half. **OOS #8.** |
| `experimental/rotary_embedding_llama/rotary_embedding_llama.cpp` | 1 | `e4fe7ef5795` | Matmul rotate_half. **OOS #8.** |
| `experimental/rotary_embedding_llama/rotary_embedding_llama_sharded.cpp` | 1 | `e1134ae5e31` | Matmul rotate_half. **OOS #8.** |
| `experimental/rotary_embedding_llama_fused_qk/.../*_sharded.cpp` | 3 | `b0166a5e82a` | cos stage + add stage. **Root cause:** runtime in_cb/out_cb (q vs k via `is_q` arg) + TRISC2 size budget rules out duplication. **OOS #7.** |
| `experimental/rotary_embedding_llama_fused_qk/.../*_row_major.cpp` | 3 | `d2ba9f29d82` | Same as sibling sharded. **OOS #7.** |

### normalization (10 PARTIAL)

| Kernel | Raw packs | First chain commit | Remaining raw stages (root-cause #) |
|---|---:|---|---|
| `groupnorm/groupnorm.cpp` | 7 | `3e5f209fa06` + `dcc18175bef` | Multi-stage block-strided normalization (x prep + (x-Ex)*rsqrt + gamma + beta) where output CB equals input CB on optional gamma/beta. **Helper gap:** chain reserves output BEFORE popping input, so same-CB in-place inside chain isn't safe. |
| `groupnorm/groupnorm_sharded_v2.cpp` | 9 | `abe68ea4c28` + `06b2806c554` | Same as groupnorm.cpp + `pack_reconfig_l1_acc(1)` for the `(x-E[x])^2` block. **OOS #6** L1 packer acc + same-CB-in-place gap. |
| `groupnorm/welford_groupnorm.cpp` | 4 | `92a19132dd7` + `20ce6da381c` + `a4115dfc276` | Welford accumulator inner loop. **GAP welford structs (§5)** + same-CB-in-place gap. |
| `groupnorm/welford_groupnorm_sharded_v2.cpp` | 3 | `66907980ee9` + `20ce6da381c` | Welford accumulator + sharded variant. **GAP welford structs (§5).** |
| `layernorm/layernorm.cpp` | 3 | `5d0e13a7f96` + `7fa2452a677` | Macro-injected SFPU activation (`SFPU_OP_INIT_ACTIVATION` / `SFPU_OP_CHAIN_0`) inside DEST window. **OOS #4** &mdash; resolvable by adding a kernel-local `ActivationMacro : UnaryOp` struct guarded by `#ifdef`. |
| `layernorm/layernorm_welford.cpp` | 4 (+4 blocking) | `547aa2f1ec3` + `f53bc329ce1` | Welford accumulator across blocks + `transpose_wh_tile` interleaved with `welford_update` / `welford_finalize_to_row`. **GAP welford structs (§5) + #2.** |
| `layernorm/layernorm_large_tensor.cpp` | 7 | `66c0ea6590b` + `0b0ca49146e` | Block A (pass-1 variance with interleaved sub_bcast + binary_dest_reuse + square + reduce_init + reduce_tile + scalar mul in one DEST window) + blocks D/E/F SFPU activation macro. **OOS #3 + #4.** |
| `layernorm/layernorm_large_tensor_welford.cpp` | 13 (+5 blocking) | `decf009afc0` + `3016a1d484f` | Welford + transpose + FUSE_PRE_ADD + pass-2 sub_bcast + gamma/beta. **GAP welford structs (§5), #2, #3.** |
| `layernorm_distributed/layernorm_post_allgather.cpp` | 2 | `6277004b227` | `chain_llk` DSL graph (normed_output / gamma_optional / beta_optional nodes). **OOS #5** &mdash; would need re-implementing the chain_llk graph layer. |
| `experimental/transformer/dit_layernorm_post_all_gather/.../layernorm_post_allgather_welford.cpp` | 2 | `638248c12f7` + `0d52a5ae5a0` | mul-by-recip_sqrt_var and mul-by-gamma blocks. **Same-CB-in-place gap** (output CB equals input CB when gamma\|\|beta is enabled). |
| `experimental/fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp` | 6 (+1 blocking) | `da3f90e6f22` | reduce_tile inside DEST + chain_llk-like pattern. **OOS #3 + #5.** |
| `experimental/ccl/rms_allgather/rms_compute.cpp` | 5 | `0fc4bc57602` | Multi-stage same-CB-in-place blocks. **Same-CB-in-place gap.** Untestable locally (multi-chip CCL op). |

### softmax attention (2 PARTIAL)

| Kernel | Raw packs | First chain commit | Remaining raw stages (root-cause #) |
|---|---:|---|---|
| `normalization/softmax/.../attention/softmax.cpp` | 2 | `d0b0d9a5f78` + `8bad3164a7b` + `1e51b2b91be` | Cumulative-wait pattern on cb_fused_attn for NUMERIC_STABLE path. **OOS #9** + runtime CB ids in some variants (**OOS #7**). |
| `normalization/softmax/.../attention/softmax_sharded.cpp` | 3 | `b1637a6cca5` | Same as softmax.cpp. **OOS #9.** |

### moreh (4 PARTIAL)

| Kernel | Raw packs | First chain commit | Remaining raw stages (root-cause #) |
|---|---:|---|---|
| `moreh_clip_grad_norm_step1/.../moreh_clip_grad_norm_step1_kernel.cpp` | 0 (commit on disk shows 0 raw, see note) | `d47573e2270` + `2dfcf1aeccf` + `61bb05922bf` | Listed as PARTIAL historically but a fresh grep finds 0 raw `pack_tile(_with_dt)` &mdash; consider promoting to FULL after a regression run. |
| `moreh_mean/moreh_mean_w.cpp` | 2 (+2 blocking matmul) | `b0afa1a96f2` | Matmul-based REDUCE_ROW path (tile-matmul + pack). **OOS #8** (tile-matmul anti-pattern; only block matmul helpers allowed per branch memory). |
| `moreh_sum/.../moreh_sum_w.cpp` | 2 (+2 blocking matmul) | `37066e7298b` | Same matmul REDUCE_ROW path as moreh_mean_w. **OOS #8.** |
| `moreh_softmax/moreh_softmax_w.cpp` | 2 | `ffe87ea3eb6` + `2cb2bb8f7a3` | Mask + reduce_init + reduce_tile inside DEST window. **OOS #3.** |
| `moreh_softmax/moreh_softmax_w_large.cpp` | 2 | `f0f7da92ea1` + `b83db8b05c0` | Same as moreh_softmax_w. **OOS #3.** |

---

## 3. UNTOUCHED kernels (137)

These have raw `pack_tile` and do NOT include the chain header. Classified by
verdict + root cause.

### 3a. MIGRATE NOW (no documented blocker, helpers exist)

These have a known chain shape. Each has a reference example from the FULL
list above. The companion file
`docs/eltwise_chain_migration_candidates.html` &sect;4 has the per-kernel
chain shape proposal.

**SFPU-binary cluster** (uses `add_binary_tile` / `sub_binary_tile` /
`mul_binary_tile` on two fresh DEST slots &mdash; chain `AddBinary<D0,D1,D0>`
in `eltwise_binary_sfpu.hpp` already expresses this; **no helper gap**):

- `eltwise/binary/.../eltwise_binary_sfpu_kernel.cpp` (audit for macro-injected op first)
- ~~`normalization/batch_norm/.../batch_norm_sfpu_kernel.cpp`~~ &mdash; **MIGRATED** (`c0507770694` + `50d82ae55a9`).
- ~~`normalization/batch_norm/.../running_statistics_sfpu_kernel.cpp`~~ &mdash; **MIGRATED** (`b95524db659`).
- ~~`normalization/batch_norm/.../running_statistics_kernel.cpp`~~ &mdash; **MIGRATED FULL** (`dd4d2b0bfdc` PARTIAL &rarr; `7fd2eedd157` FULL). FPU + DestReuseBinary path, not the SFPU-binary shape.
- ~~`moreh/moreh_sum/.../moreh_int_sum_{h,w,nc}.cpp`~~ &mdash; **MOVED to &sect;3c-int** (audit 2026-05-28): these use `sfpu_add_int`, `sfpu_sum_int_col/row`, `mask_tile(Int32)` &mdash; int-specific SFPU primitives NOT in the helper library. Original classification was incorrect; these need int-typed chain helpers before migration.

**Moreh recip + mul cluster** (CopyTile + Recip + Pack, BinaryFpu + Mul-by-bcast-scalar):

- `moreh/moreh_nll_loss_backward/.../moreh_nll_loss_backward_kernel.cpp`
- `moreh/moreh_linear_backward/.../moreh_bias_backward_single_core_hw.cpp` &mdash; clean migratable, but the `do_mask_h/do_mask_w && last_row/col` Mask block is RUNTIME-conditional; needs either runtime chain dispatch or 4-way constexpr split keyed on `do_mask_*`.
- `moreh/moreh_linear_backward/.../moreh_bias_backward_multi_core_h.cpp` &mdash; same as single_core_hw.
- ~~`moreh/moreh_clip_grad_norm_step3/.../moreh_clip_grad_norm_step3_kernel.cpp`~~ &mdash; **MIGRATED** (`98c5ebb783b` + `7ec7613c5d0`).
- `moreh/moreh_abs_pow/.../moreh_abs_pow_kernel.cpp` &mdash; abs+mask prologue
  (`Mask` op struct in `eltwise_misc.hpp`) + power composite (precedent:
  `moreh_norm_h_kernel.cpp` FULL). Has runtime-conditional mask block (same shape gotcha as bias_backward).
- ~~`moreh/moreh_sum/.../moreh_sum_nc.cpp`~~ &mdash; **MOVED to &sect;3c** (audit 2026-05-28): uses `add_tiles_init(cb_in0, cb_in1, acc_to_dest=true)`. DEST-accumulating BinaryFpu is the same helper gap that already blocks `layernorm_pre_allgather_2d` (&sect;5).
- `moreh/moreh_nll_loss/.../moreh_nll_loss_step2_kernel.cpp` &mdash; **ATTEMPTED twice, reverted both times**. The current investigation (2026-05-28) also fails on the chain-to-raw state-leak issue documented in `docs/migration_log_panels/moreh_nll_step2.html`. Test baseline already has 13 unrelated failures, masking the migration signal. Needs a dedicated debug session OR full migration of every stage (no chain-to-raw boundary).

**CCL / reduction accumulator cluster** &mdash; **MOVED to &sect;3c** (audit 2026-05-28):
Most kernels in this group use `add_tiles_init(..., acc_to_dest=true)` to fold N
device tiles into the same DEST slot. Same DEST-accumulating-BinaryFpu gap as
&sect;5. Confirmed on `llama_reduce_scatter_create_heads/.../reduction.cpp` (line 22:
`add_tiles_init(fabric_receiver_cb_id, fabric_receiver_cb_id, true)`).

Kernels to re-audit if the helper gap closes:
- `experimental/ccl/reduce_scatter_minimal_async/.../{dim_zero_line_reduction, dim_zero_ring_reduction, line_reduction, ring_reduction}.cpp`
- `experimental/ccl/{all_reduce_async, deepseek_moe_reduce_scatter, llama_reduce_scatter, llama_reduce_scatter_create_heads, moe_compute, moe_gpt}/.../*.cpp`
- `experimental/reduction/{fast_reduce_nc, deepseek_moe_fast_reduce_nc}/.../*.cpp`
- `experimental/transformer/all_reduce_create_qkv_heads/.../reduction.cpp`
- `normalization/kernel_util/compute/pre_add.h`

**Rotary HF family** (3-binary-chain shape from `rotary_embedding_llama_sharded.cpp` FULL):

- `experimental/transformer/rotary_embedding_hf/.../rotary_embedding_hf.cpp`
- `experimental/transformer/rotary_embedding_hf/.../rotary_embedding_hf_sharded.cpp`

**Ternary FPU family** (BinaryFpu(Mul) + ScalarMul + DestReuseBinary(Add)):

- `eltwise/ternary/.../ternary_addc_ops_fpu.cpp`
- `eltwise/ternary/.../ternary_addc_ops_fpu_bcast.cpp`
- `eltwise/ternary/.../ternary_addc_ops_fpu_rowbcast.cpp`

**Other**:

- `experimental/ssm/prefix_scan/.../ssm_prefix_scan.cpp` &mdash; 2 eltwise_binary + 1 copy.
- `experimental/reduction/integral_image/.../intimg_compute.cpp` &mdash; `Cumsum<Slot>` chain element exists; the `transpose_wh_tile` block stays OOS #2.

### 3b. PARTIAL-eligible (UNTOUCHED but has a clean stage + a blocking stage)

These have a clean migratable stage co-existing with an OOS-tagged stage in the same kernel. Migration here is intentionally partial.

- `experimental/ssm/repeat_and_interleave_eltwise_mul/.../ssm_eltwise_mul.cpp` (7 packs) &mdash; the `mul_tiles` + `mul_bcast_rows` blocks migrate; 4 transpose blocks stay raw (**OOS #2**).
- `reduction/accumulation/.../accumulation_compute.cpp` (3 packs) &mdash; seed pack + per-iter `DestReuseBinary<Add, DEST_TO_SRCA>` accumulator migrate; if `pack_reconfig_l1_acc` is used in the fp32 path that part stays raw (**OOS #6**).
- `reduction/generic/.../reduce_{h,w,hw}_neg.cpp` (3 packs each) &mdash; pre-reduce `CopyTile + Negative + PackTile` and post-reduce `CopyTile + Negative + (optional MulScalar) + PackTile` migrate; the central `reduce_init + reduce_tile` stays raw (**OOS #3** &mdash; use `reduce_helpers` for that stage).
- `moreh/moreh_layer_norm/.../moreh_layer_norm_{small,large}_kernel.cpp` (13 / 15 packs) &mdash; 5+ chains migrate (Var+eps&rarr;rsqrt, x-E[x] sub_bcast, *recip, gamma, beta); reduce-interleaved stages stay raw (**OOS #3**).
- `moreh/moreh_layer_norm_backward/.../moreh_layer_norm_backward_{input_grad_large,input_grad_small,gamma_beta_grad}_kernel.cpp` (19 / 15 / 10 packs) &mdash; rsqrt seed, dy*gamma bcast mul, sub_bcast, gradient accumulators migrate; welford / reduce-interleaved stages stay raw (**GAP welford structs (§5), #3**).
- `normalization/layernorm/.../layernorm_sharded.cpp` (9 packs) &mdash; same shape as `layernorm.cpp` PARTIAL; macro-injected SFPU activation stays raw (**OOS #4**).
- `normalization/layernorm/.../layernorm_sharded_post_allgather.cpp` (10 packs) &mdash; mirrors `layernorm_post_allgather.cpp` PARTIAL; chain_llk gamma/beta wiring stays raw (**OOS #5**).
- `normalization/layernorm/.../layernorm_sharded_pre_allgather.cpp` (4 packs) &mdash; squaring chain migrates; reduce tail stays raw (**OOS #3**).
- `normalization/layernorm/.../layernorm_sharded_welford.cpp` (8 packs) &mdash; same shape as `layernorm_welford.cpp` PARTIAL; welford + transpose stay raw (**GAP welford structs (§5), #2**).
- `normalization/layernorm_distributed/.../layernorm_pre_allgather_welford.cpp` (11 packs) &mdash; squaring + accumulator stages migrate; welford finalize + transpose stay raw (**GAP welford structs (§5), #2**).
- `experimental/transformer/dit_layernorm_pre_all_gather/.../layernorm_pre_allgather_welford.cpp` (4 packs) &mdash; squaring stage migrates; welford accumulator stays raw (**GAP welford structs (§5)**).
- `normalization/softmax/.../attention/compute/softmax_large_tensor.cpp` (6 packs) &mdash; plain `else` branch (`CopyTile + Exp + PackTile`) migrates; NUMERIC_STABLE / FUSED_SCALE_MASK stay raw (**OOS #9** + **OOS #7**).
- `normalization/kernel_util/compute/combine_welford.h` (2 packs) &mdash; welford finalize semantics (**GAP welford structs (§5)**).

### 3c. BLOCKED on a small helper extension

Each needs a specific helper struct / mode to be added before migration is clean.

| Kernel | Blocker | Unlock |
|---|---|---|
| `normalization/layernorm_distributed/.../layernorm_pre_allgather_2d.cpp` | DEST-accumulating BinaryFpu (`add_tiles_init(acc_to_dest=true)`) | Extend `binary_op_helpers.hpp` |
| `normalization/rmsnorm_distributed/.../rmsnorm_pre_allgather_2d.cpp` | Same DEST-acc BinaryFpu | Same |
| `normalization/layernorm_distributed/.../chain_llk.hpp` | OOS #5 (`chain_llk` DSL graph) | Substantial: re-implement chain_llk on top of eltwise_chain |
| `normalization/kernel_util/compute/numeric.h` | Header used by chain_llk &rarr; OOS #5 | Same |
| `eltwise/binary/.../eltwise_binary_kernel.cpp` | Audit needed: if macro-injected activation present &rarr; **OOS #4**, otherwise migrate-now | Verify before touching |
| `moreh/moreh_sum/.../moreh_sum_nc.cpp` | DEST-accumulating BinaryFpu (`acc_to_dest=true`) | Same as layernorm 2d above (re-audit 2026-05-28). |
| CCL accumulator cluster (~10 kernels listed in &sect;3a-corrected) | DEST-accumulating BinaryFpu (`acc_to_dest=true`) | Same. |

### 3c-int. BLOCKED on int-typed SFPU chain helpers

These three kernels use int-specific SFPU primitives (`sfpu_add_int`,
`sfpu_sum_int_col/row`, `mask_tile(DataFormat::Int32)`) that are NOT in the
current chain helper library. Originally listed in &sect;3a as "SFPU-binary
cluster" but the audit (2026-05-28) showed the underlying LLK ops are int
variants, not the generic `add_binary_tile` family.

| Kernel | Missing primitives | Unlock |
|---|---|---|
| `moreh/moreh_sum/.../moreh_int_sum_h.cpp` | `sfpu_add_int`, `sfpu_sum_int_col`, `mask_tile(Int32)` | Add `AddIntBinary` / `SumIntCol` / `MaskInt` op structs OR extend existing structs to dispatch on a `DataFormat` template param. |
| `moreh/moreh_sum/.../moreh_int_sum_w.cpp` | `sfpu_add_int`, `sfpu_sum_int_row`, `mask_tile(Int32)` | Same. |
| `moreh/moreh_sum/.../moreh_int_sum_nc.cpp` | `sfpu_add_int` | Smallest gap of the three &mdash; only needs the int binary. |

### 3d. BLOCKED on macro-injected SFPU / activation (**OOS #4**)

22 kernels use preprocessor macros (`BINARY_OP`, `PROCESS_POST_ACTIVATIONS`,
`SFPU_OP_INIT_ACTIVATION`, `BINARY_SFPU_OP`, `TERNARY_SFPU_OP`) that expand to
arbitrary SFPU calls at compile time. Chain has no element wrapping a
user-defined preprocessor macro.

**High-leverage unlock:** refactor `binary_op_utils::get_defines` to emit
chain-element identifiers (as `bcast_op_utils::get_defines` does for
`CHAIN_BCAST_OP` / `CHAIN_BCAST_DIM` in commit `65b34f7740f`). One refactor
unlocks the entire family below.

`eltwise/binary_ng/device/kernels/compute/`:
- `eltwise_binary.cpp` &nbsp; `eltwise_binary_no_bcast.cpp` &nbsp; `eltwise_binary_scalar.cpp`
- `eltwise_binary_sfpu.cpp` &nbsp; `eltwise_binary_sfpu_no_bcast.cpp` &nbsp; `eltwise_binary_sfpu_scalar.cpp`
- `eltwise_where_no_bcast.cpp` &nbsp; `eltwise_where_sfpu.cpp` &nbsp; `eltwise_where_sfpu_scalar.cpp`
- `eltwise_utils.hpp` &nbsp; `eltwise_utils_sfpu.hpp` (shared headers)

`eltwise/binary_ng/device/kernels_ng/compute/`:
- `eltwise_binary_{col,row,row_col,scalar}_bcast.cpp` (4 kernels)
- `eltwise_binary_sfpu_{col,row,row_col,scalar}_bcast.cpp` (4 kernels)
- `eltwise_where_sfpu_{row,row_col}_bcast.cpp` (2 kernels)

`eltwise/unary/device/kernels/compute/`:
- `eltwise_sfpu.cpp` (SFPU_OP_CHAIN_0)
- `lgamma_kernel.cpp` &nbsp; `lgamma_fast_kernel.cpp`
- `where_tss_kernel.cpp`

`eltwise/ternary/device/kernels/compute/` (SFPU variants):
- `ternary_addc_ops_sfpu.cpp` &nbsp; `ternary_addc_ops_sfpu_bcast.cpp`
- `ternary_addcmul_int_sfpu.cpp` &nbsp; `ternary_addcmul_int_sfpu_bcast.cpp`
- `ternary_sfpu_no_bcast_ttt.cpp` &nbsp; `ternary_sfpu_no_bcast_tts_tst.cpp`
- `ternary_sfpu_col_scalar_bcast_ttt.cpp` &nbsp; `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`
- `ternary_sfpu_row_bcast_ttt.cpp`

Other macro-injected:
- `examples/example/device/kernels/compute/eltwise_sfpu.cpp`
- `experimental/unary_backward/gelu_backward/.../eltwise_bw_gelu_approx_tanh.cpp` (also uses `copy_dest_values` &mdash; DEST-to-DEST copy, no chain element)

### 3e. OOS &mdash; out of scope for eltwise-chain

These are blocked by patterns chain is structurally not designed to express.

**Transpose-only / sort / topk (OOS #2):**

- `data_movement/transpose/.../{transpose_wh, transpose_wh_sharded}.cpp`
- `data_movement/sort/.../{sort_single_row_multi_core, sort_common.hpp}.cpp`
- `data_movement/concat/.../height_sharded_width_concat_two_tensors.cpp`
- `experimental/transformer/split_query_key_value_and_split_heads/.../transpose_wh_sharded.cpp`
- `experimental/ssm/hc_sum_reduce/.../ssm_1d_sum_reduce.cpp`
- `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`
- `reduction/topk/.../{topk, topk_common_funcs.hpp, topk_local, topk_final}.cpp`
- `reduction/sampling/.../sampling.cpp` &nbsp; `reduction/moe/.../moe.cpp`
- `experimental/deepseek_prefill/.../moe_grouped_topk.cpp`
- `experimental/reduction/deepseek_grouped_gate/.../deepseek_grouped_gate.cpp`
- `experimental/topk_router_gpt/.../compute.cpp`
- `reduction/accumulation/ema/.../ema_compute.cpp`
- `reduction/generic/.../welford_reduce_{h,w,hw}.cpp` (welford + transpose)

**Matmul (OOS #8):**

- `matmul/.../{bmm, bmm_large_block_zm_fused_bias_activation, bmm_tilize_untilize}.cpp`
- `moreh/moreh_matmul/.../moreh_matmul.cpp`
- `experimental/matmul/attn_matmul/.../transformer_attn_matmul.cpp`
- `experimental/deepseek/{moe/moe_gate_mm, mla/matmul_wo}/.../compute*.cpp`
- `experimental/conv3d/.../compute.cpp` &nbsp; `conv/conv2d/.../compute_depthwise_conv1d.cpp`
- `experimental/minimal_matmul/.../compute.cpp` &nbsp; `experimental/ccl/all_gather_minimal_matmul_async/.../compute.cpp`

**Embedding / random / sdpa fused (OOS #3, #10, #11):**

- `embedding_backward/.../embedding_backward.cpp` (**OOS #10** `reshuffle_rows_tile`)
- `{uniform, rand, bernoulli, randn}/.../compute*.cpp` (**OOS #11** rand_tile with runtime seed)
- `transformer/sdpa/.../{compute_common.hpp, compute_streaming.hpp}` (**OOS #3** fused FPU+SFPU+reduce in DEST window)

**Helper headers (composite, not own migration target):**

- `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (composite `*_to_cb` helpers; migrated indirectly as callers switch to chain)
- `ttnn/cpp/ttnn/kernel_lib/{copy_tile_helpers.hpp, matmul_kloop_helpers.hpp}` (chain lib internals)

---

## 4. Patterns left raw &mdash; root-cause taxonomy

| # | Pattern | Why chain can't express it |
|---|---|---|
| 1 | ~~Welford accumulator~~ &mdash; **reclassified 2026-05-28 to &sect;5 helper gap.** Welford's running mean/M2 live in SFPU LREG4/LREG5, **outside DEST** &mdash; chain's per-call DEST acquire/release does not conflict with welford state. Each LLK primitive (`welford_init`, `welford_clear`, `welford_update<lut_size>`, `welford_save_state`, `welford_restore_state`, `welford_finalize_to_row`, `welford_finalize_to_face`, `welford_reinit`) is a discrete SFPU op wrappable as a chain element struct, same shape as `Recip` / `Cumsum` / `Dropout` (see &sect;5 row). The reason older audits called this OOS was conflating the welford ops with the `transpose_wh_tile` that some welford kernels interleave (which IS a separate gap &mdash; OOS #2 below, also reclassifiable as a helper gap if/when transpose structs are added). |
| 2 | `transpose_wh_tile` | No chain primitive. Used in welford finalize, ema, rotary single-tile rotate_half. |
| 3 | Fused FPU + SFPU + reduce in one DEST acquire window | Chain stages cannot interleave with `reduce_tile` inside the same DEST acquire. |
| 4 | Macro-injected SFPU activation (`SFPU_OP_INIT_ACTIVATION` / `BINARY_OP` / `PROCESS_POST_ACTIVATIONS` / `BINARY_SFPU_OP` / `TERNARY_SFPU_OP`) | No element wraps a user-defined preprocessor macro that expands to arbitrary SFPU calls. Resolvable per-kernel via an `ActivationMacro : UnaryOp` shim, OR per-family via `get_defines` refactor to emit chain identifiers (precedent: `bcast_op_utils` in `65b34f7740f`). |
| 5 | `chain_llk` DSL | Separate graph DSL with its own normed_output / gamma / beta nodes. |
| 6 | L1 packer accumulator (`pack_reconfig_l1_acc(1)`) | Chain doesn't expose L1 acc mode for `PackTile`. |
| 7 | Runtime CB ids in template args (q/k branch via runtime `is_q`) | Chain elements require constexpr CB ids; the runtime branch may exceed TRISC2 size. |
| 8 | `matmul_tiles` / `matmul_block` | Eltwise-chain scope excludes matmul. Tile matmul is also an anti-pattern (only block matmul helpers should be used). |
| 9 | Cumulative wait + tile-indexed access with conditional last-tile branch | Expressible with `Bulk + Block + TileBaseRuntime` but conditional mask-on-last-tile requires chain split with careful TileBase math; low ROI vs the test surface. |
| 10 | `reshuffle_rows_tile` | No chain primitive. |
| 11 | ~~`rand_tile` with runtime seed~~ &mdash; **RESOLVED 2026-05-29.** `RandTile<Slot>` (eltwise_rand.hpp) now takes `from` / `scale` at construction; the Seed NTTP was dropped and `init()` is a no-op. Caller invokes `rand_tile_init(seed)` out-of-band with the runtime seed (same pattern as `Dropout`). First user: `rand/device/kernels/compute_uniform.cpp`. |
| 12 | Same-CB-in-place inside one chain (output CB == input CB) | Chain reserves output BEFORE popping input; in-place inside a single chain is unsafe. Affects groupnorm gamma/beta path, dit_layernorm post_allgather_welford mul-by-recip / mul-by-gamma, rms_allgather. |

## 5. Helper gaps (would unlock specific kernels)

| Gap | Unlocks | Cost |
|---|---|---|
| ~~`DestReuseBinarySfpu`~~ | **No currently-blocked kernels.** Erratum: prior versions of this file listed batch_norm_sfpu / running_statistics_sfpu / eltwise_binary_sfpu / moreh_int_sum_* under this gap. Those kernels use plain `CopyTile + CopyTile + AddBinary<D0,D1,D0>` (fresh D0/D1 each stage), already expressible via the `AddBinary` / `SubBinary` / `MulBinary` / `DivBinary` / `BinaryMax` / `BinaryMin` family in `eltwise_binary_sfpu.hpp`. A `DestReuseBinarySfpu` helper would still be useful if a future kernel wants to fuse running D0 across consecutive SFPU binaries without re-copying, but no current code needs it. | n/a |
| DEST-accumulating BinaryFpu (`add_tiles_init(acc_to_dest=true)`) | `layernorm_pre_allgather_2d.cpp`, `rmsnorm_pre_allgather_2d.cpp` | Extend `binary_op_helpers.hpp` |
| BlockIterOffset with runtime offset (`block.to_global(i)`) | Layernorm gamma/beta path | Partial &mdash; chain has `BlockIterOffset`; runtime-offset variant TBD |
| Same-CB-in-place inside chain (or chain-level rewind semantics) | groupnorm / dit_layernorm gamma+beta optional path, rms_allgather | `HeldBulk + BlockSize` for the second operand was tried, numerically wrong. Needs explicit window-rewind. |
| L1 acc PackTile mode | groupnorm_sharded_v2 `(x-E[x])^2`, minimal_matmul, deepseek_prefill | Resolves OOS #6 |
| Runtime-templated CB id chain element | rotary_embedding_llama_fused_qk q/k branch | OOS #7. May still exceed TRISC2 size. |
| `ActivationMacro : UnaryOp` (per-kernel) | layernorm.cpp, layernorm_sharded.cpp | Small. Alternative to family-level `get_defines` refactor. |
| `get_defines` refactor for binary_ng family | ~22 kernels in &sect;3d | One refactor unlocks the family (precedent: bcast). |
| **Welford chain structs** (`WelfordInit`, `WelfordClear`, `WelfordUpdate<DstSlot, LutSize>{lut, start_idx}`, `WelfordSaveState<MeanDstSlot>`, `WelfordRestoreState<MeanDstSlot>`, `WelfordFinalize<MeanDstSlot, VarDstSlot>{mean_src_idx, total_count_recip}`, `WelfordReinit<Cb>`) | Pure-welford kernels (no `transpose_wh_tile` interleaving): `reduction/generic/welford_reduce_h.cpp`, `welford_reduce_hw.cpp`. Plus the welford-accumulator stages in 8+ welford-tagged kernels that currently sit in &sect;3b (the non-transpose stages would migrate, transpose stages stay raw until the transpose gap also closes). | Mirror the `Cumsum` / `Recip` pattern in `eltwise_math.hpp` &mdash; one struct per LLK primitive, `init()` + `exec_impl()`. SFPU state lives in LREG4/LREG5 (outside DEST) so the chain's per-call DEST acquire/release does NOT conflict. |

## 6. Helper-library changes this branch

- `eltwise_chain.inl` &mdash; `window_1d<OperandKind>` helper (`14a5a61e462`): Bulk+Scalar now emits wait_front(1)/pop_front(1) instead of n_tiles.
- `bcast_op_utils::get_defines` (`65b34f7740f`) &mdash; emits `CHAIN_BCAST_OP` / `CHAIN_BCAST_DIM` for the bcast family.
- `typecast_program_factory` regular + sharded (`6589d8167d8`) &mdash; emits `CHAIN_TYPECAST_IN_DF` / `CHAIN_TYPECAST_OUT_DF`.
- `kernel_lib`: `BinaryDataFormatReconfig` / `DestReuseReconfig` SrcA/SrcB single-side selectors (`428c1dc9ae1`).
- `kernel_lib`: `BulkDrain` / `DeferredPop` emission + (kind, lifecycle) legality assert (`3fa2249c989`).
- `kernel_lib`: per-stage `pack_reconfig` emission for heterogeneous-output chains (`cca9d24572a`).
- `kernel_lib`: `eltwise_chain` emit `_with_dt` reconfig overloads, coalesce srca+srcb (`eca831d6080`).
- `kernel_lib`: split hoist decision into per-cohort flags (`4c9cd946511`).
- `kernel_lib`: tighten `is_legal_kind_lifecycle` to structural cells + ban 1D Row/Col (`6f962533515`).

### New chain elements added this session

| Element | File | Commit | Notes |
|---|---|---|---|
| `TanhDerivative<Approx, Slot>` | `eltwise_activations.hpp` | `ef60346171a` | sech²(x). Removed older non-templated dup from `eltwise_special.hpp` (`d8e38a11652`). |
| `GeluDerivative<Approx, Slot>` | `eltwise_activations.hpp` | `ef60346171a` | Polynomial-accurate GELU'(x). |
| `Logsigmoid<In0, In1, Out>` | `eltwise_activations.hpp` | `ef60346171a` | Binary in-DEST: caller pre-loads x and exp(-x). |
| `Cumsum<Slot>{first}` | `eltwise_math.hpp` | `ef60346171a` | In-DEST columnwise cumsum, runtime `first` flag. |
| `Dropout<Slot>{prob, scale}` | `eltwise_scalar.hpp` | `6cf9b8a4d11` | init is no-op; caller runs `dropout_kernel_init(seed)` outside chain. |

## 7. Cross-cutting bug fixes / lifecycle pitfalls

- **CallerManaged needs external wait_front** (commit `b83db8b05c0`):
  `CallerManaged` emits no `cb_wait_front` on the chain side. For held CBs
  without an external wait, the underlying mul/sub LLK reads unsynchronized
  &rarr; reader deadlocks at `cb_reserve_back`. Fixed in 6 softmax large
  kernels by replacing `CallerManaged` with `HeldStream`. Memory:
  `feedback_callermanaged_needs_external_wait.md`.
- **Bulk + Scalar over-waits** for held 1-tile broadcast (`b83db8b05c0`): use
  `CallerManaged + Scalar` with explicit external wait, OR
  `HeldStream + Scalar`. Memory: `feedback_bulk_scalar_overwait.md`.
- **DestReuseBinarySfpu erratum** (this revision, 2026-05-26): The gap listed
  five kernels as blocked &mdash; on re-investigation they're plain
  `CopyTile + CopyTile + BinarySfpu` and the helpers already exist. Those
  kernels moved from &sect;3c to &sect;3a MIGRATE NOW.

## 8. Methodology &mdash; per-file audit before commit

Before committing each migration:

1. `grep -nE '(add|sub|mul)_tiles|copy_tile|pack_tile|reconfig_data_format|pack_reconfig|tile_regs_(acquire|release|wait|commit)|ACQ|REL|rsqrt_tile|exp_tile|recip_tile|sqrt_tile|log_tile|tanh_tile|mask_tile|abs_tile|fill_tile|reduce_tile|transpose_wh|matmul_tiles|rand_tile|reshuffle_rows|welford' <kernel>` to enumerate every raw LLK block remaining.
2. For each remaining block, classify:
   - **migratable now** &mdash; known chain pattern, not done in this PR.
   - **needs helper update** &mdash; op struct / policy / index mode missing (see &sect;5).
   - **out of scope** &mdash; cite root-cause # from &sect;4.
3. List each in the commit's `Skipped:` section AND append/update the per-kernel row in this file.

Anti-pattern: noting only blocks happened-to-look-at while migrating the chosen stage. The audit grep is the safety net. See memory `feedback_no_lazy_partials.md`.

## 9. Companion artifacts

- `docs/eltwise_chain_migration_candidates.html` &mdash; per-kernel chain-shape proposal for the unmigrated queue, with priority ranking.
- `ttnn/cpp/ttnn/operations/normalization/migration_status.md` &mdash; original normalization-only audit (per-kernel difficulty classification + cross-op attempts).
