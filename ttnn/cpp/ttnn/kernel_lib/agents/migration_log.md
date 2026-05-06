# Migration Log — eltwise_chain helper bulk migration

Status one-liner per kernel. Format: `<status> <repo-rel-path> :: <one-line note>`

## Already migrated at HEAD (reference kernels for the pattern)

MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_identity_kernel.cpp :: pure copy via `compute_kernel_lib::copy<>` convenience wrapper
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardswish_kernel.cpp :: hardswish(x) = x * hardsigmoid(x); INP_FLOAT32 fan-out + INP_FLOAT DestReuseBinary variants
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp :: tanhshrink(x) = x - tanh(x); INP_FLOAT32 fan-out + INP_FLOAT DestReuseBinary variants
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp :: mish(x) = x * tanh(softplus(x)); USE_APPROX-templated; INP_FLOAT32 + INP_FLOAT variants

## Migrated this run

MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/kernels/compute/eltwise_bw_tanh_deriv.cpp :: grad_in = grad_out * tanh_derivative(input); two-CB CopyTile + TanhDerivative + MulBinary + Pack
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/kernels/compute/eltwise_bw_gelu_poly.cpp :: grad_in = grad_out * GeluDerivative(input); local GeluDerivative UnaryOp wraps gelu_derivative_tile<false>
MIGRATED ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_poly.cpp :: identical to gelu_bw above (different op location)
SKIPPED:complex-multi-tile-DEST ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp :: 9-DEST-slot polynomial with copy_dest_values + per-tile fill scratch; doesn't fit chain shape
MIGRATED ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp :: streaming Dropout SFPU; local Dropout UnaryOp; dropout_kernel_init(seed) once before chain
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logit_kernel.cpp :: two-stage chain — stage 1 optional Clamp + pack to scratch CB; stage 2 RsubUnary + DivBinary + Log
MIGRATED ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logsigmoid_kernel.cpp :: D0=x; D1=-x; D1=exp(D1); LogSigmoidBinary(D0,D1)→D0; local LogSigmoidBinary BinaryOp
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp :: SFPU_OP_CHAIN_0 host-defined macro
SKIPPED:complex-multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp :: where_tile + lgamma_stirling_float_tile + lgamma_adjusted_tile, multi-tile scratch
SKIPPED:complex-multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_fast_kernel.cpp :: identical pattern to lgamma_kernel
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp :: SFPU_OP_CHAIN_0
MIGRATED ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp :: pure copy via `compute_kernel_lib::copy<>` convenience wrapper

## Programming examples (small but production-relevant)

MIGRATED tt_metal/programming_examples/sfpu_eltwise_chain/kernels/compute/compute.cpp :: softplus(x) = log(exp(x) + 1) via 5-element chain
MIGRATED tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp :: streaming exp via `unary_op<Exp<>>` convenience wrapper
MIGRATED tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp :: streaming add via `binary_add` convenience wrapper
MIGRATED tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp :: single-tile add via `binary_add` convenience wrapper
MIGRATED tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp :: custom my_add_tile wrapped as MyAdd BinaryOp; chain dispatches normally
MIGRATED tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp :: custom smoothstep wrapped as Smoothstep UnaryOp with runtime edges/inv_delta
MIGRATED ttnn/examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp :: streaming add via `binary_add` convenience wrapper
MIGRATED ttnn/examples/lab_multicast/kernels/compute/tiles_copy.cpp :: pure copy via `compute_kernel_lib::copy<>` convenience wrapper

## Ternary

MIGRATED ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp :: out = a + scalar*b*c (int); FillInt + 4-DEST chain with custom MulIntInPlace/AddIntInPlace ops (BinaryOp's distinct-slot static_assert blocks aliasing)
MIGRATED ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu.cpp :: out = a + scalar*b*c (float); BinaryFpu(b,c)->Mul + optional MulUnary(scalar) + DestReuseBinary(a, Add); templated on scalar_is_not_1 runtime branch
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.cpp :: TERNARY_SFPU_OP_FUNC macro
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp :: TERNARY_SFPU_OP_FUNC macro
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp :: TERNARY_SFPU_OP_FUNC macro
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_bcast.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_rowbcast.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu_bcast.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_tts_tst.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp :: broadcast variant
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp :: broadcast variant

## Binary / binary_ng — all macro-injection or broadcast

SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp :: ELTWISE_OP, SFPU_OP_*
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp :: BINARY_SFPU_OP, SFPU_OP_*
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h.cpp :: broadcast
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_w.cpp :: broadcast
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp :: broadcast
SKIPPED:broadcast ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised.cpp :: broadcast
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary.cpp :: PREPROCESS, BINARY_OP, HAS_ACTIVATIONS, BCAST_INPUT
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp :: PREPROCESS, BINARY_OP, HAS_ACTIVATIONS macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp :: PREPROCESS, BINARY_OP macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp :: BINARY_SFPU_OP, PREPROCESS macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp :: BINARY_SFPU_OP, PREPROCESS macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp :: BINARY_SFPU_OP, PREPROCESS macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_no_bcast.cpp :: BINARY_SFPU_OP, FILL_LLK macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu.cpp :: BINARY_SFPU_OP, FILL_LLK, PREPROCESS macros
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu_scalar.cpp :: BINARY_SFPU_OP, FILL_LLK, PREPROCESS macros

## tt-train / ttml — block matmul helpers + multi-stage CB pipelines

SKIPPED:block-matmul-helpers tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/compute/silu_bw_kernel.cpp :: pack_and_push_block, multi-CB pipeline
SKIPPED:block-matmul-helpers tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/compute/swiglu_elemwise_bw_kernel.cpp :: pack_and_push_block, multi-stage
SKIPPED:reduction tt-train/sources/ttml/metal/ops/cross_entropy_fw/device/kernels/compute/cross_entropy_fw_kernel.cpp :: reduce_tile, mask_tile, unary_bcast, multi-stage
SKIPPED:reduction tt-train/sources/ttml/metal/ops/cross_entropy_bw/device/kernels/compute/cross_entropy_bw_kernel.cpp :: complex multi-stage
SKIPPED:welford tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/compute/layernorm_fw_kernel.cpp :: welford / reduction
SKIPPED:welford tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp :: welford / reduction
SKIPPED:welford tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/compute/rmsnorm_fw_kernel.cpp :: reduction + bcast
SKIPPED:welford tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/compute/rmsnorm_bw_kernel.cpp :: reduction + bcast
SKIPPED:welford tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/compute/polynorm_fw_kernel.cpp :: reduction + bcast
SKIPPED:reduction tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp :: reduction
SKIPPED:reduction tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/compute/softmax_backward_kernel.cpp :: reduction
SKIPPED:sdpa tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp :: SDPA flash, matmul + softmax
SKIPPED:sdpa tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_kv_compute_kernel.cpp :: SDPA backward
SKIPPED:sdpa tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_q_compute_kernel.cpp :: SDPA backward

## DeepSeek B1 unified ops — all-RISC, sharded CB rd_ptr manipulation

SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/eltwise_add.hpp :: BRISC/NCRISC/TRISC unified, CB read-pointer offset, IsActiveCore template
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/eltwise_mul.hpp :: same pattern as eltwise_add.hpp

## tt_metal/kernels/compute

SKIPPED:macro-injection tt_metal/kernels/compute/eltwise_binary.cpp :: ELTWISE_OP, SFPU_OP_CHAIN_0, ARCH_QUASAR multi-arch
SKIPPED:macro-injection tt_metal/kernels/compute/eltwise_sfpu.cpp :: SFPU_OP_CHAIN_0
SKIPPED:no-compute tt_metal/kernels/compute/blank.cpp :: empty kernel
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/eltwise_sfpu.cpp :: SFPU_OP_CHAIN_0
SKIPPED:macro-injection ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp :: TYPECAST_LLK macro

## Held-DEST / mid-loop pack accumulators

SKIPPED:held-dest ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_all.cpp :: held-DEST recurrence with mid-loop pack
SKIPPED:held-dest ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp :: held-DEST recurrence
SKIPPED:held-dest ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp :: BINARY_OP macro + accumulator CB ping-pong

## Normalization — welford / reductions / multi-stage CB pipelines

SKIPPED:welford ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/*.cpp :: welford / reduction / sharded
SKIPPED:welford ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/*.cpp :: welford / reduction
SKIPPED:reduction ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp :: ACQ()/REL() macros + bcast + reduce
SKIPPED:reduction ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather*.cpp :: reduction
SKIPPED:reduction ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/*.cpp :: reduction / welford
SKIPPED:multi-stage ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp :: PREPROCESS-style multi-stage with intermediate CBs
SKIPPED:multi-stage ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp :: same as batch_norm_kernel + last_srca_cb tracking
SKIPPED:moreh ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_kernel.cpp :: sub_tiles_to_cb / mul_tiles_to_cb / add_tiles_to_cb (moreh helpers)
SKIPPED:moreh ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_sfpu_kernel.cpp :: moreh helpers + multi-CB

## Experimental / CCL / SDPA / matmul / transpose / pool — all out-of-scope

SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp :: matmul_block + pack_untilize + ring all-to-all
SKIPPED:no-compute ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute_dummy.cpp :: pure CB drain
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp :: multi-tile DEST reduction
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp :: multi-tile DEST reduction
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/kernels/compute/reduction.cpp :: multi-tile DEST reduction
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/reduction.cpp :: multi-tile DEST reduction
SKIPPED:welford ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/compute/layernorm_pre_allgather_welford.cpp :: welford
SKIPPED:welford ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp :: welford
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp :: reduction + bcast
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp :: reduction
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute*.cpp :: matmul
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp :: matmul
SKIPPED:complex ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/*/device/kernels/compute/*.cpp :: dispatch / combine / topk
SKIPPED:transpose ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/*.cpp :: rotary embedding
SKIPPED:transpose ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/*.cpp :: rotary embedding
SKIPPED:reshuffle ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp :: reshuffle_rows_tile + cache mailbox
SKIPPED:transpose ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp :: transpose_wh + tilize/pack_untilize helpers
SKIPPED:transpose ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh*.cpp :: transpose_wh
SKIPPED:transpose ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw*.cpp :: transpose_xw
SKIPPED:tilize-untilize ttnn/cpp/ttnn/operations/data_movement/{tilize,untilize}/device/kernels/compute/*.cpp :: not eltwise
SKIPPED:acquire-dst ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp :: acquire_dst()/release_dst() pattern
SKIPPED:matmul ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/*.cpp :: matmul / bmm
SKIPPED:not-eltwise ttnn/cpp/ttnn/operations/pool/**/device/kernels/compute/*.cpp :: pooling
SKIPPED:not-eltwise ttnn/cpp/ttnn/operations/conv/**/device/kernels/compute*.cpp :: convolution
SKIPPED:reduction ttnn/cpp/ttnn/operations/reduction/**/device/kernels/compute/*.cpp :: reductions
SKIPPED:bcast ttnn/cpp/ttnn/operations/data_movement/bcast/**/device/kernels/compute/*.cpp :: broadcast
SKIPPED:rand ttnn/cpp/ttnn/operations/{uniform,bernoulli,randn,rand}/device/kernels/compute*.cpp :: RandTile element has non-static init() incompatible with chain's static E::init() dispatch path
SKIPPED:helper ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp :: tilize/untilize helpers, not eltwise
SKIPPED:helper ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/*.cpp :: page-level update logic
SKIPPED:no-op ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_no_bcast_to.cpp :: empty kernel
SKIPPED:no-op tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp :: no compute
SKIPPED:no-op tt_metal/programming_examples/contributed/multicast/kernels/compute/void_compute_kernel.cpp :: no compute
SKIPPED:helpers ttnn/ttnn/operations/toy_*/kernels/compute.cpp :: already use binary_op_helpers / tilize_helpers / etc.
SKIPPED:tiny ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_*.cpp :: 1-line seed init, no benefit from chain

## Tests and synthetic kernels — out of scope per task

SKIPPED:test-fixture tests/tt_metal/tt_metal/test_kernels/compute/*.cpp :: per task: do not touch test files
SKIPPED:test-fixture tools/tests/triage/hang_apps/*/kernels/compute/*.cpp :: per task: do not touch test files

## Run 7 — additional reconnaissance (extends prior coverage)

### moreh — every compute kernel re-examined; all genuinely incompatible
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/moreh_abs_pow_kernel.cpp :: power_tile_to_cb + mid-loop mask_tile + copy_tile_init switching
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp :: mul_tiles_to_cb / add_tiles_to_cb / sub_tiles_to_cb (held-CB pingpong) — multi-stage Adam recurrence
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp :: same pattern as adam
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/moreh_sgd.cpp :: pure moreh_common helpers + held-CB ping-pong
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/moreh_clip_grad_norm_step1_kernel.cpp :: power_tile_to_cb + mid-loop mask + held-DEST accumulator
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/moreh_clip_grad_norm_step2_kernel.cpp :: held-DEST add accumulator + power_tile_to_cb
SKIPPED:bcast ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/moreh_clip_grad_norm_step3_kernel.cpp :: mul_tiles_bcast_scalar (broadcast not supported in chain v1)
SKIPPED:bcast ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp :: mul_tiles_bcast_scalar
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp :: reduce_tile + mask_tile + bcast_cols_to_cb (multi-stage softmax)
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/*.cpp :: all softmax variants — reductions + bcast
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/*.cpp :: same pattern
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/*.cpp :: layernorm reductions
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/*.cpp :: layernorm bw
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/**/kernels/*.cpp :: norm reductions
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/*.cpp :: norm bw
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp :: reduce
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/**/kernels/*.cpp :: sum reductions
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/*.cpp :: mean reductions
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp :: mean bw
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/device/**/kernels/*.cpp :: nll reductions
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/*.cpp :: nll
SKIPPED:reduction ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_*.cpp :: reduce + mask_tile
SKIPPED:matmul ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/moreh_matmul.cpp :: matmul
SKIPPED:acquire-dst ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp :: ACQ()/REL() + reduce
SKIPPED:acquire-dst ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp :: ACQ()/REL() + bcast_scalar
SKIPPED:moreh-helpers ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/*.cpp :: reader/writer kernels (no compute)

### tt-train / ttml — block matmul helpers + multi-stage CB pipelines (re-confirmed)
SKIPPED:block-matmul-helpers tt-train/sources/ttml/metal/optimizers/sgd/device/kernels/compute/sgd_kernel.cpp :: pack_and_push_block + mul_tiles_bcast_scalar
SKIPPED:block-matmul-helpers tt-train/sources/ttml/metal/optimizers/adamw/device/kernels/compute/adamw_kernel.cpp :: same pattern
SKIPPED:reduction tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/compute/polynorm_fw_kernel.cpp :: norm reduction

### experimental / ssm / bcast_to / paged_cache / hang_device / integral_image
SKIPPED:transpose ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/ssm_eltwise_mul.cpp :: transpose_wh + bcast_rows
SKIPPED:complex ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp :: prefix scan (multi-stage)
SKIPPED:bcast ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_*.cpp :: broadcast variants
SKIPPED:test-fixture ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp :: test hang loop
SKIPPED:cumsum ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/intimg_compute.cpp :: cumsum + transpose_wh_dest + bcast
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp :: reduce_tile
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_reduce.cpp :: reduce
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp :: multi-stage gating
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp :: topk
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp :: reduce
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/*.cpp :: reduce
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp :: conv3d matmul
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp :: matmul
SKIPPED:transpose ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/*.cpp :: rotary
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp :: attn matmul
SKIPPED:transpose ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp :: split + transpose
SKIPPED:tilize ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/deepseek_moe_post_combine_tilize_compute.cpp :: tilize
SKIPPED:reduction ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_compute.cpp :: reduce
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp :: matmul
SKIPPED:matmul ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/compute.cpp :: moe matmul
SKIPPED:tilize ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_compute.cpp :: tilize
SKIPPED:tilize ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/tilize_compute.cpp :: tilize

### models/demos/deepseek_v3_b1/unified_kernels — all RISC-unified, sharded CB, persistent loop
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/local_reduce.hpp :: persistent_loop reduce
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/gated_local_reduce_kernel.cpp :: doesn't exist (gated_reduce.hpp instead)
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/gated_reduce.hpp :: persistent_loop + gated reduce
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/sdpa_reduce_worker.hpp :: SDPA reduce
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/reduce_to_all_b1.hpp :: reduce + multi-RISC
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/rope.hpp :: rotary embedding
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/broadcast.hpp :: broadcast
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp :: flash MLA
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/persistent_loop.hpp :: persistent loop infrastructure (not eltwise)
SKIPPED:unified-op models/demos/deepseek_v3_b1/unified_kernels/termination.hpp :: termination signal (not eltwise)

### binary_ng kernels_ng (sub-folder for v2 broadcast kernels) — all bcast variants
SKIPPED:bcast ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_*_bcast.cpp :: all bcast variants — chain v1 doesn't support bcast
SKIPPED:bcast ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_sfpu_*_bcast.cpp :: all sfpu bcast variants
SKIPPED:bcast ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_where_sfpu_*_bcast.cpp :: where bcast variants

### binary / binary_ng compute kernels — runtime per_core_block_size with i*2/i*3 multi-DEST scratch
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp :: PREPROCESS + i,i,i pack pattern over runtime per_core_block_size
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp :: PREPROCESS + multi-tile DEST
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp :: i*2/i*2+1 multi-DEST scratch + macros
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp :: i*2/i*2+1 multi-DEST scratch + macros
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp :: same
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_no_bcast.cpp :: i*3/i*3+1/i*3+2 multi-DEST + FILL_LLK + BINARY_SFPU_OP
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu.cpp :: same pattern
SKIPPED:multi-tile-DEST ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu_scalar.cpp :: same pattern

### Conclusion for Run 7
The chain helper migration is essentially complete on the easy/medium targets covered by the
current chain API surface (single-tile DEST, no broadcast, no held-DEST, no multi-RISC unified,
no reductions, no multi-stage CB pipelines, no transpose/tilize, no matmul). The remaining
production kernels are blocked on chain API gaps (broadcast support — needed by ~30+ moreh /
tt-train / binary_ng kernels; multi-tile DEST scratch — needed by binary_ng with
`num_tiles_per_cycle > 1`; held-DEST recurrence — needed by Adam-family optimizers).

## Partial migrations — moreh / tt-train batch (Run 7 follow-up)

Each entry: `PARTIAL <repo-rel-path> :: migrated: <stage list>; skipped: <stage list> — reason: <reason>`

PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp :: migrated: rsqrt(var+eps) chain (BinaryFpu(Add) + Rsqrt + PackTile); skipped: xsum/xmm/xmm2/xmm2sum accumulators, gamma/beta bcast loops — reason: in-place CB recurrence + bcast reductions stay raw with `compute_kernel_lib::reduce<>`
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp :: migrated: rsqrt(var+eps) chain (BinaryFpu(Add) + Rsqrt + PackTile); skipped: same as small kernel — reason: same
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp :: migrated: cb_recip_nrstd = n_recip_n[1] * rstd[0] (BinaryFpu(Mul, Cols|Scalar) + Pack), cb_tmp1 = ndymdysum - yydysum (BinaryFpu(Sub) + Pack), cb_dx = tmp1 * recip_nrstd (BinaryFpu(Mul) + Pack); skipped: dycopy mask block, dyadd/ydyadd held-DEST accumulators, xmm/y bcast block, ndy/yydysum bcast + reduces — reason: mid-loop conditional mask, in-place CB recurrence, multi-stage scratch
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp :: migrated: cb_recip_nrstd = n_recip_n[1] * rstd[0] (BinaryFpu(Mul, Cols|Scalar) + Pack), cb_tmp4 = ndymdysum - yydysum (BinaryFpu(Sub) + Pack), cb_dx = tmp4 * recip_nrstd (BinaryFpu(Mul) + Pack); skipped: same as small kernel — reason: same
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp :: migrated: cb_ydy = cb_y * cb_dycopy (BinaryFpu(Mul) + Pack); skipped: dycopy mask block, dyadd/ydyadd held-DEST accumulators, "Just copy" stages, reduce<> — reason: mid-loop mask + held-DEST + with_dt CopyTile reconfig with old_cb=0
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/moreh_clip_grad_norm_step3_kernel.cpp :: migrated: y = x * clip_coef_clamped scalar bcast (BinaryFpu(Mul, Scalar) + Pack); skipped: nothing — reason: full main loop migrated (broadcast IS supported in chain v1, contradicting earlier log entry)
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp :: migrated: f(x) chain — Copy + (Nez | Abs) + (Negative if MINUS_INF) + Pack across IS_ZERO/MINUS_INF macro permutations; skipped: Add/Max accumulator into cb_cal, final cb_y copy/negative — reason: in-place CB recurrence on cb_cal (cb_pop_front + cb_push_back same CB)
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/moreh_nll_loss_step2_kernel.cpp :: migrated: cb_divisor_recip = 1/cb_divisor (CopyTile(WaitAndPop) + Recip + Pack) under #if defined(DIVISOR); skipped: per-tile negative + WEIGHT/DIVISOR mul-bcast pipeline — reason: cross-stage DEST handoff via temporary CBs
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp :: migrated: cb_tmp1 = 1/cb_divisor (CopyTile(NoWaitNoPop) + Recip + Pack) under #if defined(DIVISOR); skipped: main bcast multiplications + negative for tmp2/input_grad — reason: cross-stage scratch handoff and in-loop mul_bcast_scalar with held tmp1
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp :: migrated: full main loop — BinaryFpu(Add, Cols/Rows/Scalar) + PackTile when bcast OR CopyTile + PackTile when no bcast; skipped: nothing
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp :: migrated: full main loop — stage 1 (Add/Copy + Pack to cb_intermed0) + stage 2 (BinaryFpu(Mul, Scalar) cb_intermed0 * cb_scalar -> cb_out0 + Pack); skipped: nothing
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp :: migrated: stage 2 — BinaryFpu(Mul, Scalar) cb_intermed0 * cb_scalar -> cb_out0 + Pack; skipped: enable_reload accumulator stage — reason: held-DEST recurrence + in-place CB on cb_intermed0
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp (extras) :: migrated additionally: dyadd seed (CopyTile(NoWaitNoPop) + Pack), cb_ydy = cb_y[wt]*cb_dycopy[wt] (BinaryFpu(Mul) + Pack via Pinned indices), ydyadd seed (CopyTile + Pack); skipped: held-DEST add accumulators for dyadd/ydyadd, mid-loop mask blocks
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp (extras) :: migrated additionally: dyadd seed (CopyTile + Pack), cb_ydy = cb_y * cb_dycopy single-tile mul (BinaryFpu(Mul) + Pack), ydyadd seed (CopyTile + Pack); skipped: same as small kernel

### Already-migrated reference partials (run 7 baseline)
The following were already partial-migrated in earlier runs and remain unchanged:
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp :: migrated: |x| chain (CopyTile + Abs + Pack); skipped: power_tile_to_cb, held-DEST add accumulator
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp :: migrated: copy seed for cb_cal under row_idx==0 branch (CopyTile + Pack); skipped: f(x) mid-loop mask block, held-DEST accumulator
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/moreh_abs_pow_kernel.cpp :: header comment indicates intent; full migration of |x| stage blocked by mid-loop mask injection — left raw

### Skip rationales applied this run (no migration possible under chain v1)
- moreh_norm/moreh_norm_h, moreh_norm_w (and ord_other/moreh_norm_w): runtime `do_mask_h/w` mid-loop mask injection in |x| stage.
- moreh_clip_grad_norm_step1: same |x| + mid-loop mask + held-DEST accumulator pattern as moreh_abs_pow.
- moreh_clip_grad_norm_step2: in-place CB recurrence on cb_x (cb_pop_front + cb_push_back same CB).
- moreh_layer_norm_backward_input_grad_*: dycopy/xmm mask blocks (mid-loop conditional mask), dyadd/ydyadd held-DEST accumulators (in-place CB recurrence), bcast scratch handoff stages (cb_ndy/cb_yydysum/cb_ndymdysum cross-stage scratch).
- moreh_softmax_*, moreh_softmax_backward_*: mostly `*_tile_to_cb` moreh helpers (already chain-equivalent), reduce + mask-tile state machines, multi-stage CB pipelines.
- moreh_bias_backward_*: mid-loop mask + reduce_init/reduce_uninit state machine.
- moreh_adam, moreh_adamw, moreh_sgd: `*_tiles_to_cb` moreh helpers + held-DEST recurrence on cb_tmp1/cb_tmp2 (cb_pop_front + cb_push_back) for bias correction / sqrt / add-eps stages.
- tt-train silu_bw, swiglu_elemwise_bw, cross_entropy_*, sgd, adamw: block matmul helpers (`pack_and_push_block`) + multi-stage CB pipelines (HQ rule: keep block matmul on raw).

## Partial migrations — transformer / experimental / matmul / SDPA / data_movement batch

PARTIAL ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp :: migrated: mul/sum/copy single-tile helpers via templated wrappers (BinaryFpu Add/Mul + CopyTile + PackTile chains); skipped: pack_block_rows_into_tiles / pack_block_tiles_into_rows (untilize_block / tilize_block — out of chain scope per HQ skip rules)
PARTIAL ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_col_bcast_to.cpp :: migrated: per-tile UnaryBcast<Col> + PackTile body inside outer (n,c,th) loop; skipped: outer iteration with `num_tiles_read += Wt - start_tw` early-exit stays raw (chain helper iterates a single linear count)
PARTIAL ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_row_bcast_to.cpp :: migrated: per-tile UnaryBcast<Row> + PackTile body inside outer (n,c,th,tw) loop; skipped: outer 4-axis early-exit iteration stays raw
PARTIAL ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/compute/compute_interleaved_scalar_bcast_to.cpp :: migrated: per-tile UnaryBcast<Scalar> + PackTile body inside outer (n,c) loop; skipped: outer iteration with `num_tiles_read += HtWt - start_t` (one input tile contributes more than 1 to the counter)
PARTIAL ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp :: migrated: ones-tile fill+pack block (FillScalar + PackTile chain); skipped: matmul + SFPU-on-PACK + SwiGLU/SiLU activation main loops (matmul out of scope; SFPU-on-pack uses PACK-thread llk_math_eltwise_* which the chain doesn't model)
PARTIAL ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/compute.cpp :: migrated: ones-tile fill+pack block (FillScalar + PackTile chain); skipped: matmul + SFPU-on-PACK SwiGLU activation main loops

### Skip rationales for this batch (per-target, no migration possible under chain v1)
- transformer/sdpa/compute_common.hpp: most blocks use deprecated `acquire_dst()/release_dst()`; modern blocks use `pack_tile<true>` + `llk_pack_reconfig_l1_acc(1)` toggle (L1 accumulation — chain v1 explicitly out of scope); SDPA-specific cumulative `cb_wait_front(base+i)` patterns; matmul / softmax-reduce multi-stage; row-major dst-batch packing.
- experimental/ccl/all_reduce_async/reduction.cpp, llama_reduce_scatter*/reduction.cpp, llama_reduce_scatter_create_heads/reduction.cpp, all_reduce_create_qkv_heads/reduction.cpp: multi-block `add_tiles(...,i,j,p_idx)` accumulation into shared DEST slots over `num_blocks` outer iters (NOT chain-shaped — multiple FPU writes per slot before pack).
- experimental/ccl/strided_reduce_scatter_async/minimal_ring_reduction.cpp: all eltwise sub-blocks use deprecated `acquire_dst()/release_dst()`.
- experimental/ccl/reduce_scatter_minimal_async/ring_reduction.cpp: 3-tensor add (copy + add 3) is multi-stage in-DEST, not chain-shaped.
- experimental/ccl/all_gather_minimal_matmul_async/compute.cpp: `copy_block`, `add_bias_block` use `acquire_dst()`; `add_bias_and_addcmul_block` uses cumulative tile_id + pack-to-same-CB in-place lifecycle (pop-after-pack); matmul main loop out of scope.
- experimental/ccl/rms_allgather/rms_compute.cpp: runtime `subblock_w_volatile`, runtime cumulative `index_subblock_w_offset+w` indexes (non-BlockIter); reduce stage; pack_reconfig_l1_acc inferred via reduce_init state.
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_pre_allgather.cpp: `llk_pack_reconfig_l1_acc(1)` toggled inside per-tile pack loop — chain v1 GAP per eltwise_chain.hpp doc-comment.
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp: rsqrt sub-block is in-place on reduce_result_cb (pop-before-reserve); cos+sin sub-block uses non-standard `tile_regs_acquire(); tile_regs_wait();` together (compute and pack interleaved before commit) — chain enforces standard ordering. ROPE matmul / mul-inplace blocks have non-block-iter `rope_cos_tile_in_head` runtime indices.
- experimental/transformer/dit_layernorm_post_all_gather/layernorm_post_allgather_welford.cpp, dit_layernorm_pre_all_gather/layernorm_pre_allgather_welford.cpp: welford LLK + transpose_wh — separate helpers per HQ skip rules.
- experimental/reduction/fast_reduce_nc/reduce_nc.cpp, deepseek_moe_fast_reduce_nc_reduce.cpp: multi-block `add_tiles(..., k, 0, dst0)` accumulation into single DEST slot over runtime `num_input_tiles_iter`.
- experimental/reduction/integral_image/intimg_compute.cpp: ReadCBGuard / WriteCBGuard RAII state machine + runtime `enable_reload ? cb_acc : cb_start` CB selection + cumsum_tile (specialized SFPU); multi-stage compute (add + copy reload pattern).
- experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate.cpp: topk_local_sort + transpose_wh — already partially migrated for non-topk eltwise stages; remaining stages out of scope.
- experimental/ssm/repeat_and_interleave_eltwise_mul/ssm_eltwise_mul.cpp: every block uses non-standard `tile_regs_acquire(); tile_regs_wait();` together pattern (compute and pack interleaved before commit) + transpose_wh.
- experimental/conv3d/compute.cpp: `add_block_inplace` and `add_bias_inplace` are in-place patterns with pop-before-reserve lifecycle and CB capacity = `num_tiles` (chain reserve-before-pop deadlocks at full CB); fp32_partials inner block is multi-stage (copy + add into same DEST); matmul main loop out of scope.
- experimental/transformer/rotary_embedding/, rotary_embedding_llama/: all use `ACQ()/REL()` macros aliased to `acquire_dst()/release_dst()`.
- experimental/unary_backward/gelu_backward/eltwise_bw_gelu_approx_tanh.cpp: 9-DEST polynomial with non-standard `tile_regs_acquire(); tile_regs_wait();` together pattern + per-tile fill scratch.
- experimental/topk_router_gpt/, deepseek_grouped_gate, moe_grouped_topk: topk_local_sort / topk_merge + transpose_wh — separate helper area.
- experimental/deepseek/{mla/matmul_wo, moe/moe_gate_mm}/compute.cpp, attn_matmul, group_attn_matmul: pure matmul kernels.
- matmul/bmm_large_block_zm_fused_bias_activation*.cpp, bmm.cpp, bmm_large_block_zm.cpp, llama_all_gather_matmul_async/bmm_large_block_zm_fused_bias_activation_gathered.cpp: matmul + bias path encapsulated in `bias_add_helpers.hpp` already; main loop is matmul (skip per HQ rules).
- transformer/sdpa{,_decode,_windowed}/compute/sdpa*.cpp, joint_sdpa.cpp, ring_joint_sdpa.cpp, exp_ring_joint_sdpa.cpp: pure helper-driven, no inline eltwise blocks to migrate at the kernel level.
- data_movement/{transpose,permute,sort,tilize,untilize,concat}: transpose_wh / topk / tilize / untilize — separate helpers per HQ skip rules.
- embedding_backward: reshuffle_rows_tile (specialized SFPU) — separate helper.

## Partial migrations — macro-injection multi-stage compute kernels (binary_ng / where / ternary bcast / row-bcast)

PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp :: migrated: no-activations stride-2 DEST scratch via custom BlockCopyTileStride2{Lhs,Rhs} + LocalBlockSfpuBinary (wraps BINARY_SFPU_OP) + BlockPackTileStride2; skipped: HAS_ACTIVATIONS path (PREPROCESS / PROCESS_POST_ACTIVATIONS macros); skipped: remainder (raw inline via process_sfpu_tiles)
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp :: migrated: no-activations stride-2 chain with BlockCopyTileStride2ScalarRhs (always tile-0 from scalar CB); skipped: HAS_ACTIVATIONS path; skipped: remainder (raw)
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_no_bcast.cpp :: migrated: stride-3 DEST scratch chain — BlockCopyTileStride3Cond + LocalWhereStage (wraps WHERE_TTS/TST FILL_LLK + BINARY_SFPU_OP) + BlockPackTileStride3; skipped: nothing (single migration scope)
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu.cpp :: migrated: per-iter stride-3 chain inside outer freq loop (BCAST_INPUT-conditional wait/pop policy on cb_in0/cb_in1); skipped: bcast operand wait/pop kept at outer scope
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_sfpu_scalar.cpp :: migrated: stride-3 chain with NoWaitNoPop on cb_in1 (scalar), WaitAndPop on cb_in0; skipped: cb_in1 wait at kernel top kept raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_bcast.cpp :: migrated: no-activations inner stage replaced with BlockBinaryFpu<BroadcastDim::None> + BlockPackTile (single-tile per outer iter); skipped: HAS_ACTIVATIONS path raw; skipped: unary_bcast row-broadcast preamble + PREPROCESS stages stay raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_col_bcast.cpp :: migrated: no-activations BlockBinaryFpu+BlockPackTile inner stage (process_tile templated on CBs to flow constexpr CBs through); skipped: HAS_ACTIVATIONS path raw; skipped: unary_bcast col preamble raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_col_bcast.cpp :: migrated: no-activations BlockBinaryFpu+BlockPackTile inner stage (process_tile templated, cb_left/cb_right made constexpr); skipped: HAS_ACTIVATIONS path raw; skipped: unary_bcast row preamble raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_scalar_bcast.cpp :: migrated: no-activations BlockBinaryFpu+BlockPackTile inner stage (process_tile templated); skipped: HAS_ACTIVATIONS path raw; skipped: unary_bcast scalar preamble raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_sfpu_row_bcast.cpp :: migrated: no-activations stride-2 DEST scratch chain (NoWaitNoPop policies — outer scope owns CB lifecycle); skipped: HAS_ACTIVATIONS path raw; skipped: unary_bcast row preamble raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_where_sfpu_row_bcast.cpp :: migrated: stride-3 chain (BlockCopyTileStride3Cond + LocalWhereStage + BlockPackTileStride3) inside outer iteration; skipped: unary_bcast row preamble raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp :: migrated: SFPU stage as eltwise_chain(LoadA/B/C → LocalTernarySfpuStage(wraps TERNARY_SFPU_OP_FUNC) → PackTile<NoReserveNoPush>); skipped: BCAST_A/B/C unary_bcast preambles + outer wait/reserve/push raw
PARTIAL ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp :: migrated: SFPU stage as eltwise_chain(LoadA/B/C → LocalTernarySfpuStage(FillTileTag — runtime scalar_arg captured) → PackTile<NoReserveNoPush>); skipped: BCAST_A/B/C wait/pop raw at outer scope
PARTIAL ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp :: migrated: SFPU stage as eltwise_chain(LoadA/B/C → LocalTernarySfpuStage → PackTile<NoReserveNoPush>) inside templated process_tile; skipped: BCAST_A/B/C outer wait/pop raw

### Skip rationales for kernels NOT migrated this run
SKIPPED:multi-stage-FPU ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_bcast.cpp :: per-iter is multi-stage FPU+DestReuse (mul_tiles → mul_unary → binary_dest_reuse_tiles ELWADD) inside one acquire window; chain helper owns the full acquire/release window so we can't split the multi-stage compute.
SKIPPED:multi-stage-FPU ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_rowbcast.cpp :: same multi-stage FPU+DestReuse pattern as fpu_bcast.
SKIPPED:multi-stage-LLK ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu_bcast.cpp :: per-iter has fill_tile_int + mul_int_tile + mul_int_tile + add_int_tile multi-step compute (no single TERNARY_SFPU_OP_FUNC macro to wrap as one chain element).
SKIPPED:multi-stage-LLK ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_tts_tst.cpp :: TTS/TST conditional copy_tile vs FILL_LLK depending on compile_time_arg(1); needs separate FillLlk + copy chain element shapes per branch (extends LocalWhereStage with per-branch fill — left for follow-up).
SKIPPED:runtime-block-size ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp :: per_core_block_size is a runtime arg; BlockBinaryFpu requires compile-time BlockSize. Same kernel structure as binary_ng but without the CT-arg.
SKIPPED:runtime-block-size ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp :: same — runtime per_core_block_size.

## Run 8 — partial migrations on previously-SKIPPED reduction / moreh kernels (chain-shaped sub-blocks)

Per HQ "Partial Migration": replace only the chain-shaped stages, leave reduce-fold / accumulator
/ welford / power_tile_to_cb / mask_tile / bcast / `_with_dt` bodies on raw LLK or moreh helpers.

PARTIAL ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp :: migrated: blocks::sigmoid (CopyTile + Sigmoid + PackTile streaming chain), blocks::add_bias (BinaryFpu Add — NoWaitNoPop A pre-waited + WaitAndPop B per-tile + PackTile); skipped: process_and_sort_tiles / topk* / normalize_scores / scale (transpose_wh / topk_local_sort / bcast_scalar) — reason: not chain-shaped, bcast, or acquire_dst.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp :: migrated: |x| chain (CopyTile + Abs + PackTile, FP32_DEST_ACC reconfig pre-chain), seed cb_xpowadd (CopyTile + PackTile); skipped: power_tile_to_cb (moreh helper), add_tiles fold into cb_xpowadd (held-CB recurrence) — reason: helper / held-CB.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp :: migrated: seed cb_xpowadd (row_idx==0); skipped: |x| stage (mask_tile when do_mask_h && row_idx==Ht-1; mask not in chain helpers), power_tile_to_cb, add_tiles fold — reason: mask / helper / held-CB.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp :: migrated: seed cb_xpowadd (col_idx==0); skipped: |x| stage (mask_tile branch), power_tile_to_cb, add_tiles fold — reason: mask / helper / held-CB.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp :: migrated: seed cb_cal (row_idx==0 copy chain), post-reduce write-out to cb_y with optional Negative under MINUS_INF; skipped: f(x) preprocessing (mask_tile + binary_max_tile fold — neither in chain helpers), reduce — reason: missing helpers.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp :: migrated: seed cb_cal, post-reduce cb_y write-out (Negative under MINUS_INF); skipped: f(x) (mask_tile + binary_max_tile fold), reduce — reason: same as moreh_norm_h.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp :: migrated: f(x) chain (Copy + UnaryNe|Abs + Negative? + Pack — full IS_ZERO/MINUS_INF combinations), seed cb_cal, post-loop cb_y write-out; skipped: binary_max_tile fold (no DEST-DEST max struct), add_tiles fold — reason: missing helper / held-CB.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/moreh_clip_grad_norm_step1_kernel.cpp :: migrated: seed cb_xpowadd (tile_idx==0 copy chain); skipped: main |x| stage (mask_tile branch), power_tile_to_cb, add_tiles fold — reason: mask / helper / held-CB.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp :: migrated: rsqrt(var+eps) chain (already in repo), seed cb_xmm2sum (inner_idx==0 && j==0 copy chain); skipped: Sum[x]/E[x] stage (mask_tile + bcast), x-mean stage (bcast), Sum[(x-E[x])^2] (mul/add fold), gamma*y+beta (bcast), final write-out — reason: mask / bcast / fold.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp :: migrated: seed cb_dyadd (inner_idx==0 WaitNoPop chain), seed cb_ydyadd (inner_idx==0 chain), final cb_dgamma copy when not lastdim/groupnorm, final cb_dbeta copy when not lastdim/groupnorm; skipped: dy*gamma + masks, x-mean (bcast), y*dy chain (already migrated), add_tiles folds, mask_tile branches, reduces — reason: bcast / mask / held-CB / reduce.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp :: migrated: seed cb_dyadd (wt==0 NoWaitNoPop chain), seed cb_ydyadd (wt==0 chain); skipped: dy*gamma + mask, x-mean (bcast), Wt-loop mul body, post-reduce final write-out (bcast) — reason: bcast / mask / fold / reduce.
PARTIAL ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp :: migrated: seed cb_dyadd (wt==0 WaitNoPop chain), seed cb_ydyadd (wt==0 chain), n*dy mul chain (cb_n_recip_n[0] * cb_dycopy → cb_ndy via BinaryFpu Mul + PackTile); skipped: dy*gamma + mask, x-mean / y mul (bcast), n*dy-Sum[dy] (bcast), x-mean (bcast), final write-out (bcast) — reason: bcast / mask / fold.

