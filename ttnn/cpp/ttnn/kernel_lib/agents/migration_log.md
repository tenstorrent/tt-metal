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
