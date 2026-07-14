// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/concat_heads_matmul/device/concat_heads_matmul_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Build the MatmulParams that drive the stock 1D-mcast factory. program_config is auto-derived
// (matches what ttnn.matmul would pick for this [1,1,seq,K] x [K,N] shape).
ttnn::prim::MatmulParams make_params(
    const ConcatHeadsMatmulParams& op, std::optional<ttnn::operations::matmul::MatmulProgramConfig> cfg) {
    ttnn::prim::MatmulParams p{};
    p.program_config = cfg;
    p.bcast_batch = true;
    p.output_mem_config = op.output_mem_config;
    p.output_dtype = op.output_dtype;
    p.compute_kernel_config = op.compute_kernel_config;
    return p;
}

ttnn::prim::MatmulInputs make_inputs(const Tensor& in0, const Tensor& weight) {
    ttnn::prim::MatmulInputs in{};
    in.input_tensors.push_back(in0);
    in.input_tensors.push_back(weight);
    in.optional_input_tensors.emplace_back(std::nullopt);  // no bias (optional<const Tensor> isn't copy-assignable)
    return in;
}

}  // namespace

ConcatHeadsMatmulProgramFactory::cached_program_t ConcatHeadsMatmulProgramFactory::create(
    const ConcatHeadsMatmulParams& operation_attributes,
    const ConcatHeadsMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& attn = tensor_args.attn;
    const auto& weight = tensor_args.weight;

    uint32_t seq = attn.padded_shape()[2];
    uint32_t K = attn.padded_shape()[1] * attn.padded_shape()[3];  // nh * hd
    // Build-time-only view: concat-heads is the contiguous tile order for seq<=1 tile, so attn's
    // buffer reinterpreted as [1,1,seq,K] IS the concat result. NOT a traced op.
    ttnn::Shape in0_shape({1, 1, seq, K});
    Tensor in0 = tt::tt_metal::view(attn, in0_shape, in0_shape);

    // Use the caller-supplied tuned config if given (matches the denoise O-proj exactly); else
    // auto-derive what ttnn.matmul would pick for this shape.
    auto cfg = operation_attributes.program_config.has_value() ? operation_attributes.program_config.value()
                                                               : ttnn::operations::matmul::get_program_config(
                                                                     in0,
                                                                     weight,
                                                                     /*transpose_a=*/false,
                                                                     /*transpose_b=*/false,
                                                                     /*bias_single_tile_size=*/0,
                                                                     make_params(operation_attributes, std::nullopt));
    TT_FATAL(
        std::holds_alternative<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(cfg),
        "concat_heads_matmul: expected a 1D-mcast program config for this shape");

    std::vector<Tensor> out_vec = {tensor_return_value};

    // Build the program via the stock 1D-mcast matmul helper. Main refactored the matmul
    // program-factory API: the old static ::create returning a cached_program_t was replaced
    // by this free helper that populates a caller-owned Program and returns a CachedProgram.
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;
    auto mm = ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper(
        program,
        in0,
        {weight},
        /*bias=*/std::nullopt,
        out_vec,
        /*broadcast_batch=*/true,
        operation_attributes.compute_kernel_config,
        cfg,
        /*untilize_out=*/false,
        fused_op_signaler,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/std::nullopt);

    return cached_program_t{
        std::move(mm.program),
        ConcatHeadsMatmulSharedVariables{
            std::move(mm.shared_variables),
            cfg,
            operation_attributes.output_mem_config,
            operation_attributes.output_dtype,
            operation_attributes.compute_kernel_config}};
}

void ConcatHeadsMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatHeadsMatmulParams& operation_attributes,
    const ConcatHeadsMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& sv = cached_program.shared_variables;
    const auto& attn = tensor_args.attn;
    const auto& weight = tensor_args.weight;

    uint32_t seq = attn.padded_shape()[2];
    uint32_t K = attn.padded_shape()[1] * attn.padded_shape()[3];
    ttnn::Shape in0_shape({1, 1, seq, K});
    Tensor in0 = tt::tt_metal::view(attn, in0_shape, in0_shape);

    ConcatHeadsMatmulParams op = operation_attributes;
    op.output_mem_config = sv.output_mem_config;
    op.output_dtype = sv.output_dtype;
    op.compute_kernel_config = sv.compute_kernel_config;
    auto params = make_params(op, sv.program_config);
    auto inputs = make_inputs(in0, weight);
    std::vector<Tensor> out_vec = {tensor_return_value};

    // New override API: takes the Program + shared_variables directly (the old
    // cached_program_t::proxy indirection was removed in the matmul refactor).
    ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments(
        cached_program.program, sv.mm_shared, params, inputs, out_vec);
}

}  // namespace ttnn::experimental::prim
