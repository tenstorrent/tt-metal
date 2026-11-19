// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_dense.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

void SDPAOperation::validate_on_program_cache_miss(const SDPAParams& attrs, const SDPAInputs& tensors) {
    // resolve_dense_recipe checks the argument rules before the program-cache lookup; recipe_program checks shapes,
    // types and layouts when it builds the program.
    TT_FATAL(attrs.precision.has_value(), "SDPA: the precision recipe is resolved before dispatch");
    (void)tensors;
}

SDPAOperation::spec_return_value_t SDPAOperation::compute_output_specs(
    const SDPAParams& attrs, const SDPAInputs& tensors) {
    return ttnn::operations::transformer::sdpa::detail::dense_recipe_output_spec(attrs, tensors);
}

SDPAOperation::tensor_return_value_t SDPAOperation::create_output_tensors(
    const SDPAParams& attrs, const SDPAInputs& tensors) {
    return create_device_tensor(compute_output_specs(attrs, tensors), tensors.q.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>
SDPAOperation::create_op_performance_model(
    const SDPAParams& args, const SDPAInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const bool has_v = tensor_args.v.has_value();
    const auto& input_tensor_v = tensor_args.v.value_or(tensor_args.k);

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    // Build input tensors list
    Tensors input_tensors = {tensor_args.q, tensor_args.k};
    if (tensor_args.v.has_value()) {
        input_tensors.emplace_back(tensor_args.v.value());
    }

    // Calculate arch specific parameters
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>(
            input_tensors, output_tensor, 0);
    }

    // Get main dimensions for Q*K and softmax(QK^T/sqrt) * V matmuls
    auto q_shape = input_tensor_q.logical_shape();
    auto k_shape = input_tensor_k.logical_shape();
    auto v_shape = input_tensor_v.logical_shape();
    TT_ASSERT(q_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor Q rank != 4");
    TT_ASSERT(k_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor K rank != 4");
    if (!args.use_mla) {
        TT_ASSERT(v_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor V rank != 4");
    }

    bool is_chunked_prefill = args.chunk_start_idx.has_value() || args.chunk_start_idx_tensor.has_value();

    uint32_t batch_size = q_shape[0];
    uint32_t num_heads_q = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t DH = q_shape[3];

    // Compute Sk based on chunked mode
    const uint32_t Sk = (is_chunked_prefill)
                            ? (args.chunk_start_idx.has_value() ? q_shape[2] + args.chunk_start_idx.value()
                                                                : k_shape[2])  // flexible: use K length as upper bound
                            : k_shape[2];

    // Compute DV based on MLA mode
    // Note: For MLA without V, use K's head dimension; otherwise use V's head dimension
    const uint32_t DV = (args.use_mla && !has_v) ? k_shape[3] : v_shape[3];

    TT_ASSERT(q_shape[0] == k_shape[0], "ScaledDotProductAttention perf model: Q and K have unequal batch size!");
    TT_ASSERT(q_shape[3] == k_shape[3], "ScaledDotProductAttention perf model: Q and K have unequal hidden dim!");

    CoreCoord compute_grid_dims = args.program_config.has_value()
                                      ? args.program_config->compute_with_storage_grid_size
                                      : output_tensor.device()->compute_with_storage_grid_size();
    tt::tt_metal::MathFidelity math_fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    int ideal_dev_clock_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        batch_size,
        num_heads_q,
        Sq,
        Sk,
        DH,
        DV,
        args.is_causal,
        math_fidelity,
        compute_grid_dims.x * compute_grid_dims.y);

    // TODO: somehow account for overhead of fused masking and softmax?

    return operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>(
        input_tensors, output_tensor, ideal_dev_clock_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {
Tensor sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<Tensor>& chunk_start_idx_tensor,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& cu_window_seqlens,
    uint32_t windowed_q_token_offset,
    const std::optional<Tensor>& windowed_q_token_offset_tensor,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry,
    bool output_concat_heads,
    const std::optional<Tensor>& attn_mask_block_map,
    std::optional<ttnn::operations::transformer::SDPAPrecision> precision) {
    using OperationType = ttnn::prim::SDPAOperation;
    OperationType::operation_attributes_t attrs{
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = std::move(program_config),
            .is_causal = is_causal,
            .chunk_start_idx = chunk_start_idx,
            .chunk_start_idx_tensor = chunk_start_idx_tensor,
            .compute_kernel_config = compute_kernel_config,
            .use_mla = use_mla,
            .head_dim_v = head_dim_v,
            .sliding_window_size = sliding_window_size,
            .windowed_mode = !cu_window_seqlens.has_value() ? WindowedMode::None
                             : is_causal                    ? WindowedMode::Causal
                                                            : WindowedMode::Bidirectional,
            .windowed_q_token_offset = windowed_q_token_offset,
            .paged_cache_geometry =
                paged_cache_geometry.value_or(ttnn::operations::transformer::PagedCacheGeometryOverride{}),
            .output_concat_heads = output_concat_heads,
            .precision = precision.value_or(ttnn::operations::transformer::SDPAPrecision::STANDARD),
        };
    OperationType::tensor_args_t tensor_args{
            .q = input_tensor_q,
            .k = input_tensor_k,
            .v = input_tensor_v,
            .attn_mask = attn_mask,
            .page_table = page_table_tensor,
            .chunk_start_idx_tensor = chunk_start_idx_tensor,
            .attention_sink = attention_sink,
            .cu_window_seqlens = cu_window_seqlens,
            .windowed_q_token_offset_tensor = windowed_q_token_offset_tensor,
            .attn_mask_block_map = attn_mask_block_map,
        };
    ttnn::operations::transformer::sdpa::detail::resolve_dense_recipe(attrs, tensor_args);
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}
}  // namespace ttnn::prim
