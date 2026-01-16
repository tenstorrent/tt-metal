// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::transformer::sdpa_windowed {

constexpr uint32_t cu_window_seqlens_nelements = 4096;
// [INFO] 1024 is large enough for 300DPI images and it was increased to 4096 to support larger images.
//        Even larger number of elements can be supported if needed.
static_assert(cu_window_seqlens_nelements == 4096, "cu_window_seqlens_nelements must be 4096");

struct WindowedScaledDotProductAttentionDeviceOperation {
    using operation_attributes_t = SdpaWindowedParams;
    using tensor_args_t = SdpaWindowedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::WindowedSDPAProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::transformer::sdpa_windowed

namespace ttnn::prim {
ttnn::operations::transformer::sdpa_windowed::WindowedScaledDotProductAttentionDeviceOperation::tensor_return_value_t
windowed_scaled_dot_product_attention(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& cu_window_seqlens,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::prim
