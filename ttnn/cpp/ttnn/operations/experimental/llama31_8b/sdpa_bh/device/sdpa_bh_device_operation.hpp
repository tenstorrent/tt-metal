// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

struct SdpaBhProgramFactory;

struct SdpaBhOperation {
    using operation_attributes_t = ttnn::prim::SDPAParams;
    using tensor_args_t = ttnn::prim::SDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SdpaBhProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct SdpaBhProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SdpaBhOperation::operation_attributes_t& operation_attributes,
        const SdpaBhOperation::tensor_args_t& tensor_args,
        SdpaBhOperation::tensor_return_value_t& tensor_return_value);
};

bool sdpa_bh_supported(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_sdpa_bh(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool output_concat_heads = false);
}  // namespace ttnn::prim
