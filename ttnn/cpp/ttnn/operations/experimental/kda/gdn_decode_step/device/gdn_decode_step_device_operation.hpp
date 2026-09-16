// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>
#include <optional>

#include "gdn_decode_step_device_operation_types.hpp"
#include "gdn_decode_step_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepOperation {
    using operation_attributes_t = GdnDecodeStepParams;
    using tensor_args_t = GdnDecodeStepInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GdnDecodeStepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor gdn_decode_step(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const Tensor& state,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    tt::tt_metal::DataType output_dtype,
    const std::optional<Tensor>& conv_hist = std::nullopt,
    const std::optional<Tensor>& conv_taps = std::nullopt,
    uint32_t qkvz_dim = 0);

}  // namespace ttnn::experimental::prim
