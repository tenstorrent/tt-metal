// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>

#include "gdn_spec_step_device_operation_types.hpp"
#include "gdn_spec_step_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct GdnSpecStepOperation {
    using operation_attributes_t = GdnSpecStepParams;
    using tensor_args_t = GdnSpecStepInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GdnSpecStepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor gdn_spec_step(
    const Tensor& qkvzab,
    const Tensor& win_a,
    const Tensor& win_b,
    const Tensor& ring,
    const Tensor& ctrl,
    const Tensor& taps,
    const Tensor& dt_bias,
    const Tensor& neg_exp_A,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t conv_kernel,
    uint32_t qkvz_dim,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    uint32_t hnew_depth,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    tt::tt_metal::DataType output_dtype);

}  // namespace ttnn::experimental::prim
