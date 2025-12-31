// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_post_all_gather_device_operation_types.hpp"
#include "layernorm_post_all_gather_program_factory.hpp"
#include "layernorm_post_all_gather_2d_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include <optional>
#include <variant>

namespace ttnn::operations::normalization::layernorm_post_all_gather {

struct LayerNormPostAllGatherDeviceOperation {
    using operation_attributes_t = layernorm_post_all_gather::operation_attributes_t;
    using tensor_args_t = layernorm_post_all_gather::tensor_args_t;
    using spec_return_value_t = layernorm_post_all_gather::spec_return_value_t;
    using tensor_return_value_t = layernorm_post_all_gather::tensor_return_value_t;
    using program_factory_t =
        std::variant<program::LayerNormPostAllGatherProgramFactory, program::LayerNormPostAllGather2DProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const Tensor& stats,
        const std::optional<Tensor>& gamma,
        const std::optional<Tensor>& beta,
        LayerNormDistributedType norm_type,
        float eps,
        const tt::tt_metal::MemoryConfig& memory_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        const std::optional<tt::tt_metal::DataType>& output_dtype,
        const std::optional<bool>& use_2d_core_grid,
        const LayerNormDistributedDefaultProgramConfig& program_config);
};

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather

namespace ttnn::prim {
constexpr auto layernorm_post_all_gather = ttnn::register_operation<
    "ttnn::prim::layernorm_post_all_gather",
    ttnn::operations::normalization::layernorm_post_all_gather::LayerNormPostAllGatherDeviceOperation>();
}  // namespace ttnn::prim
