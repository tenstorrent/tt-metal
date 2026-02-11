// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "moe_device_operation_types.hpp"
#include "moe_program_factory.hpp"

namespace ttnn::operations::experimental::moe {

struct MoEDeviceOperation {
    using operation_attributes_t = moe::operation_attributes_t;
    using tensor_args_t = moe::tensor_args_t;
    using spec_return_value_t = moe::spec_return_value_t;
    using tensor_return_value_t = moe::tensor_return_value_t;
    using program_factory_t = std::variant<program::MoEProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& w0_w1_tensor,
        const Tensor& w2_tensor,
        const Tensor& output_tensor,
        const uint32_t num_experts,
        const uint32_t layer_id,
        const tt::tt_metal::CoreRangeSet& output_shard_core_ranges);
};

}  // namespace ttnn::operations::experimental::moe

namespace ttnn::prim {
constexpr auto moe =
    ttnn::register_operation<"ttnn::prim::moe", ttnn::operations::experimental::moe::MoEDeviceOperation>();
}  // namespace ttnn::prim
