// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "plusone_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "plusone_device_operation_types.hpp"

namespace ttnn::operations::experimental::plusone {

struct PlusOneDeviceOperation {
    using operation_attributes_t = plusone::operation_attributes_t;
    using tensor_args_t = plusone::tensor_args_t;
    using spec_return_value_t = plusone::spec_return_value_t;
    using tensor_return_value_t = plusone::tensor_return_value_t;
    using program_factory_t = std::variant<program::PlusOneProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
        bool skip_negative_entries = false);
};

}  // namespace ttnn::operations::experimental::plusone

namespace ttnn::prim {
constexpr auto plus_one =
    ttnn::register_operation<"ttnn::prim::plus_one", ttnn::operations::experimental::plusone::PlusOneDeviceOperation>();
}  // namespace ttnn::prim
