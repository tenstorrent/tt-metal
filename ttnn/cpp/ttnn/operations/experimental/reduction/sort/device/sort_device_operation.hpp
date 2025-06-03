// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_program_factory.hpp"
#include "sort_device_operation_types.hpp"

#include "ttnn/decorators.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::operations::experimental::reduction::sort {

struct SortDeviceOperation {
    using operation_attributes_t = sort::operation_attributes_t;
    using tensor_args_t = sort::tensor_args_t;
    using spec_return_value_t = sort::spec_return_value_t;
    using tensor_return_value_t = sort::tensor_return_value_t;
    using program_factory_t = std::variant<
        sort::program::SortProgramFactorySingleRowSingleCore,
        sort::program::SortProgramFactorySingleRowMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const int8_t dim,
        const bool descending,
        const bool stable,
        const MemoryConfig& output_memory_config,
        const std::vector<std::optional<Tensor>>& output_tensors);
};

}  // namespace ttnn::operations::experimental::reduction::sort

namespace ttnn::prim {
constexpr auto sort = ttnn::
    register_operation<"ttnn::prim::sort", ttnn::operations::experimental::reduction::sort::SortDeviceOperation>();
}  // namespace ttnn::prim
