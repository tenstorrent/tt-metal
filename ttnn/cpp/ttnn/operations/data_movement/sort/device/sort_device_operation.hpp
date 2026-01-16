// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_program_factory.hpp"
#include "sort_device_operation_types.hpp"

#include "ttnn/decorators.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::operations::data_movement::sort {

struct SortDeviceOperation {
    using operation_attributes_t = SortParams;
    using tensor_args_t = SortInputs;
    using spec_return_value_t = sort::spec_return_value_t;
    using tensor_return_value_t = sort::tensor_return_value_t;
    using program_factory_t = std::variant<
        sort::program::SortProgramFactorySingleRowSingleCore,
        sort::program::SortProgramFactoryCrossCoreDataExchange,
        sort::program::SortProgramFactorySingleRowMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement::sort

namespace ttnn::prim {
ttnn::operations::data_movement::sort::SortDeviceOperation::tensor_return_value_t sort(
    const Tensor& input_tensor,
    int8_t dim,
    bool descending,
    bool stable,
    const MemoryConfig& output_memory_config,
    const std::vector<std::optional<Tensor>>& output_tensors);
}  // namespace ttnn::prim
