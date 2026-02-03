// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gather_device_operation_types.hpp"
#include "gather_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::prim {

struct GatherDeviceOperation {
    using operation_attributes_t = GatherParams;
    using tensor_args_t = GatherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<GatherProgramFactorySingleRowSingleCore, GatherProgramFactorySingleRowMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

Tensor gather(
    const Tensor& input_tensor,
    int8_t dim,
    const Tensor& input_index_tensor,
    bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensors,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim
