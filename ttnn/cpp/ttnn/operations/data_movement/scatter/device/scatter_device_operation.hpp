// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/decorators.hpp"

#include "../scatter_enums.hpp"
#include "scatter_device_operation_types.hpp"

#include "scatter_program_factory.hpp"
#include "scatter_reduce_bfloat16_program_factory.hpp"

namespace ttnn::operations::data_movement::scatter {

using namespace tt::tt_metal;

struct ScatterDeviceOperation {
    using operation_attributes_t = scatter::operation_attributes_t;
    using tensor_args_t = scatter::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<scatter::ScatterProgramFactory, scatter::ScatterReduceBfloat16ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement::scatter

namespace ttnn::prim {
ttnn::Tensor scatter(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const MemoryConfig& output_memory_config,
    const ttnn::operations::data_movement::scatter::ScatterReductionType& opt_reduction,
    const std::optional<CoreRangeSet>& sub_core_grid);
}  // namespace ttnn::prim
