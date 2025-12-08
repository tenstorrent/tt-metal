// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation_types.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"

namespace ttnn::operations::data_movement::indexed_fill {

struct IndexedFillDeviceOperation {
    using operation_attributes_t = indexed_fill::operation_attributes_t;
    using tensor_args_t = indexed_fill::tensor_args_t;
    using spec_return_value_t = indexed_fill::spec_return_value_t;
    using tensor_return_value_t = indexed_fill::tensor_return_value_t;
    using program_factory_t = std::variant<indexed_fill::program::IndexedFillProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& batch_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        int64_t dim);
};

}  // namespace ttnn::operations::data_movement::indexed_fill

namespace ttnn::prim {
constexpr auto indexed_fill = ttnn::register_operation<
    "ttnn::prim::indexed_fill",
    ttnn::operations::data_movement::indexed_fill::IndexedFillDeviceOperation>();
}  // namespace ttnn::prim
