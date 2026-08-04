// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_device_operation.hpp"

#include <tt-metalium/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

// Phase 4a fills in the real L1-fit / core-count selection between the three factories.
GatherCodegenDeviceOperation::program_factory_t GatherCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GatherCodegenProgramFactoryInterleaved{};
}

// supported_by_codegen() (gather_codegen_supported.hpp) gates entry into this prim; phase 4a
// fills in the correctness checks that belong here.
void GatherCodegenDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {
}

GatherCodegenDeviceOperation::spec_return_value_t GatherCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    const auto output_shape = tensor_args.input_index_tensor.logical_shape();
    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout(
            tensor_args.input_tensor.dtype(),
            PageConfig(tensor_args.input_tensor.layout()),
            attributes.output_mem_config));
}

GatherCodegenDeviceOperation::tensor_return_value_t GatherCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<GatherCodegenDeviceOperation::tensor_return_value_t>
GatherCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t&, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
}

Tensor gather_codegen(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<GatherCodegenDeviceOperation>(
        GatherCodegenParams{dim, sparse_grad, output_memory_config, sub_core_grids},
        GatherCodegenInputs{input_tensor, input_index_tensor, output_tensor});
}

}  // namespace ttnn::prim
