// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "materialize_device_operation.hpp"
#include "ttnn/operations/eltwise/lazy/expression.hpp"

namespace ttnn::operations::fused {

MaterializeDeviceOperation::program_factory_t MaterializeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void MaterializeDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

void MaterializeDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

MaterializeDeviceOperation::spec_return_value_t MaterializeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    namespace metal = tt::tt_metal;

    // TODO broadcast
    return TensorSpec(
        tensor_args.input_tensors.front().logical_shape(),
        metal::TensorLayout(
            operation_attributes.dtype, metal::PageConfig(Layout::TILE), operation_attributes.memory_config));
}

MaterializeDeviceOperation::tensor_return_value_t MaterializeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensors.front().device());
}

std::tuple<MaterializeDeviceOperation::operation_attributes_t, MaterializeDeviceOperation::tensor_args_t>
MaterializeDeviceOperation::invoke(lazy::FunctionView expression) {
    namespace metal = tt::tt_metal;

    std::map<tt::CBIndex, std::size_t> inputs;
    ttsl::SmallVector<lazy::Param> params;
    std::vector<Tensor> input_tensors;

    lazy::traverse(
        ttsl::overloaded{
            [](const Tensor&) {},
            [&](lazy::FunctionView function) {
                for (const auto argument : function.arguments()) {
                    if (auto tensor = argument.tensor()) {
                        inputs.emplace(argument.cb_index(), input_tensors.size());
                        input_tensors.push_back(*tensor);
                    }
                }

                const auto sub_params = function.params();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }},
        expression);

    const auto& input_tensor = input_tensors.front();
    auto device = input_tensor.device();
    auto memory_config = input_tensor.memory_config();
    auto worker_cores =
        device->worker_cores(metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());

    return {
        operation_attributes_t{
            .compute_kernel_source = lazy::to_compute_kernel_string(expression),
            .circular_buffers = expression.circular_buffers(),
            .inputs = std::move(inputs),
            .output = expression.cb_index(),
            .params = std::move(params),
            .memory_config = std::move(memory_config),
            .dtype = expression.dtype(),
            .worker_grid = std::move(worker_cores)},
        tensor_args_t{.input_tensors = std::move(input_tensors)}};
}

}  // namespace ttnn::operations::fused
