// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include <enchantum/enchantum.hpp>
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

using namespace common;

IntImgDeviceOperation::program_factory_t IntImgDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return IntImgProgramFactory{};
}

void IntImgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    IDevice* device{tensor_args.input_tensor.device()};
    const auto grid_size = device->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x >= EXPECTED_AVAILABLE_CORES_IN_ROW,
        "Expected at least {} cores to be available in the core grid along a row, there are {} instead",
        EXPECTED_AVAILABLE_CORES_IN_ROW,
        grid_size.x);
    TT_FATAL(
        grid_size.y >= EXPECTED_AVAILABLE_CORES_IN_COLUMN,
        "Expected at least {} cores to be available in the core grid along a column, there are {} instead",
        EXPECTED_AVAILABLE_CORES_IN_ROW,
        grid_size.y);
}

void IntImgDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

IntImgDeviceOperation::spec_return_value_t IntImgDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto output_layout{Layout::TILE};
    const auto& input_tensor = tensor_args.input_tensor;
    const auto output_shape{input_tensor.logical_shape()};
    return TensorSpec{output_shape, TensorLayout{input_tensor.dtype(), output_layout, input_tensor.memory_config()}};
}

IntImgDeviceOperation::tensor_return_value_t IntImgDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

IntImgDeviceOperation::invocation_result_t IntImgDeviceOperation::invoke(const Tensor& input_tensor) {
    const auto tile_spec = input_tensor.tensor_spec().tile();
    const Tensor zero_tile =
        ttnn::zeros(Shape{tile_spec.get_tile_shape()}, input_tensor.dtype(), input_tensor.layout());
    return {operation_attributes_t{}, tensor_args_t{input_tensor, zero_tile}};
}

}  // namespace ttnn::operations::experimental::reduction
