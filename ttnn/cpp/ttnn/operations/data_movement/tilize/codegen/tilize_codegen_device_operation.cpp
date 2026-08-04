// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include "tilize_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeCodegenDeviceOperation::program_factory_t TilizeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return TilizeCodegenProgramFactory{};
}

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // The native op's structural preconditions (TilizeDeviceOperation::validate_on_program_cache_miss)
    // come first: supported_by_codegen() answers over layout/dtype/memory config, all of which answer
    // for a host or deallocated tensor too, so without these the first thing to notice would be a
    // null device/buffer dereference in the factory.
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    // Native's stick-size invariant: the reader moves whole TILE_W-wide element groups, which is
    // only byte-addressable for an even element size.
    TT_FATAL(
        (input_tensor.padded_shape()[-1] * input_tensor.element_size()) % 2 == 0, "Stick size must be divisible by 2");

    TT_FATAL(
        supported_by_codegen(operation_attributes, tensor_args),
        "tilize: inputs not supported by the codegen implementation");
}

TilizeCodegenDeviceOperation::spec_return_value_t TilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(
            operation_attributes.output_dtype, PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

TilizeCodegenDeviceOperation::tensor_return_value_t TilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

Tensor tilize_codegen(const Tensor& input_tensor, const TilizeCodegenParams& params) {
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(params, TilizeCodegenInputs{input_tensor});
}

}  // namespace ttnn::prim
