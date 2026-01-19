// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/madd/device/madd_device_operation.hpp"

namespace ttnn::operations::madd {

static std::array<uint32_t, 4> get_input_shape(const Tensor& x) {
    const Shape& x_shape = x.logical_shape();
    return {x_shape[0], x_shape[1]};
}

spec_return_value_t MAddOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& a = tensor_args.a;
    const Tensor& b = tensor_args.b;
    const std::array<uint32_t, 4> a_shape = get_input_shape(a);
    const std::array<uint32_t, 4> b_shape = get_input_shape(b);

    const uint32_t a_h = a_shape[0];
    const uint32_t b_w = b_shape[1];

    const ttnn::Shape output_shape = ttnn::Shape({a_h, b_w});

    const tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;  // upsample only outputs row major data

    const tt::tt_metal::DataType output_data_type = a.dtype();

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(output_data_type, tt::tt_metal::PageConfig(output_layout), args.output_mem_config));
}

tensor_return_value_t MAddOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.a.device());
}

static void validate_operand(const Tensor& x) {
    TT_FATAL(x.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(x.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(x.layout() == tt::tt_metal::Layout::TILE, "Operands must be tiled!");
    TT_FATAL(
        x.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is supported for tiled input");
    TT_FATAL(x.padded_shape() == x.logical_shape(), "Only tile aligned tile input is currently supported");

    TT_FATAL(x.dtype() == tt::tt_metal::DataType::FLOAT32, "Only float32 type supported, found {}", x.dtype());
}

void MAddOperation::validate_on_program_cache_miss(
    [[maybe_unused]] const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& c = tensor_args.c;
    validate_operand(a);
    validate_operand(b);
    validate_operand(c);

    const std::array<uint32_t, 4> a_shape = get_input_shape(a);
    const std::array<uint32_t, 4> b_shape = get_input_shape(b);
    const std::array<uint32_t, 4> c_shape = get_input_shape(c);

    const uint32_t a_h = a_shape[0];
    const uint32_t a_w = a_shape[1];
    const uint32_t b_h = b_shape[0];
    const uint32_t b_w = b_shape[1];
    const uint32_t c_h = c_shape[0];
    const uint32_t c_w = c_shape[1];

    TT_FATAL(a_w == b_h, "Matrix multiplication shape mismatch: A width {} must equal B height {}", a_w, b_h);
    TT_FATAL(b_w == c_w, "Matrix multiplication shape mismatch: B width {} must equal C width {}", b_w, c_w);
    TT_FATAL(
        a_h == c_h,
        "Matrix multiplication shape mismatch: A height {} must equal C height {}",
        a_h,
        c_h);  // Not really, broadcasting should be enabled
}

void MAddOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

MAddOperation::program_factory_t MAddOperation::select_program_factory(
    [[maybe_unused]] const operation_attributes_t& args, [[maybe_unused]] const tensor_args_t& tensor_args) {
    return program::MAddProgramFactory{};
}

}  // namespace ttnn::operations::madd

namespace ttnn::prim {
ttnn::Tensor madd(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& c,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::madd::MAddOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_mem_config, compute_kernel_config},
        OperationType::tensor_args_t{a, b, c});
}  // namespace ttnn::prim
