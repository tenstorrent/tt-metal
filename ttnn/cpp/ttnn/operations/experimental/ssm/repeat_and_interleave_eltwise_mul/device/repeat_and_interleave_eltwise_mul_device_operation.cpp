// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::experimental::ssm::repeat_mul {

namespace {
constexpr uint32_t HIDDEN_SIZE = 5120;
}

RepeatAndInterleaveEltwiseMulDeviceOperation::program_factory_t
RepeatAndInterleaveEltwiseMulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::RepeatAndInterleaveEltwiseMulProgramFactory{};
}

void RepeatAndInterleaveEltwiseMulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RepeatAndInterleaveEltwiseMulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.a;
    const auto& input_tensor_b = tensor_args.b;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.memory_config;
    auto output_datatype = args.dtype;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
        output_datatype = preallocated_output_tensor->dtype();
    }

    TT_FATAL(
        (input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE),
        "Inputs to ssm_eltwise_mul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to ssm_eltwise_mul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to ssm_eltwise_mul need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to ssm_eltwise_mul need to be on the same device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input b!");
    TT_FATAL(
        input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");
    TT_FATAL(
        input_tensor_b.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_b.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input b!");

    TT_FATAL(
        out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Unsupported memory layout for output!");
    TT_FATAL(
        output_datatype == tt::tt_metal::DataType::BFLOAT16 || output_datatype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Batch not supported for input a!");
    TT_FATAL((bshape[0] == 1 and bshape[1] == 1), "Batch not supported for input b!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input a!");
    TT_FATAL((bshape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input b!");
    TT_FATAL((ashape[2] == bshape[2]), "Num of users must match in both of the input!");
    TT_FATAL((ashape[3] != bshape[3]), "Use eltwise mul for same size inputs!");
    TT_FATAL(
        (ashape[3] == TILE_WIDTH || ashape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input a width must be 32 or 32*5120!");
    TT_FATAL(
        (bshape[3] == HIDDEN_SIZE || bshape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input b width must be 32 or 32*5120!");

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocated output tensor is used, RepeatAndInterleaveEltwiseMul operation requires its shape to "
            "match the computed shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        TT_FATAL(
            (preallocated_output_tensor.value().layout() == Layout::TILE),
            "RepeatAndInterleaveEltwiseMul operation requires output tensor to be in Tile layout.");
    }
}

TensorSpec RepeatAndInterleaveEltwiseMulDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor_a = tensor_args.a;
    const auto& shape_a = input_tensor_a.padded_shape();
    Shape output_shape({shape_a[0], shape_a[1], shape_a[2], TILE_WIDTH * HIDDEN_SIZE});
    return TensorSpec(output_shape, TensorLayout(args.dtype, PageConfig(Layout::TILE), args.memory_config));
}

Tensor RepeatAndInterleaveEltwiseMulDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.a.device());
}

tt::stl::hash::hash_t RepeatAndInterleaveEltwiseMulDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.a;
    const auto& input_tensor_b = tensor_args.b;
    const auto& input_shape_a = input_tensor_a.padded_shape();
    const auto& input_shape_b = input_tensor_b.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);

    // Determine compile-time defines based on shapes
    bool repeat_in0 = (input_shape_a[-1] == TILE_WIDTH);
    bool repeat_interleave_in1 = (input_shape_b[-1] == HIDDEN_SIZE);

    operation::Hash hash = operation::hash_operation<RepeatAndInterleaveEltwiseMulDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor_a.dtype(),
        input_tensor_b.dtype(),
        input_tensor_a.memory_config(),
        input_tensor_b.memory_config(),
        args.memory_config,
        args.math_fidelity,
        input_shape_a.volume(),
        input_shape_b.volume(),
        repeat_in0,
        repeat_interleave_in1);

    return hash;
}

}  // namespace ttnn::operations::experimental::ssm::repeat_mul

namespace ttnn::prim {

ttnn::operations::experimental::ssm::repeat_mul::RepeatAndInterleaveEltwiseMulDeviceOperation::tensor_return_value_t
repeat_and_interleave_eltwise_mul(
    const Tensor& a,
    const Tensor& b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<MathFidelity> math_fidelity,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::experimental::ssm::repeat_mul::RepeatAndInterleaveEltwiseMulDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(a.memory_config()),
        .dtype = dtype.value_or(a.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{.a = a, .b = b, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
