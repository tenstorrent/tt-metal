// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm::hc_sum_reduce {

HCSumReduceDeviceOperation::program_factory_t HCSumReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::HCSumReduceProgramFactory{};
}

void HCSumReduceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void HCSumReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input;
    TT_FATAL((input_tensor_a.layout() == Layout::TILE), "Inputs to ssm_1d_sum_reduce must be tilized");

    // TODO: Uplift to support mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to ssm_1d_sum_reduce need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr, "Operands to ssm_1d_sum_reduce need to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");

    TT_FATAL(
        args.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Unsupported memory layout for output!");
    TT_FATAL(
        args.dtype == tt::tt_metal::DataType::BFLOAT16 || args.dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    constexpr uint32_t latent = 32;
    const auto& ashape = input_tensor_a.padded_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Dim 1 and 2 are expected to be 1 in input a!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Batch size must be divisible by 32 for input a!");
    TT_FATAL((ashape[3] % TILE_WIDTH == 0), "Final dim must be a multiple of 32!");
    TT_FATAL(((ashape[3] / TILE_WIDTH) % latent == 0), "Final dim/TILE_SIZE must be a multiple of latent size!");
}

TensorSpec HCSumReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    constexpr uint32_t latent = 32;
    const auto& input_tensor_a = tensor_args.input;
    const auto& shape_a = input_tensor_a.padded_shape();
    Shape output_shape({shape_a[0], shape_a[1], shape_a[2], shape_a[3] / latent});
    return TensorSpec(output_shape, TensorLayout(args.dtype, PageConfig(Layout::TILE), args.memory_config));
}

Tensor HCSumReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t HCSumReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<HCSumReduceDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        args.math_fidelity,
        input_shape.volume());

    return hash;
}

}  // namespace ttnn::operations::experimental::ssm::hc_sum_reduce

namespace ttnn::prim {

ttnn::operations::experimental::ssm::hc_sum_reduce::HCSumReduceDeviceOperation::tensor_return_value_t hc_sum_reduce(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<MathFidelity> math_fidelity) {
    using OperationType = ttnn::operations::experimental::ssm::hc_sum_reduce::HCSumReduceDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(input.memory_config()),
        .dtype = dtype.value_or(input.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
