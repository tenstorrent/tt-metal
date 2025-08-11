// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isunique_device_op.hpp"

#include <magic_enum/magic_enum.hpp>

namespace ttnn::operations::experimental::isunique {

using namespace common;

IsUniqueDeviceOperation::program_factory_t IsUniqueDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return IsUniqueProgramFactory{};
}

void IsUniqueDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void IsUniqueDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto& input_rank = input_shape.rank();
    const auto& input_dtype = input_tensor.dtype();

    const auto& index_hint_tensor = tensor_args.index_hint_tensor;
    const auto& index_hint_shape = index_hint_tensor.logical_shape();
    const auto& index_hint_dtype = index_hint_tensor.dtype();

    const auto& dim = args.dim.value_or(FIRST_DIMENSION);

    // input_tensor.logical_shape() == idx_hint_tensor.logical_shape()
    // idx_hint_shape.dtype() is integer
    // dim in rank
    TT_FATAL(!input_tensor.is_sharded(), "");
    TT_FATAL(input_tensor.buffer() != nullptr, "");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "");
    TT_FATAL(input_shape == index_hint_shape, "");
    TT_FATAL(-input_rank <= dim && dim < input_rank, "");
    TT_FATAL(is_integer_format(datatype_to_dataformat_converter(index_hint_dtype)), "")
}

// this method creates an `uint8` tensor serving as a boolean mask, which is "semi-compliant" to Torch's behavior
// although Torch returns a strict `bool` tensor, the `uint8` available in ttnn usually carries the same meaning
// e.g. check out https://docs.pytorch.org/docs/stable/generated/torch.any.html
IsUniqueDeviceOperation::spec_return_value_t IsUniqueDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return {
        tensor_args.input_tensor.logical_shape(),
        {DataType::UINT8, {Layout::ROW_MAJOR}, *args.memory_config},
    };
}

IsUniqueDeviceOperation::tensor_return_value_t IsUniqueDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out) {
        return *tensor_args.optional_out;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

IsUniqueDeviceOperation::invocation_result_t IsUniqueDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& index_hint_tensor,
    const bool& invert,
    const std::optional<int32_t>& dim,
    const OptimalHeuristic& optimal_heuristic,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& first_occurrences_tensor,
    const std::optional<Tensor>& optional_out,
    const QueueId& queue_id) {
    return {
        {invert, dim, optimal_heuristic, memory_config},
        {input_tensor, index_hint_tensor, first_occurrences_tensor, optional_out}};
}

}  // namespace ttnn::operations::experimental::isunique
