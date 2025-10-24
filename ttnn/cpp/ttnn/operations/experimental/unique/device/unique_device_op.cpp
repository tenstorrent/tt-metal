// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unique_device_op.hpp"

#include <enchantum/enchantum.hpp>
#include "ttnn/operations/experimental/unique/device/unique_device_op_types.hpp"
#include "ttnn/operations/experimental/unique/unique_common.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::unique {

using namespace common;

UniqueDeviceOperation::program_factory_t UniqueDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    // there might be different factories for `return_counts` and `dim`
    return UniqueProgramFactory{};
}

void UniqueDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void UniqueDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(!input_tensor.is_sharded(), "Input tensor is sharded");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor's buffer is null");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor is not on a device");
}

// this method creates an `uint8` tensor serving as a boolean mask, which is "semi-compliant" to Torch's behavior
// although Torch returns a strict `bool` tensor, the `uint8` available in ttnn usually carries the same meaning
// e.g. check out https://docs.pytorch.org/docs/stable/generated/torch.any.html
UniqueDeviceOperation::spec_return_value_t UniqueDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return {
        {
            Shape{tensor_args.input_tensor.logical_volume()},
            {tensor_args.input_tensor.dtype(),
             {OUTPUT_TENSOR_LAYOUT},
             args.memory_config.has_value() ? *(args.memory_config) : tensor_args.input_tensor.memory_config()},
        },
        {
            Shape{1},
            {DataType::UINT32,
             {Layout::ROW_MAJOR},
             args.memory_config.has_value() ? *(args.memory_config) : tensor_args.input_tensor.memory_config()},
        },

    };
}

UniqueDeviceOperation::tensor_return_value_t UniqueDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.input_tensor.device())};
}

UniqueDeviceOperation::invocation_result_t UniqueDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& first_occurrences_tensor,
    const uint32_t& single_fetch_subchunk_size,
    const bool& sorted,
    const bool& return_inverse,
    const bool& return_counts,
    const std::optional<int32_t>& dim,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        {single_fetch_subchunk_size, sorted, return_inverse, return_counts, dim, memory_config},
        {input_tensor, first_occurrences_tensor}};
}

}  // namespace ttnn::operations::experimental::unique
