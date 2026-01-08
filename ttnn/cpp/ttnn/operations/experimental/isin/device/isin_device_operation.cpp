// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::isin {
// constexpr int32_t FIRST_DIMENSION = 0;
constexpr auto OUTPUT_TENSOR_DATA_TYPE = DataType::UINT32;
constexpr auto OUTPUT_TENSOR_LAYOUT = Layout::ROW_MAJOR;
// constexpr uint32_t OUTPUT_TENSOR_RANK = 1;
// using namespace common;
using namespace tt;
using namespace tt::tt_metal;

IsInDeviceOperation::program_factory_t IsInDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return IsInProgramFactory{};
}

void IsInDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void IsInDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& elements = tensor_args.elements_tensor;
    const auto& test_elements = tensor_args.test_elements_tensor;

    TT_FATAL(
        elements.dtype() == test_elements.dtype(),
        "Elements and test elements tensors don't share the same dtype (elements: {}, test_elements: {})",
        enchantum::to_string(elements.dtype()),
        enchantum::to_string(test_elements.dtype()));

    TT_FATAL(!elements.is_sharded(), "Elements tensor is sharded");
    TT_FATAL(elements.buffer() != nullptr, "Elements tensor's buffer is null");
    TT_FATAL(elements.storage_type() == StorageType::DEVICE, "Elements tensor is not on a device");

    TT_FATAL(!test_elements.is_sharded(), "Test elements tensor is sharded");
    TT_FATAL(test_elements.buffer() != nullptr, "Test elements tensor's buffer is null");
    TT_FATAL(test_elements.storage_type() == StorageType::DEVICE, "Test elements tensor is not on a device");
}

// this method creates an `uint8` tensor serving as a boolean mask, which is "semi-compliant" to Torch's behavior
// although Torch returns a strict `bool` tensor, the `uint8` available in ttnn usually carries the same meaning
// e.g. check out https://docs.pytorch.org/docs/stable/generated/torch.any.html
IsInDeviceOperation::spec_return_value_t IsInDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return {
        Shape{tensor_args.elements_tensor.logical_volume()},
        {OUTPUT_TENSOR_DATA_TYPE, {OUTPUT_TENSOR_LAYOUT}, tensor_args.elements_tensor.memory_config()},
    };
}

IsInDeviceOperation::tensor_return_value_t IsInDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out) {
        TT_FATAL(
            tensor_args.optional_out->dtype() == OUTPUT_TENSOR_DATA_TYPE,
            "Preallocated output should be of uint32 dtype");
        return *tensor_args.optional_out;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.elements_tensor.device());
}

}  // namespace ttnn::operations::experimental::isin

namespace ttnn::prim {
ttnn::operations::experimental::isin::tensor_return_value_t isin(
    const Tensor& elements,
    const Tensor& test_elements,
    uint32_t single_fetch_subchunk_size,
    bool assume_unique,
    bool invert,
    const std::optional<Tensor>& optional_out) {
    using OperationType = ttnn::operations::experimental::isin::IsInDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .assume_unique = assume_unique,
        .invert = invert,
        .single_fetch_subchunk_size = single_fetch_subchunk_size,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .elements_tensor = elements,
        .test_elements_tensor = test_elements,
        .optional_out = optional_out,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
