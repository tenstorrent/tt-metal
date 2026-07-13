// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::examples {

BhDramReadDeviceOperation::program_factory_t BhDramReadDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DramBankCore{};
}

void BhDramReadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "bh_dram_read: input tensor must be on device");
    TT_FATAL(input_tensor.memory_config().is_dram(), "bh_dram_read: input tensor must be in DRAM");
    TT_FATAL(!input_tensor.memory_config().is_sharded(), "bh_dram_read: input tensor must be DRAM-interleaved");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "bh_dram_read: input tensor must be tilized (reads are tile-paged)");
}

BhDramReadDeviceOperation::spec_return_value_t BhDramReadDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Read-only: the output spec mirrors the input.
    return tensor_args.input_tensor.tensor_spec();
}

BhDramReadDeviceOperation::tensor_return_value_t BhDramReadDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Read-only: alias the input tensor as the output; no new allocation.
    return tensor_args.input_tensor;
}

}  // namespace ttnn::operations::examples

namespace ttnn::prim {
ttnn::operations::examples::BhDramReadDeviceOperation::tensor_return_value_t bh_dram_read(const Tensor& input_tensor) {
    using OperationType = ttnn::operations::examples::BhDramReadDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
