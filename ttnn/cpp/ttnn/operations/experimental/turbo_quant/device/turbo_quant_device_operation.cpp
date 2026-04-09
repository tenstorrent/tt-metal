// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "turbo_quant_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::turbo_quant {

void TurboQuantDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.layout() == Layout::TILE, "Input must be in TILE layout");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input must be BFLOAT16, got {}", input.dtype());

    if (attrs.op_type == TurboQuantOpType::BUCKETIZE) {
        TT_FATAL(!attrs.params.empty(), "Bucketize requires at least 1 boundary");
        TT_FATAL(attrs.params.size() <= 15, "Maximum 15 boundaries (4-bit)");
    } else {
        TT_FATAL(attrs.params.size() >= 2, "Gather requires at least 2 centroids");
        TT_FATAL(attrs.params.size() <= 16, "Maximum 16 centroids (4-bit)");
    }
}

TurboQuantDeviceOperation::spec_return_value_t TurboQuantDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(input.layout()), MemoryConfig{}));
}

TurboQuantDeviceOperation::tensor_return_value_t TurboQuantDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(attrs, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::turbo_quant

namespace ttnn::prim {

Tensor turbo_quant_bucketize(const Tensor& input_tensor, const std::vector<float>& boundaries) {
    using Op = ttnn::operations::experimental::turbo_quant::TurboQuantDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            ttnn::operations::experimental::turbo_quant::TurboQuantOpType::BUCKETIZE, boundaries},
        Op::tensor_args_t{input_tensor});
}

Tensor turbo_quant_gather_centroids(const Tensor& input_tensor, const std::vector<float>& centroids) {
    using Op = ttnn::operations::experimental::turbo_quant::TurboQuantDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            ttnn::operations::experimental::turbo_quant::TurboQuantOpType::GATHER_CENTROIDS, centroids},
        Op::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim
