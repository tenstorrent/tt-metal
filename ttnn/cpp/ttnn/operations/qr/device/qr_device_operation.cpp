// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qr_device_operation.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::qr {

namespace {

constexpr uint32_t kMaxTileDim = 32;

}  // namespace

void QrDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "QR: input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "QR: input tensor must be allocated in a device buffer");
    TT_FATAL(input.layout() == Layout::TILE, "QR: input tensor must be in TILE layout");
    TT_FATAL(input.dtype() == DataType::FLOAT32, "QR: only Float32 input is supported");
    TT_FATAL(input.logical_shape().rank() == 2, "QR: input must be rank-2");
    TT_FATAL(
        input.logical_shape()[-2] <= kMaxTileDim && input.logical_shape()[-1] <= kMaxTileDim,
        "QR: both input dimensions must be at most 32 (single tile), got {}",
        input.logical_shape());
}

QrDeviceOperation::spec_return_value_t QrDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto shape = tensor_args.input.logical_shape();
    const uint32_t m = shape[-2];
    const uint32_t n = shape[-1];
    const uint32_t k = std::min(m, n);

    auto layout = tt::tt_metal::TensorLayout(
        DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.memory_config);

    ttnn::Shape q_shape = tensor_args.input.logical_shape();
    q_shape[-1] = k;
    auto q_spec = tt::tt_metal::TensorSpec(q_shape, layout);

    ttnn::Shape r_shape = tensor_args.input.logical_shape();
    r_shape[-2] = k;
    auto r_spec = tt::tt_metal::TensorSpec(r_shape, layout);
    return {q_spec, r_spec};
}

QrDeviceOperation::tensor_return_value_t QrDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto [q_spec, r_spec] = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input.device();
    return {create_device_tensor(q_spec, device), create_device_tensor(r_spec, device)};
}

}  // namespace ttnn::operations::qr

namespace ttnn::prim {

std::tuple<Tensor, Tensor> qr(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::qr::QrDeviceOperation;
    TT_FATAL(input.device() != nullptr, "QR: input tensor must be on device");

    auto operation_attributes = OperationType::operation_attributes_t{
        memory_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
