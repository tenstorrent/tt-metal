// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_halo_device_operation.hpp"
#include "neighbor_pad_halo_device_operation_types.hpp"
#include "neighbor_pad_halo_program_factory.hpp"

#include <array>
#include <cstdint>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void NpHaloDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    TT_FATAL(
        input_shape.size() == 5,
        "NpHalo: Activation tensor must have 5 dimensions (BTHWC). got {}",
        input_shape.size());
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "NpHalo: Activation tensor must be row-major.");
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32,
        "NpHalo: Activation tensor must be bfloat16 or float32. got {}",
        input_tensor.dtype());

    TT_FATAL(
        args.padding_mode == "zeros" || args.padding_mode == "replicate",
        "NpHalo: Padding mode must be zeros or replicate. got {}",
        args.padding_mode);

    TT_FATAL(args.np_padding_h > 0, "NpHalo: np_padding_h must be > 0 (H-halo must be needed).");

    // Only the 2D (H+W) compact-buffer path is implemented (the deployed VAE always pads both H and W).
    TT_FATAL(args.np_pad_dim2.has_value(), "NpHalo: requires 2D padding (H+W); np_pad_dim2 must be set.");

    // The halo_buffer is pre-allocated by the caller and written in place; it must be a DRAM tensor
    // with the same dtype as the input (the exchange is a raw stick copy, no arithmetic).
    const auto& halo_buffer = tensor_args.halo_buffer;
    TT_FATAL(
        halo_buffer.dtype() == input_tensor.dtype(),
        "NpHalo: halo_buffer dtype ({}) must match input dtype ({}).",
        halo_buffer.dtype(),
        input_tensor.dtype());
}

TensorSpec NpHaloDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // The op writes into the pre-allocated compact halo buffer and returns it.
    return tensor_args.halo_buffer.tensor_spec();
}

Tensor NpHaloDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tensor_args.halo_buffer;
}

ttsl::hash::hash_t NpHaloDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    operation::Hash hash = operation::hash_operation<NpHaloDeviceOperation>(
        args, input_tensor.dtype(), input_tensor.memory_config(), input_tensor.logical_shape());
    return hash;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor neighbor_pad_halo(
    const Tensor& input, const Tensor& halo_buffer, const ttnn::experimental::prim::NpHaloParams& params) {
    using OperationType = ttnn::experimental::prim::NpHaloDeviceOperation;

    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input, .halo_buffer = halo_buffer};

    return ttnn::device_operation::launch<OperationType>(params, tensor_args);
}

}  // namespace ttnn::prim
