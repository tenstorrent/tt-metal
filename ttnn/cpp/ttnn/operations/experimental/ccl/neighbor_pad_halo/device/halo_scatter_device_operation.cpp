// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "halo_scatter_device_operation.hpp"
#include "halo_scatter_device_operation_types.hpp"
#include "halo_scatter_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void NpHaloScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& compact = tensor_args.compact_buffer;
    const auto& padded = tensor_args.padded_buffer;

    TT_FATAL(compact.layout() == Layout::ROW_MAJOR, "HaloScatter: compact_buffer must be row-major.");
    TT_FATAL(padded.layout() == Layout::ROW_MAJOR, "HaloScatter: padded_buffer must be row-major.");
    TT_FATAL(
        compact.dtype() == padded.dtype(),
        "HaloScatter: compact ({}) and padded ({}) dtype must match.",
        compact.dtype(),
        padded.dtype());
    TT_FATAL(
        compact.dtype() == DataType::BFLOAT16 || compact.dtype() == DataType::FLOAT32,
        "HaloScatter: dtype must be bfloat16 or float32. got {}",
        compact.dtype());

    const auto& pshape = padded.logical_shape();
    TT_FATAL(pshape.size() >= 4, "HaloScatter: padded_buffer must be rank >= 4 [.., H+2pH, W+2pW, C].");
    const uint32_t rank = pshape.size();
    TT_FATAL(
        pshape[rank - 3] > 2 * args.np_padding_h && pshape[rank - 2] > 2 * args.np_padding_w,
        "HaloScatter: padded H/W ({}, {}) must exceed 2*pad ({}, {}).",
        pshape[rank - 3],
        pshape[rank - 2],
        2 * args.np_padding_h,
        2 * args.np_padding_w);
    TT_FATAL(args.np_padding_h > 0, "HaloScatter: np_padding_h must be > 0.");
}

TensorSpec NpHaloScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // The op writes into the pre-allocated padded buffer and returns it.
    return tensor_args.padded_buffer.tensor_spec();
}

Tensor NpHaloScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tensor_args.padded_buffer;
}

ttsl::hash::hash_t NpHaloScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& padded = tensor_args.padded_buffer;
    return operation::hash_operation<NpHaloScatterDeviceOperation>(
        args, padded.dtype(), padded.memory_config(), padded.logical_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor halo_scatter(
    const Tensor& compact_buffer,
    const Tensor& padded_buffer,
    const ttnn::experimental::prim::NpHaloScatterParams& params) {
    using OperationType = ttnn::experimental::prim::NpHaloScatterDeviceOperation;
    auto tensor_args =
        OperationType::tensor_args_t{.compact_buffer = compact_buffer, .padded_buffer = padded_buffer};
    return ttnn::device_operation::launch<OperationType>(params, tensor_args);
}

}  // namespace ttnn::prim
