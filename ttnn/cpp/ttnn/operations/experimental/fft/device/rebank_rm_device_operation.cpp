// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rebank_rm_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {


RebankRmDeviceOperation::program_factory_t
RebankRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return RebankRmFactory{};
}

void RebankRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    const auto& x     = args.input;
    const uint32_t chunk = attrs.chunk_size;

    TT_FATAL(x.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "rebank_rm: only Float32 / BFloat16 supported.");
    TT_FATAL(x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "rebank_rm: input must be ROW_MAJOR.");

    const auto& shape = x.padded_shape();
    TT_FATAL(shape.size() >= 2u,
        "rebank_rm: input must have >= 2 dims (got {}).", shape.size());

    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    TT_FATAL(rebank_is_pow2(chunk) && chunk >= 1u && chunk <= N,
        "rebank_rm: chunk_size must be pow-2 in [1, N={}] (got {}).", N, chunk);
    TT_FATAL(N % chunk == 0u,
        "rebank_rm: N={} must be divisible by chunk_size={}.", N, chunk);
    // N need not be a power of 2; only chunk must be pow-2 (for output-page alignment).
}

RebankRmDeviceOperation::spec_return_value_t
RebankRmDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    const auto& in     = args.input;
    const auto& s      = in.padded_shape();
    const uint32_t N   = static_cast<uint32_t>(s[-1]);
    const uint32_t chunk = attrs.chunk_size;

    uint32_t B_total = 1u;
    for (int d = 0; d < static_cast<int>(s.size()) - 1; ++d)
        B_total *= static_cast<uint32_t>(s[d]);

    // Output is always 2D: (B_total * N/chunk, chunk).
    const ttnn::Shape out_shape{
        ttnn::SmallVector<uint32_t>{B_total * (N / chunk), chunk}};

    TensorLayout layout(in.dtype(), PageConfig(in.layout()), in.memory_config());
    return TensorSpec(std::move(out_shape), std::move(layout));
}

RebankRmDeviceOperation::tensor_return_value_t
RebankRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

tt::stl::hash::hash_t RebankRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return tt::tt_metal::operation::hash_operation<RebankRmDeviceOperation>(
        attrs.chunk_size,
        args.input.dtype(),
        args.input.memory_config(),
        args.input.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor rebank_rm(const Tensor& input, uint32_t chunk_size) {
    using Op = ttnn::experimental::prim::RebankRmDeviceOperation;
    Op::operation_attributes_t attrs{ .chunk_size = chunk_size };
    Op::tensor_args_t          t_args{ .input = input };
    return ttnn::device_operation::launch<Op>(attrs, t_args);
}

}  // namespace ttnn::prim
