// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rebank_rm_merge_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"
#include "rebank_rm_device_operation_types.hpp"  // rebank_is_pow2

namespace ttnn::experimental::prim {


RebankRmMergeDeviceOperation::program_factory_t
RebankRmMergeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return RebankRmMergeFactory{};
}

void RebankRmMergeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    const auto& x = args.input;
    const uint32_t cpm = attrs.chunks_per_merge;

    TT_FATAL(x.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "rebank_rm_merge: only Float32 / BFloat16 supported.");
    TT_FATAL(x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "rebank_rm_merge: input must be ROW_MAJOR.");

    const auto& shape = x.padded_shape();
    TT_FATAL(shape.size() == 2u,
        "rebank_rm_merge: input must be 2D (got {}D).", shape.size());

    const uint32_t rows = static_cast<uint32_t>(shape[0]);
    TT_FATAL(cpm >= 1u,
        "rebank_rm_merge: chunks_per_merge must be ≥ 1 (got {}).", cpm);
    // chunks_per_merge need not be a power of 2; the kernel uses compile-time
    // constant division which the compiler optimises via reciprocal multiplication.
    TT_FATAL(rows % cpm == 0u,
        "rebank_rm_merge: input rows {} not divisible by chunks_per_merge {}.", rows, cpm);
}

RebankRmMergeDeviceOperation::spec_return_value_t
RebankRmMergeDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    const auto& in = args.input;
    const auto& s  = in.padded_shape();
    const uint32_t rows = static_cast<uint32_t>(s[0]);
    const uint32_t N1   = static_cast<uint32_t>(s[1]);
    const uint32_t cpm  = attrs.chunks_per_merge;

    // Output: (B, N1*cpm) where B = rows/cpm.
    const ttnn::Shape out_shape{
        ttnn::SmallVector<uint32_t>{rows / cpm, N1 * cpm}};

    TensorLayout layout(in.dtype(), PageConfig(in.layout()), in.memory_config());
    return TensorSpec(std::move(out_shape), std::move(layout));
}

RebankRmMergeDeviceOperation::tensor_return_value_t
RebankRmMergeDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

tt::stl::hash::hash_t RebankRmMergeDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return tt::tt_metal::operation::hash_operation<RebankRmMergeDeviceOperation>(
        attrs.chunks_per_merge,
        args.input.dtype(),
        args.input.memory_config(),
        args.input.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor rebank_rm_merge(const Tensor& input, uint32_t chunks_per_merge) {
    using Op = ttnn::experimental::prim::RebankRmMergeDeviceOperation;
    Op::operation_attributes_t attrs{ .chunks_per_merge = chunks_per_merge };
    Op::tensor_args_t          t_args{ .input = input };
    return ttnn::device_operation::launch<Op>(attrs, t_args);
}

}  // namespace ttnn::prim
