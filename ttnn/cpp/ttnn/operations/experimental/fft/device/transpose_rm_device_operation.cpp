// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_rm_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"   // ttnn::Shape, ttnn::SmallVector

namespace ttnn::experimental::prim {

namespace {

// Lift the last two dims of a >=2D padded shape into (A, C).
inline std::pair<uint32_t, uint32_t> last_two_dims(const tt::tt_metal::Shape& s) {
    return { static_cast<uint32_t>(s[-2]), static_cast<uint32_t>(s[-1]) };
}

}  // namespace

TransposeRmDeviceOperation::program_factory_t
TransposeRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return TransposeRmFactory{};
}

void TransposeRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& args)
{
    const auto& x = args.input;
    TT_FATAL(x.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "transpose_rm: only Float32 and BFloat16 supported (got {}).",
        static_cast<int>(x.dtype()));
    TT_FATAL(x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "transpose_rm: only ROW_MAJOR layout supported.");

    const auto& shape = x.padded_shape();
    TT_FATAL(shape.size() >= 2,
        "transpose_rm: input must have >=2 dims (got {}).", shape.size());

    const auto [A, C] = last_two_dims(shape);
    TT_FATAL(A >= 32u && (A % 32u) == 0u,
        "transpose_rm: A (second-to-last dim) must be a multiple of 32 and >=32 (got {}).", A);
    TT_FATAL(C >= 32u && (C % 32u) == 0u,
        "transpose_rm: C (last dim) must be a multiple of 32 and >=32 (got {}).", C);
}

TransposeRmDeviceOperation::spec_return_value_t
TransposeRmDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    const auto& in = args.input;
    const auto& in_shape = in.padded_shape();

    // Output shape: same except last two dims swapped.
    ttnn::SmallVector<uint32_t> out_dims;
    out_dims.reserve(in_shape.size());
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 2; ++d) {
        out_dims.push_back(static_cast<uint32_t>(in_shape[d]));
    }
    const auto [A, C] = last_two_dims(in_shape);
    out_dims.push_back(C);
    out_dims.push_back(A);

    ttnn::Shape out_shape{out_dims};
    TensorLayout layout(in.dtype(), PageConfig(in.layout()), in.memory_config());
    return TensorSpec(std::move(out_shape), std::move(layout));
}

TransposeRmDeviceOperation::tensor_return_value_t
TransposeRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

tt::stl::hash::hash_t TransposeRmDeviceOperation::compute_program_hash(
    const operation_attributes_t&, const tensor_args_t& args)
{
    return tt::tt_metal::operation::hash_operation<TransposeRmDeviceOperation>(
        args.input.dtype(),
        args.input.memory_config(),
        args.input.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor transpose_rm(const Tensor& input) {
    using OperationType = ttnn::experimental::prim::TransposeRmDeviceOperation;
    OperationType::operation_attributes_t attrs{};
    OperationType::tensor_args_t          args{ .input = input };
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
