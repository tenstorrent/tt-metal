// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "depth_to_space_pad_device_operation.hpp"
#include "depth_to_space_pad_device_operation_types.hpp"
#include "depth_to_space_pad_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void DepthToSpacePadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.conv_out;
    TT_FATAL(x.layout() == Layout::ROW_MAJOR, "DepthToSpacePad: conv_out must be row-major.");
    TT_FATAL(
        x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::FLOAT32,
        "DepthToSpacePad: dtype must be bfloat16 or float32. got {}",
        x.dtype());
    TT_FATAL(
        x.logical_shape().size() == 5,
        "DepthToSpacePad: conv_out must be rank 5 [B,T,H,W,p1*p2*p3*C]. got {}",
        x.logical_shape().size());
    const uint32_t block = args.p1 * args.p2 * args.p3;
    TT_FATAL(block > 0, "DepthToSpacePad: p1*p2*p3 must be > 0.");
    TT_FATAL(
        x.logical_shape()[4] % block == 0,
        "DepthToSpacePad: channels ({}) must be divisible by p1*p2*p3 ({}).",
        x.logical_shape()[4],
        block);
    TT_FATAL(
        args.drop_first == 0 || (args.drop_first == 1 && args.p1 == 2), "DepthToSpacePad: drop_first requires p1==2.");
}

TensorSpec DepthToSpacePadDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& s = tensor_args.conv_out.logical_shape();
    const uint32_t C = s[4] / (args.p1 * args.p2 * args.p3);
    const uint32_t T_out = s[1] * args.p1 - args.drop_first;
    const uint32_t Hp_out = s[2] * args.p2 + 2 * args.np_padding_h;
    const uint32_t Wp_out = s[3] * args.p3 + 2 * args.np_padding_w;
    ttnn::Shape output_shape({s[0], T_out, Hp_out, Wp_out, C});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tensor_args.conv_out.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

Tensor DepthToSpacePadDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.conv_out.device());
}

ttsl::hash::hash_t DepthToSpacePadDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.conv_out;
    return operation::hash_operation<DepthToSpacePadDeviceOperation>(
        args, x.dtype(), x.memory_config(), x.logical_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor depth_to_space_pad(const Tensor& conv_out, const ttnn::experimental::prim::DepthToSpacePadParams& params) {
    using OperationType = ttnn::experimental::prim::DepthToSpacePadDeviceOperation;
    auto tensor_args = OperationType::tensor_args_t{.conv_out = conv_out};
    return ttnn::device_operation::launch<OperationType>(params, tensor_args);
}

}  // namespace ttnn::prim
