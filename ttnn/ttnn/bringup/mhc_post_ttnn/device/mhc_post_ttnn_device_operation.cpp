// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_ttnn_device_operation.hpp"

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

void MhcPostDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    for (const Tensor* t : {&tensor_args.input, &tensor_args.residual, &tensor_args.post, &tensor_args.comb}) {
        TT_FATAL(t->storage_type() == ttnn::StorageType::DEVICE, "mhc_post: every operand must be on a device");
    }
}

void MhcPostDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcPostDeviceOperation::spec_return_value_t MhcPostDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // ttnn.allocate_tensor_on_device(Shape(residual.shape), residual.dtype, TILE_LAYOUT, device, DRAM_MEMORY_CONFIG)
    const auto& x = tensor_args.residual;
    return tt::tt_metal::TensorSpec(
        x.logical_shape(),
        tt::tt_metal::TensorLayout(
            x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
}

MhcPostDeviceOperation::tensor_return_value_t MhcPostDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.residual.device());
}

ttsl::hash::hash_t MhcPostDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Everything create_program_descriptor reads except buffer addresses: the four tensors' specs and the compute
    // config fields it copies. The device (grid) is fixed per program cache.
    const auto& cc = attrs.compute_config;
    return tt::tt_metal::operation::hash_operation<MhcPostDeviceOperation>(
        static_cast<uint32_t>(cc.math_fidelity),
        cc.fp32_dest_acc_en,
        cc.math_approx_mode,
        attrs.comb_transposed,
        tensor_args.input.tensor_spec(),
        tensor_args.residual.tensor_spec(),
        tensor_args.post.tensor_spec(),
        tensor_args.comb.tensor_spec());
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn

namespace ttnn::prim::bringup {

ttnn::Tensor mhc_post_ttnn(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    bool comb_transposed) {
    using OperationType = ttnn::operations::bringup::mhc_post_ttnn::MhcPostDeviceOperation;
    auto attrs =
        OperationType::operation_attributes_t{.compute_config = compute_config, .comb_transposed = comb_transposed};
    auto args = OperationType::tensor_args_t{.input = input, .residual = residual, .post = post, .comb = comb};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim::bringup
