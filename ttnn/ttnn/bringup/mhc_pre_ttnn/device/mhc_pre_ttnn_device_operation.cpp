// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_ttnn_device_operation.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

void MhcPreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    for (const Tensor* t : {&tensor_args.input, &tensor_args.proj_weight, &tensor_args.proj_bias}) {
        TT_FATAL(t->storage_type() == ttnn::StorageType::DEVICE, "mhc_pre: every operand must be on a device");
    }
}

void MhcPreDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcPreDeviceOperation::spec_return_value_t MhcPreDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // ttnn.allocate_tensor_on_device(Shape(lead + [last]), dtype, TILE_LAYOUT, device, DRAM_MEMORY_CONFIG)
    const auto& x = tensor_args.input;
    const auto& s = x.logical_shape();
    const uint32_t n = attrs.n;
    auto spec = [&](uint32_t last, DataType dtype) {
        ttsl::SmallVector<uint32_t> dims;
        for (size_t i = 0; i + 1 < s.rank(); ++i) {
            dims.push_back(s[i]);
        }
        dims.push_back(last);
        return tt::tt_metal::TensorSpec(
            ttnn::Shape(dims),
            tt::tt_metal::TensorLayout(
                dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    };
    return {spec(s[-1] / n, x.dtype()), spec(n, DataType::FLOAT32), spec(n * n, DataType::FLOAT32)};
}

MhcPreDeviceOperation::tensor_return_value_t MhcPreDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto [ys, ps, cs] = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input.device();
    return {create_device_tensor(ys, device), create_device_tensor(ps, device), create_device_tensor(cs, device)};
}

ttsl::hash::hash_t MhcPreDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Everything create_program_descriptor reads except buffer addresses: the input specs, n, the scalars (they are
    // runtime args the cache hit does not re-patch), the compute config fields it copies and the kernel-define env
    // switch. The device (grid, L1 size) is fixed per program cache.
    const char* defines = std::getenv("MHC_PRE_KERNEL_DEFINES");
    const auto& cc = attrs.compute_config;
    return tt::tt_metal::operation::hash_operation<MhcPreDeviceOperation>(
        attrs.n,
        attrs.scale[0],
        attrs.scale[1],
        attrs.scale[2],
        attrs.sinkhorn_iters,
        attrs.eps,
        attrs.norm_eps,
        static_cast<uint32_t>(cc.math_fidelity),
        cc.math_approx_mode,
        tensor_args.input.tensor_spec(),
        tensor_args.proj_weight.tensor_spec(),
        tensor_args.proj_bias.tensor_spec(),
        std::string(defines == nullptr ? "" : defines));
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> mhc_pre_ttnn(
    const ttnn::Tensor& input,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias,
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreParams& params) {
    using OperationType = ttnn::operations::bringup::mhc_pre_ttnn::MhcPreDeviceOperation;
    auto args = OperationType::tensor_args_t{.input = input, .proj_weight = proj_weight, .proj_bias = proj_bias};
    return ttnn::device_operation::launch<OperationType>(params, args);
}

}  // namespace ttnn::prim::bringup
