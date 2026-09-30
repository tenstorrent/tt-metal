// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_xing_device_operation.hpp"

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

void MhcPreXingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.storage_type() == ttnn::StorageType::DEVICE, "mhc_pre_xing: input must be on a device");
    if (tensor_args.streams.has_value()) {
        TT_FATAL(
            tensor_args.streams->storage_type() == ttnn::StorageType::DEVICE,
            "mhc_pre_xing: streams must be on a device");
    }
}

void MhcPreXingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcPreXingDeviceOperation::spec_return_value_t MhcPreXingDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& s = tensor_args.input.logical_shape();
    auto spec = [&](uint32_t last) {
        ttsl::SmallVector<uint32_t> dims;
        for (size_t i = 0; i + 1 < s.rank(); ++i) {
            dims.push_back(s[i]);
        }
        dims.push_back(last);
        return tt::tt_metal::TensorSpec(
            ttnn::Shape(dims),
            tt::tt_metal::TensorLayout(
                DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    };
    spec_return_value_t specs;
    if (attrs.pack_stats) {
        specs.push_back(spec(s[-1]));
        return specs;
    }
    if (attrs.compute_coef) {
        specs.push_back(spec(attrs.n * (attrs.n + 2)));
    }
    if (tensor_args.streams.has_value()) {
        specs.push_back(spec(tensor_args.streams->logical_shape()[-1] / attrs.n));
    }
    return specs;
}

MhcPreXingDeviceOperation::tensor_return_value_t MhcPreXingDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    tensor_return_value_t out;
    for (const auto& spec : compute_output_specs(attrs, tensor_args)) {
        out.push_back(create_device_tensor(spec, tensor_args.input.device()));
    }
    return out;
}

ttsl::hash::hash_t MhcPreXingDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The scalars (scale, base, eps, clamp, iterations) are runtime args patched on every call.
    const auto& cc = attrs.compute_config;
    const bool has_streams = tensor_args.streams.has_value();
    return tt::tt_metal::operation::hash_operation<MhcPreXingDeviceOperation>(
        attrs.n,
        attrs.compute_coef,
        attrs.pack_stats,
        has_streams,
        static_cast<uint32_t>(cc.math_fidelity),
        cc.math_approx_mode,
        tensor_args.input.tensor_spec(),
        has_streams ? tensor_args.streams->tensor_spec() : tensor_args.input.tensor_spec());
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {

std::vector<ttnn::Tensor> mhc_pre_xing(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& streams,
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingParams& params) {
    using OperationType = ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingDeviceOperation;
    auto args = OperationType::tensor_args_t{.input = input, .streams = streams};
    return ttnn::device_operation::launch<OperationType>(params, args);
}

}  // namespace ttnn::prim::bringup
