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
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.storage_type() == ttnn::StorageType::DEVICE, "mhc_pre_xing: input must be on a device");
    if (tensor_args.streams.has_value()) {
        TT_FATAL(
            tensor_args.streams->storage_type() == ttnn::StorageType::DEVICE,
            "mhc_pre_xing: streams must be on a device");
    }
    TT_FATAL(attrs.given_pre == tensor_args.collapse_pre.has_value(), "mhc_pre_xing: given_pre without collapse_pre");
}

void MhcPreXingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcPreXingDeviceOperation::spec_return_value_t MhcPreXingDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& s = tensor_args.input.logical_shape();
    auto spec = [&](uint32_t last, DataType dt = DataType::FLOAT32) {
        ttsl::SmallVector<uint32_t> dims;
        for (size_t i = 0; i + 1 < s.rank(); ++i) {
            dims.push_back((attrs.partials > 1 && i + 3 == s.rank()) ? 1 : s[i]);  // partials: (1, 1, T, ...)
        }
        dims.push_back(last);
        return tt::tt_metal::TensorSpec(
            ttnn::Shape(dims),
            tt::tt_metal::TensorLayout(
                dt, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
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
        // y in the streams' dtype (fp32, or bf16 streams -> bf16 y: P.20)
        specs.push_back(spec(tensor_args.streams->logical_shape()[-1] / attrs.n, tensor_args.streams->dtype()));
    }
    return specs;
}

MhcPreXingDeviceOperation::tensor_return_value_t MhcPreXingDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& t = tensor_args;
    const auto specs = compute_output_specs(attrs, t);
    // the given buffer of output k: [hc | packed] then [y] (the order compute_output_specs lists them)
    const bool has_hc = attrs.pack_stats || attrs.compute_coef;
    if (!has_hc) {
        TT_FATAL(!t.hc_output.has_value(), "mhc_pre_xing: hc_output given, but the call computes no hc");
    }
    if (!t.streams.has_value() || attrs.pack_stats) {
        TT_FATAL(!t.y_output.has_value(), "mhc_pre_xing: y_output given, but the call computes no y");
    }
    tensor_return_value_t out;
    for (size_t k = 0; k < specs.size(); ++k) {
        const auto& given = (k == 0 && has_hc) ? t.hc_output : t.y_output;
        if (!given.has_value()) {
            out.push_back(create_device_tensor(specs[k], t.input.device()));
            continue;
        }
        TT_FATAL(
            given->storage_type() == ttnn::StorageType::DEVICE && given->is_allocated(),
            "mhc_pre_xing: a given output must be an allocated device tensor");
        TT_FATAL(
            given->tensor_spec() == specs[k],
            "mhc_pre_xing: given output {} spec {} is not the op's {}",
            k,
            given->tensor_spec(),
            specs[k]);
        for (const auto* x : {&t.input}) {
            TT_FATAL(given->buffer()->address() != x->buffer()->address(), "mhc_pre_xing: a given output is an input");
        }
        if (t.streams.has_value()) {
            TT_FATAL(
                given->buffer()->address() != t.streams->buffer()->address(),
                "mhc_pre_xing: a given output is an input");
        }
        if (t.collapse_pre.has_value()) {
            TT_FATAL(
                given->buffer()->address() != t.collapse_pre->buffer()->address(),
                "mhc_pre_xing: a given output is an input");
        }
        out.push_back(*given);
    }
    return out;
}

ttsl::hash::hash_t MhcPreXingDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The scalars (scale, base, eps, clamp, iterations) are runtime args patched on every call.
    const auto& cc = attrs.compute_config;
    const bool has_streams = tensor_args.streams.has_value();
    if (attrs.partials > 1) {  // a separate program (define MHC_XING_PARTIALS); off: the hashes below, unchanged
        return tt::tt_metal::operation::hash_operation<MhcPreXingDeviceOperation>(
            attrs.n,
            attrs.compute_coef,
            attrs.pack_stats,
            has_streams,
            static_cast<uint32_t>(cc.math_fidelity),
            cc.math_approx_mode,
            tensor_args.input.tensor_spec(),
            has_streams ? tensor_args.streams->tensor_spec() : tensor_args.input.tensor_spec(),
            attrs.deepseek,
            attrs.given_pre,
            attrs.given_pre ? tensor_args.collapse_pre->tensor_spec() : tensor_args.input.tensor_spec(),
            attrs.partials);
    }
    if (attrs.given_pre) {  // a separate program (DM define + the pre tensor); off: the hashes below, unchanged
        return tt::tt_metal::operation::hash_operation<MhcPreXingDeviceOperation>(
            attrs.n,
            attrs.compute_coef,
            attrs.pack_stats,
            has_streams,
            static_cast<uint32_t>(cc.math_fidelity),
            cc.math_approx_mode,
            tensor_args.input.tensor_spec(),
            has_streams ? tensor_args.streams->tensor_spec() : tensor_args.input.tensor_spec(),
            attrs.deepseek,
            attrs.given_pre,
            tensor_args.collapse_pre->tensor_spec());
    }
    if (attrs.deepseek) {  // a separate program (compute define); the off hash is the one it always was
        return tt::tt_metal::operation::hash_operation<MhcPreXingDeviceOperation>(
            attrs.n,
            attrs.compute_coef,
            attrs.pack_stats,
            has_streams,
            static_cast<uint32_t>(cc.math_fidelity),
            cc.math_approx_mode,
            tensor_args.input.tensor_spec(),
            has_streams ? tensor_args.streams->tensor_spec() : tensor_args.input.tensor_spec(),
            attrs.deepseek);
    }
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
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingParams& params,
    const std::optional<ttnn::Tensor>& collapse_pre,
    const std::optional<ttnn::Tensor>& hc_output,
    const std::optional<ttnn::Tensor>& y_output) {
    using OperationType = ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingDeviceOperation;
    auto args = OperationType::tensor_args_t{
        .input = input, .streams = streams, .collapse_pre = collapse_pre, .hc_output = hc_output, .y_output = y_output};
    return ttnn::device_operation::launch<OperationType>(params, args);
}

}  // namespace ttnn::prim::bringup
