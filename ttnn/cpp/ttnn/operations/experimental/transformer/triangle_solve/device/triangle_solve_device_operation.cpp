// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "triangle_solve_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

void ts_check_shape(const Tensor& t, std::initializer_list<uint32_t> expected, const std::string& name) {
    const auto& s = t.logical_shape();
    TT_FATAL(
        static_cast<size_t>(s.rank()) == expected.size(),
        "{} rank mismatch: got {} expected {}",
        name,
        s.rank(),
        expected.size());
    size_t i = 0;
    for (auto e : expected) {
        TT_FATAL(static_cast<uint32_t>(s[i]) == e, "{} dim[{}] expected {} got {}", name, i, e, s[i]);
        ++i;
    }
}

void check_device_tiled_bf16(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "{} must be bfloat16, got {}", name, t.dtype());
    ts_check_shape(t, {1, 1, TILE_HEIGHT, TILE_WIDTH}, name);
}

}  // namespace

void TriangleSolveDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& in) {
    check_device_tiled_bf16(in.l_neg, "l_neg");
    check_device_tiled_bf16(in.rhs, "rhs");
}

TriangleSolveDeviceOperation::spec_return_value_t TriangleSolveDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& /*in*/) {
    const auto& mc = attrs.output_mem_config;
    tt::tt_metal::TensorSpec x_spec(
        ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mc));
    return {x_spec};
}

TriangleSolveDeviceOperation::tensor_return_value_t TriangleSolveDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.rhs.device();
    return {create_device_tensor(specs[0], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> triangle_solve(
    const Tensor& l_neg,
    const Tensor& rhs,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Op = ttnn::experimental::prim::TriangleSolveDeviceOperation;

    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        Op::tensor_args_t{
            .l_neg = l_neg,
            .rhs = rhs,
        });
}

}  // namespace ttnn::prim
