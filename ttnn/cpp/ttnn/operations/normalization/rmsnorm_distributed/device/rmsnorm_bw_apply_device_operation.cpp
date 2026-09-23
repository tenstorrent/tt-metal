// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_apply_device_operation.hpp"

#include <algorithm>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::normalization::rmsnorm_distributed_bw {
namespace {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

void validate_fp32_tile(const Tensor& tensor, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "rmsnorm_bw_apply: {} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "rmsnorm_bw_apply: {} must be allocated", name);
    TT_FATAL(
        tensor.layout() == Layout::TILE, "rmsnorm_bw_apply: {} must have TILE layout, got {}", name, tensor.layout());
    TT_FATAL(tensor.dtype() == DataType::FLOAT32, "rmsnorm_bw_apply: {} must be FLOAT32, got {}", name, tensor.dtype());
    TT_FATAL(!tensor.is_sharded(), "rmsnorm_bw_apply: {} must be interleaved", name);
}

void validate_same_device(const Tensor& a, const Tensor& b, std::string_view a_name, std::string_view b_name) {
    TT_FATAL(a.device() == b.device(), "rmsnorm_bw_apply: {} and {} must be on the same device", a_name, b_name);
}

}  // namespace

ApplyOccupancy compute_apply_occupancy(const Tensor& x) {
    const auto& padded = x.padded_shape();
    const uint32_t tile_h = x.tensor_spec().tile().get_height();
    const uint32_t tile_w = x.tensor_spec().tile().get_width();
    const auto grid = x.device()->compute_with_storage_grid_size();
    ApplyOccupancy occ;
    occ.Wt = padded[3] / tile_w;
    occ.num_rows = padded[0] * padded[1] * (padded[2] / tile_h);
    occ.grid_x = static_cast<uint32_t>(grid.x);
    occ.grid_y = static_cast<uint32_t>(grid.y);
    const uint32_t max_cores = occ.grid_x * occ.grid_y;
    occ.num_cores = std::min(max_cores, occ.num_rows);
    occ.n_rows_used = (occ.num_cores + occ.grid_x - 1) / occ.grid_x;
    return occ;
}

void RMSNormBwApplyOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(!operation_attributes.memory_config.is_sharded(), "rmsnorm_bw_apply: memory_config must be interleaved");

    validate_fp32_tile(tensor_args.x, "x");
    validate_fp32_tile(tensor_args.dy, "dy");
    validate_fp32_tile(tensor_args.inv_rms, "inv_rms");
    validate_fp32_tile(tensor_args.d, "d");
    validate_same_device(tensor_args.x, tensor_args.dy, "x", "dy");
    validate_same_device(tensor_args.x, tensor_args.inv_rms, "x", "inv_rms");
    validate_same_device(tensor_args.x, tensor_args.d, "x", "d");
    TT_FATAL(
        tensor_args.x.logical_shape() == tensor_args.dy.logical_shape(),
        "rmsnorm_bw_apply: x and dy shapes must match, got {} vs {}",
        tensor_args.x.logical_shape(),
        tensor_args.dy.logical_shape());

    const auto& x_shape = tensor_args.x.logical_shape();
    TT_FATAL(x_shape.rank() == 4, "rmsnorm_bw_apply: x must be rank-4, got rank {}", x_shape.rank());
    const uint32_t tile_h = tensor_args.x.tensor_spec().tile().get_height();
    const uint32_t tile_w = tensor_args.x.tensor_spec().tile().get_width();
    TT_FATAL(tile_h == TILE_HEIGHT && tile_w == TILE_WIDTH, "rmsnorm_bw_apply: only 32x32 tiles are supported");

    const auto occ = compute_apply_occupancy(tensor_args.x);
    TT_FATAL(occ.Wt > 0 && occ.num_rows > 0, "rmsnorm_bw_apply: empty tensor");
    TT_FATAL(
        occ.grid_x <= TILE_HEIGHT,
        "rmsnorm_bw_apply: grid.x ({}) exceeds in-tile gather rows ({})",
        occ.grid_x,
        TILE_HEIGHT);
    TT_FATAL(
        occ.n_rows_used <= TILE_HEIGHT,
        "rmsnorm_bw_apply: grid rows used ({}) exceed in-tile gather rows ({})",
        occ.n_rows_used,
        TILE_HEIGHT);

    const auto validate_row_stats = [&](const Tensor& tensor, std::string_view name) {
        const auto& shape = tensor.logical_shape();
        TT_FATAL(shape.rank() == 4, "rmsnorm_bw_apply: {} must be rank-4, got rank {}", name, shape.rank());
        TT_FATAL(shape[3] == 1, "rmsnorm_bw_apply: {} last dim must be 1, got {}", name, shape[3]);
        for (int dim = 0; dim < 3; ++dim) {
            TT_FATAL(
                shape[dim] == x_shape[dim],
                "rmsnorm_bw_apply: {} dim {} is {} but x's is {}",
                name,
                dim,
                shape[dim],
                x_shape[dim]);
        }
    };
    validate_row_stats(tensor_args.inv_rms, "inv_rms");
    validate_row_stats(tensor_args.d, "d");

    if (tensor_args.gamma.has_value()) {
        validate_fp32_tile(tensor_args.gamma.value(), "gamma");
        validate_same_device(tensor_args.x, tensor_args.gamma.value(), "x", "gamma");
        const auto& g_shape = tensor_args.gamma->logical_shape();
        TT_FATAL(
            g_shape.rank() == 4 && g_shape[0] == 1 && g_shape[1] == 1 && g_shape[2] == 1 && g_shape[3] == x_shape[3],
            "rmsnorm_bw_apply: gamma must have shape [1, 1, 1, {}], got {}",
            x_shape[3],
            g_shape);
    }
}

RMSNormBwApplyOperation::spec_return_value_t RMSNormBwApplyOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<std::optional<tt::tt_metal::TensorSpec>> specs(2);
    specs[0] = tt::tt_metal::TensorSpec(
        tensor_args.x.logical_shape(),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), operation_attributes.memory_config));
    if (tensor_args.gamma.has_value()) {
        specs[1] = tt::tt_metal::TensorSpec(
            tensor_args.gamma->logical_shape(),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), tensor_args.gamma->memory_config()));
    }
    return specs;
}

RMSNormBwApplyOperation::tensor_return_value_t RMSNormBwApplyOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.x.device();
    std::vector<std::optional<Tensor>> outputs(2);
    outputs[0] = create_device_tensor(*specs[0], device);
    if (specs[1].has_value()) {
        outputs[1] = create_device_tensor(*specs[1], device);
    }
    return outputs;
}

ttsl::hash::hash_t RMSNormBwApplyOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto occ = compute_apply_occupancy(tensor_args.x);
    const auto gamma_mem_config = tensor_args.gamma.has_value() ? tensor_args.gamma->memory_config() : MemoryConfig{};
    return ttsl::hash::hash_objects_with_default_seed(
        operation_attributes.memory_config,
        operation_attributes.compute_kernel_config,
        occ.Wt,
        occ.num_cores,
        occ.grid_x,
        occ.grid_y,
        tensor_args.gamma.has_value(),
        tensor_args.x.memory_config(),
        tensor_args.dy.memory_config(),
        tensor_args.inv_rms.memory_config(),
        tensor_args.d.memory_config(),
        gamma_mem_config);
}

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw

namespace ttnn::prim {

std::vector<std::optional<Tensor>> rmsnorm_bw_apply(
    const Tensor& x,
    const Tensor& dy,
    const std::optional<Tensor>& gamma,
    const Tensor& inv_rms,
    const Tensor& d,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::normalization::rmsnorm_distributed_bw::RMSNormBwApplyOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{memory_config, compute_kernel_config},
        OperationType::tensor_args_t{x, dy, gamma, inv_rms, d});
}

}  // namespace ttnn::prim
