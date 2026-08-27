// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>

#include "device/dit_fused_distributed_groupnorm_device_operation.hpp"
#include "device/dit_fused_distributed_groupnorm_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm_grid_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;

namespace ttnn::experimental {

namespace {

uint32_t cluster_width(const MeshDevice& mesh_device, uint32_t cluster_axis) {
    const auto& mesh_view = mesh_device.get_view();
    return static_cast<uint32_t>((cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols());
}

}  // namespace

ttnn::Tensor dit_fused_distributed_groupnorm(
    const ttnn::Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation) {
    // Always launch the fused device op. Width-1 runs local PRE+POST (no fabric);
    // width>1 runs PRE → fabric AG → POST. Packed RM γ/β + input_mask (same as ttnn.group_norm).
    return ttnn::prim::dit_fused_distributed_groupnorm(
        input_tensor,
        num_groups,
        epsilon,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        input_mask,
        weight,
        bias,
        memory_config,
        compute_kernel_config,
        persistent_output_buffer,
        subdevice_id,
        fused_activation);
}

std::optional<ttnn::Tensor> dit_fused_distributed_groupnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_groups,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device) {
    TT_FATAL(num_groups >= 1, "num_groups must be >= 1, got {}", num_groups);
    TT_FATAL(cluster_axis < 2, "cluster_axis must be 0 or 1, got {}", cluster_axis);
    const auto& in_shape = input_tensor.logical_shape();
    TT_FATAL(in_shape.rank() == 4, "input rank must be 4 ([N, 1, H*W, C]), got {}", in_shape.rank());
    const uint32_t channels = in_shape[3];
    TT_FATAL(channels % TILE_WIDTH == 0, "C ({}) must be divisible by TILE_WIDTH ({})", channels, TILE_WIDTH);
    TT_FATAL(channels % num_groups == 0, "C ({}) must be divisible by num_groups ({})", channels, num_groups);
    const uint32_t ring_size = cluster_width(mesh_device, cluster_axis);
    const uint32_t grid_x = mesh_device.compute_with_storage_grid_size().x;
    auto sizing = ttnn::experimental::prim::gn_make_sizing(num_groups, ring_size, channels, grid_x);
    if (sizing.is_local) {
        return std::nullopt;
    }

    return ttnn::create_device_tensor(
        ttnn::experimental::prim::make_stats_tensor_spec(sizing), &const_cast<MeshDevice&>(mesh_device));
}

}  // namespace ttnn::experimental

namespace ttnn::experimental::prim {

DitFusedDistributedGroupnormSizing gn_make_sizing(
    uint32_t num_groups, uint32_t ring_size, uint32_t channels, uint32_t grid_x) {
    DitFusedDistributedGroupnormSizing s;
    s.num_groups = num_groups;
    s.is_local = (ring_size <= 1);
    // Multi-core mcast layout: masters = num_virtual_cols; each owns num_groups/cols groups.
    s.num_masters =
        ttnn::operations::normalization::compute_num_virtual_cols(grid_x, static_cast<int>(num_groups), channels);
    s.num_groups_per_core = num_groups / s.num_masters;
    // Per-master stick: bf16 [mean, var] over its groups (num_groups_per_core * 4 B), rounded up to
    // NOC_DRAM_READ_ALIGNMENT_BYTES (64 on Blackhole). Each master NoC-reads its own sub-stick at
    // DRAM offset slot*stick_bytes; a non-64-aligned offset reads back as zero on BH (which halved
    // the variance → √2 output scale). Rounding keeps every sub-stick offset 64-aligned.
    s.stick_bytes = ((s.num_groups_per_core * 4u + 63u) / 64u) * 64u;
    // A single forwarder coalesces all masters' sub-sticks into one packet (whole device stat is
    // num_groups*4 B ≤ one fabric packet), so there is one DRAM chunk per device.
    s.num_forwarders = s.is_local ? 0u : 1u;
    s.num_chunks_per_device = s.num_forwarders;  // max_rounds == 1
    s.total_pages = s.is_local ? 0u : ring_size;
    s.page_size_bytes = s.num_masters * s.stick_bytes;
    return s;
}

DitFusedDistributedGroupnormSizing compute_sizing(const DitFusedDistributedGroupnormParams& args, const Tensor& input) {
    const uint32_t channels = input.logical_shape()[3];
    const uint32_t grid_x = input.device()->compute_with_storage_grid_size().x;
    return gn_make_sizing(args.num_groups, args.ring_size, channels, grid_x);
}

tt::tt_metal::TensorSpec make_stats_tensor_spec(const DitFusedDistributedGroupnormSizing& sizing) {
    // One ROW_MAJOR fp32 page per device; each page holds all masters' 64 B-aligned sub-sticks.
    const uint32_t floats_per_page = sizing.page_size_bytes / sizeof(float);
    ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, floats_per_page});
    tt::tt_metal::MemoryConfig stats_mem{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        stats_shape,
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), stats_mem));
}

}  // namespace ttnn::experimental::prim
