// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>

#include "device/dit_fused_distributed_groupnorm_device_operation.hpp"
#include "device/dit_fused_distributed_groupnorm_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

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
    bool use_welford,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    (void)cluster_width(mesh_device, cluster_axis);

    // Always launch the fused device op. Width-1 runs local PRE+POST (no fabric);
    // width>1 runs PRE → fabric AG → POST. Plain […,C] γ/β (not GN-packed).
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
        std::move(compute_kernel_config),
        use_welford,
        persistent_output_buffer,
        num_preferred_links,
        subdevice_id);
}

std::optional<ttnn::Tensor> dit_fused_distributed_groupnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_groups,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links) {
    (void)input_tensor;
    (void)num_links;
    const uint32_t ring_size = cluster_width(mesh_device, cluster_axis);
    const uint32_t channels = input_tensor.logical_shape()[3];
    const uint32_t grid_x = mesh_device.compute_with_storage_grid_size().x;
    auto sizing = ttnn::experimental::prim::gn_make_sizing(num_groups, ring_size, channels, grid_x);
    if (sizing.is_local) {
        return std::nullopt;
    }

    // One ROW_MAJOR fp32 page per device; each page holds all masters' 64 B-aligned sub-sticks.
    const uint32_t floats_per_page = sizing.page_size_bytes / sizeof(float);
    ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, floats_per_page});
    MemoryConfig stats_mem{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::TensorSpec spec(
        stats_shape,
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), stats_mem));
    return tt::tt_metal::create_device_tensor(spec, &const_cast<MeshDevice&>(mesh_device));
}

}  // namespace ttnn::experimental

namespace ttnn::experimental::prim {

uint32_t gn_num_virtual_cols(uint32_t channels, uint32_t num_groups, uint32_t grid_x) {
    constexpr uint32_t tile_w = 32u;
    // Mirror groupnorm_mcast_program_factory.cpp: start at min(grid.x, num_groups) and shrink until
    // the per-column width is tile-aligned and the groups divide evenly across columns. cols==1
    // always satisfies both (channels % tile_w == 0 is validated, num_groups % 1 == 0), so the loop
    // terminates.
    uint32_t cols = std::min(grid_x, num_groups);
    while (cols > 1u && ((channels / cols) % tile_w != 0u || (num_groups % cols) != 0u)) {
        cols -= 1u;
    }
    return cols;
}

DitFusedDistributedGroupnormSizing gn_make_sizing(
    uint32_t num_groups, uint32_t ring_size, uint32_t channels, uint32_t grid_x) {
    DitFusedDistributedGroupnormSizing s;
    s.num_groups = num_groups;
    s.is_local = (ring_size <= 1);
    // Multi-core mcast layout: masters = num_virtual_cols; each owns num_groups/cols groups.
    s.num_masters = gn_num_virtual_cols(channels, num_groups, grid_x);
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

}  // namespace ttnn::experimental::prim
