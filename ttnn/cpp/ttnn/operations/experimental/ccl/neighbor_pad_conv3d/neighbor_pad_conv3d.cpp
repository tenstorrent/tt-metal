// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_conv3d.hpp"
#include "device/neighbor_pad_conv3d_device_operation.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/math.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor neighbor_pad_conv3d(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    const std::optional<ttnn::Tensor>& bias,
    const ttnn::Tensor& halo_buffer,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    uint32_t np_cluster_axis,
    size_t np_num_links,
    ttnn::ccl::Topology np_topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const GlobalSemaphore& w_neighbor_semaphore,
    uint32_t np_pad_dim2,
    uint32_t np_pad2_left,
    uint32_t np_pad2_right,
    uint32_t np_pad2_cluster_axis,
    size_t np_pad2_num_links,
    const ttnn::experimental::prim::Conv3dConfig& conv_config,
    uint32_t output_channels,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& dilation,
    const std::string& padding_mode,
    uint32_t groups,
    tt::tt_metal::DataType dtype,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(np_padding_h > 0, "neighbor_pad_conv3d: np_padding_h must be > 0");
    TT_FATAL(groups >= 1, "neighbor_pad_conv3d: groups must be >= 1");

    // Derive ring size from mesh view along the cluster axis
    auto* mesh_device = input.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t np_ring_size = (np_cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(
        np_ring_size > 1,
        "neighbor_pad_conv3d: requires num_devices > 1 along cluster axis {}, got {}",
        np_cluster_axis,
        np_ring_size);

    // Resolve topology for the fabric
    tt::tt_fabric::Topology resolved_topology =
        ::ttnn::ccl::get_usable_topology(input, std::make_optional(np_topology), np_cluster_axis);

    // Resolve compute kernel config using arch-aware helper (matches conv3d convention)
    auto resolved_compute_kernel_config = ttnn::init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);

    // Resolve output memory config
    MemoryConfig output_mem_config = memory_config.value_or(input.memory_config());

    // Build the fused params struct
    ttnn::experimental::prim::NpConv3dParams params(
        np_padding_h,
        np_padding_w,
        np_cluster_axis,
        np_ring_size,
        resolved_topology,
        h_neighbor_semaphore,
        barrier_semaphore,
        w_neighbor_semaphore,
        np_pad_dim2 > 0 ? std::make_optional(np_pad_dim2) : std::nullopt,
        np_pad2_left,
        np_pad2_right,
        np_pad_dim2 > 0 ? std::make_optional(np_pad2_cluster_axis) : std::nullopt,
        np_num_links,
        np_pad2_num_links,
        input.memory_config(),  // np output mem config (NP writes into halo_buffer, not a new tensor)
        conv_config,
        output_mem_config,
        resolved_compute_kernel_config,
        dtype,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        padding_mode,
        groups);

    return ttnn::prim::neighbor_pad_conv3d(input, weight, bias, halo_buffer, params);
}

}  // namespace ttnn::experimental
