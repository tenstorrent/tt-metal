// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation_types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

// Conv3dConfig plus the controls read only by the fused NeighborPad+Conv3d op (its program factory,
// device op, and Python router). Subclassing keeps every conv3d blocking field and lets a config
// build/route exactly like a Conv3dConfig, while these extra fields stay out of the shared upstream
// struct. attribute_values() below is sliced to the base Conv3dConfig fields so this config hashes
// identically to a standalone Conv3dConfig; the two scheme flags that select a structurally different
// program (halo_last, force_spatial_parallel) are folded into the program hash by
// NpConv3dDeviceOperation::compute_program_hash instead, keeping the scheme out of the base struct's
// hash while still distinguishing cached programs per scheme. The remaining fields are per-call
// runtime addresses the Python wrapper allocates each dispatch; they are not part of any hash. Every
// other fused parameter (the per-T progress batch size, the region-progress link stride, the halo
// buffer geometry, and the always-on halo-buffer mode) is derived in the program factory from the
// base blocking fields, the op's NP topology, and the input tensor shape — so it is not stored here.
struct NpConv3dConfig : Conv3dConfig {
    using Conv3dConfig::Conv3dConfig;  // inherit the base constructor so Python builds it identically

    // Per-call runtime addresses, allocated and reset by the Python wrapper each dispatch (not hashed).
    // region_progress_sem_addr: per-(region,link) progress GlobalSemaphore L1 addrs, indexed
    //   [region*num_links + link] with region {H-top=0, H-bot=1, W-left=2, W-right=3}. Count =
    //   4 regions × num_links: 8 on BH-LB (num_links=2), up to 4*MAX_PAD2_NUM_LINKS=16 on 4x8. The link
    //   stride is the op's np_num_links (the factory derives it). One producer per (region,link) → each
    //   sem is monotonic in that link's T-batches; a conv3d edge tile maps the batch it needs to the
    //   owning link and polls only that sem — race-free without cross-link order.
    // h_halo_buffer_addr: compact halo buffer in DRAM, layout [H_top | H_bot | W_left | W_right].
    std::array<uint32_t, 16> region_progress_sem_addr = {};
    uint32_t h_halo_buffer_addr = 0;

    // Pin t_out_parallel=1 and fill the grid with H/W (then C) parallelism so every core walks the
    // full t-range. The reader's per-t-block halo wait then ramps, and interior (h,w) tiles touch no
    // device edge and need no halo — overlapping NP under conv3d compute. Falls back to temporal fill
    // when the spatial dims cannot fill the grid.
    bool force_spatial_parallel = false;

    // halo_last: bulk-core two-phase. The conv runs on the full output across all conv cores, but each
    // core processes its blocks in two passes — interior blocks first (no halo, overlapping NP), then
    // boundary blocks after the NP gate. Hides NP under the interior without reserving cores or paying
    // a per-pixel reuse penalty (boundary blocks are full H×W blocks). Uses the same progress-semaphore
    // gate as the temporal-overlap path.
    bool halo_last = false;

    // Hash only the base conv3d blocking fields — identical to Conv3dConfig. The two scheme flags above
    // are folded into the program hash by NpConv3dDeviceOperation::compute_program_hash (see struct
    // note), not here, so this config stays hash-compatible with standalone conv3d.
    static constexpr auto attribute_names = Conv3dConfig::attribute_names;
    auto attribute_values() const { return Conv3dConfig::attribute_values(); }
};

struct NpConv3dParams {
    // NP topology: H-fabric and optional W-fabric halo exchange
    uint32_t np_padding_h;     // H padding per side (1 for k333)
    uint32_t np_padding_w;     // W padding per side (0 if W-halo not needed)
    uint32_t np_cluster_axis;  // mesh axis for H parallelism
    uint32_t np_ring_size;     // number of H-parallel devices
    ttnn::ccl::Topology np_topology;
    GlobalSemaphore h_neighbor_semaphore;
    GlobalSemaphore barrier_semaphore;
    std::optional<uint32_t> np_pad_dim2;  // W-axis dim index (optional)
    uint32_t np_pad2_left = 0;
    uint32_t np_pad2_right = 0;
    std::optional<uint32_t> np_pad2_cluster_axis;
    size_t np_num_links = 2;
    size_t np_pad2_num_links = 2;
    tt::tt_metal::MemoryConfig np_output_mem_config;
    GlobalSemaphore w_neighbor_semaphore;

    // Conv3d kernel params (same as Conv3dParams; Np subclass carries the fused-only controls)
    NpConv3dConfig conv_config;
    tt::tt_metal::MemoryConfig conv_output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::DataType dtype;
    uint32_t output_channels;
    std::array<uint32_t, 3> kernel_size;
    std::array<uint32_t, 3> stride;
    std::array<uint32_t, 3> padding;
    std::array<uint32_t, 3> dilation;
    std::string padding_mode;
    uint32_t groups = 1;

    // Constructor required because GlobalSemaphore is not default constructible
    NpConv3dParams(
        uint32_t np_padding_h_,
        uint32_t np_padding_w_,
        uint32_t np_cluster_axis_,
        uint32_t np_ring_size_,
        ttnn::ccl::Topology np_topology_,
        const GlobalSemaphore& h_neighbor_semaphore_,
        const GlobalSemaphore& barrier_semaphore_,
        const GlobalSemaphore& w_neighbor_semaphore_,
        std::optional<uint32_t> np_pad_dim2_,
        uint32_t np_pad2_left_,
        uint32_t np_pad2_right_,
        std::optional<uint32_t> np_pad2_cluster_axis_,
        size_t np_num_links_,
        size_t np_pad2_num_links_,
        tt::tt_metal::MemoryConfig np_output_mem_config_,
        const NpConv3dConfig& conv_config_,
        tt::tt_metal::MemoryConfig conv_output_mem_config_,
        DeviceComputeKernelConfig compute_kernel_config_,
        tt::tt_metal::DataType dtype_,
        uint32_t output_channels_,
        const std::array<uint32_t, 3>& kernel_size_,
        const std::array<uint32_t, 3>& stride_,
        const std::array<uint32_t, 3>& padding_,
        const std::array<uint32_t, 3>& dilation_,
        const std::string& padding_mode_,
        uint32_t groups_) :
        np_padding_h(np_padding_h_),
        np_padding_w(np_padding_w_),
        np_cluster_axis(np_cluster_axis_),
        np_ring_size(np_ring_size_),
        np_topology(np_topology_),
        h_neighbor_semaphore(h_neighbor_semaphore_),
        barrier_semaphore(barrier_semaphore_),
        np_pad_dim2(np_pad_dim2_),
        np_pad2_left(np_pad2_left_),
        np_pad2_right(np_pad2_right_),
        np_pad2_cluster_axis(np_pad2_cluster_axis_),
        np_num_links(np_num_links_),
        np_pad2_num_links(np_pad2_num_links_),
        np_output_mem_config(std::move(np_output_mem_config_)),
        w_neighbor_semaphore(w_neighbor_semaphore_),
        conv_config(conv_config_),
        conv_output_mem_config(std::move(conv_output_mem_config_)),
        compute_kernel_config(compute_kernel_config_),
        dtype(dtype_),
        output_channels(output_channels_),
        kernel_size(kernel_size_),
        stride(stride_),
        padding(padding_),
        dilation(dilation_),
        padding_mode(padding_mode_),
        groups(groups_) {}

    // Hash: NP topology fields + conv3d config (excluding per-call addresses)
    static constexpr auto attribute_names = std::make_tuple(
        "np_padding_h",
        "np_padding_w",
        "np_cluster_axis",
        "np_ring_size",
        "np_topology",
        "np_num_links",
        "np_pad2_num_links",
        "conv_config",
        "dtype",
        "output_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "padding_mode",
        "groups");

    auto attribute_values() const {
        return std::forward_as_tuple(
            np_padding_h,
            np_padding_w,
            np_cluster_axis,
            np_ring_size,
            np_topology,
            np_num_links,
            np_pad2_num_links,
            conv_config,
            dtype,
            output_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            padding_mode,
            groups);
    }
};

struct NpConv3dInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias_tensor;
    Tensor halo_buffer;  // compact halo buffer in DRAM (pre-allocated)
};

}  // namespace ttnn::experimental::prim
