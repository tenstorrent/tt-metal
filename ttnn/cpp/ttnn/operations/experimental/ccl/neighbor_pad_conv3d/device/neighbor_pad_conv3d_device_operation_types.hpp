// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation_types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

struct NpConv3dParams {
    // NP topology (from NeighborPadAsyncParams — the fields needed for H-fabric-only path)
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
    size_t np_num_links = 1;  // always 1 when progress sem active
    size_t np_pad2_num_links = 1;
    tt::tt_metal::MemoryConfig np_output_mem_config;
    GlobalSemaphore w_neighbor_semaphore;

    // Conv3d kernel params (same as Conv3dParams)
    Conv3dConfig conv_config;
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
        const Conv3dConfig& conv_config_,
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
