// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

// Standalone halo-only neighbor-pad: the fabric H+W halo exchange from the fused
// neighbor_pad_conv3d op with the conv3d stage removed. It writes ONLY the compact halo
// buffer [H-top | H-bot | W-left | W-right]; there is no interior copy and no conv. All
// conv params (weights, blocking, kernel/stride/dilation, output channels) are gone —
// this op is pure fabric transport, benchmarked toward DRAM + fabric bandwidth.
struct NpHaloParams {
    // NP topology: H-fabric and W-fabric halo exchange
    uint32_t np_padding_h;     // H padding per side (1 for k333)
    uint32_t np_padding_w;     // W padding per side
    uint32_t np_cluster_axis;  // mesh axis for H parallelism
    uint32_t np_ring_size;     // number of H-parallel devices
    ttnn::ccl::Topology np_topology;
    GlobalSemaphore h_neighbor_semaphore;
    GlobalSemaphore barrier_semaphore;
    GlobalSemaphore w_neighbor_semaphore;
    std::optional<uint32_t> np_pad_dim2;  // W-axis dim index (required for the 2D compact layout)
    uint32_t np_pad2_left = 0;
    uint32_t np_pad2_right = 0;
    std::optional<uint32_t> np_pad2_cluster_axis;
    size_t np_num_links = 2;
    size_t np_pad2_num_links = 2;
    tt::tt_metal::MemoryConfig np_output_mem_config;
    std::string padding_mode;
    // Padded-input mode (opt-in, default 0 = contiguous input). When >0 the input tensor is a padded
    // [.., H+2*input_pad_h, W+2*input_pad_w, C] buffer and the readers exchange the halo of its INTERIOR
    // (offset by input_pad_h/w, row stride = padded W, frame stride = padded H*W). Lets the copy-free
    // decode feed conv3d's padded-output buffer straight into the exchange with no interior repack.
    uint32_t input_pad_h = 0;
    uint32_t input_pad_w = 0;

    // Padded-output (fused) mode (opt-in, default false): additionally scatter the exchanged compact halo
    // + interior (from input) into a padded [.., H+2pH, W+2pW, C] output (tensor_args.padded_output),
    // folding halo_scatter into this op. Interior copy overlaps the fabric exchange; border waits on an
    // exchange-done sem. border_only: interior already present in padded_output, only the border is written.
    bool output_padded = false;
    bool border_only = false;

    // GlobalSemaphore is not default constructible, so an explicit constructor is required.
    NpHaloParams(
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
        const std::string& padding_mode_) :
        np_padding_h(np_padding_h_),
        np_padding_w(np_padding_w_),
        np_cluster_axis(np_cluster_axis_),
        np_ring_size(np_ring_size_),
        np_topology(np_topology_),
        h_neighbor_semaphore(h_neighbor_semaphore_),
        barrier_semaphore(barrier_semaphore_),
        w_neighbor_semaphore(w_neighbor_semaphore_),
        np_pad_dim2(np_pad_dim2_),
        np_pad2_left(np_pad2_left_),
        np_pad2_right(np_pad2_right_),
        np_pad2_cluster_axis(np_pad2_cluster_axis_),
        np_num_links(np_num_links_),
        np_pad2_num_links(np_pad2_num_links_),
        np_output_mem_config(std::move(np_output_mem_config_)),
        padding_mode(padding_mode_) {}

    // Hash: NP topology + padding geometry (semaphores are per-call addresses, excluded).
    static constexpr auto attribute_names = std::make_tuple(
        "np_padding_h",
        "np_padding_w",
        "np_cluster_axis",
        "np_ring_size",
        "np_topology",
        "np_pad2_left",
        "np_pad2_right",
        "np_num_links",
        "np_pad2_num_links",
        "padding_mode",
        "input_pad_h",
        "input_pad_w",
        "output_padded",
        "border_only");

    auto attribute_values() const {
        return std::forward_as_tuple(
            np_padding_h,
            np_padding_w,
            np_cluster_axis,
            np_ring_size,
            np_topology,
            np_pad2_left,
            np_pad2_right,
            np_num_links,
            np_pad2_num_links,
            padding_mode,
            input_pad_h,
            input_pad_w,
            output_padded,
            border_only);
    }
};

struct NpHaloInputs {
    Tensor input_tensor;
    Tensor halo_buffer;  // compact halo buffer in DRAM (pre-allocated); the op's output in compact mode
    // Padded-output (fused) mode: the padded buffer the fused scatter writes and the op returns. In this
    // mode halo_buffer is an internal compact staging buffer. std::nullopt in compact mode.
    std::optional<Tensor> padded_output;
};

}  // namespace ttnn::experimental::prim
