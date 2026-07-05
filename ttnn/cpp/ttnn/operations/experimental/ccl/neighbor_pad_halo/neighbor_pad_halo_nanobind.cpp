// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_halo_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_halo/neighbor_pad_halo.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_halo/halo_scatter.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_neighbor_pad_halo(nb::module_& mod) {
    ttnn::bind_function<"neighbor_pad_halo", "ttnn.experimental.">(
        mod,
        R"doc(
        Standalone halo-only neighbor-pad (no conv, no interior copy).

        Exchanges halo rows between neighboring devices via fabric into a compact
        pre-allocated DRAM halo buffer [H-top | H-bot | W-left | W-right], and returns
        that buffer. This is the fabric H+W exchange from neighbor_pad_conv3d with the
        conv3d stage removed — pure transport, benchmarked toward DRAM + fabric bandwidth.
        The caller must pre-allocate `halo_buffer` with the correct size (see NP design docs).

        Args:
            input (ttnn.Tensor): Unpadded activation tensor [B, T, H, W, C] row-major.
            halo_buffer (ttnn.Tensor): Pre-allocated compact DRAM halo buffer (also the output).
            np_padding_h (int): H-halo rows per side (typically 1 for k=3).
            np_padding_w (int): W-halo columns per side.
            np_cluster_axis (int): Mesh axis for H-parallel devices.
            np_num_links (int): Number of fabric links for the H exchange.
            np_topology (ttnn.Topology): Fabric topology (Linear or Ring).
            h_neighbor_semaphore (ttnn.GlobalSemaphore): H-neighbor handshake semaphore.
            barrier_semaphore (ttnn.GlobalSemaphore): NP barrier semaphore.
            w_neighbor_semaphore (ttnn.GlobalSemaphore): W-neighbor handshake semaphore.
            np_pad_dim2 (int): Secondary (W-axis) padding dimension index (must be > 0).
            np_pad2_left (int): Padding amount left for the secondary dim.
            np_pad2_right (int): Padding amount right for the secondary dim.
            np_pad2_cluster_axis (int): Cluster axis for the secondary dim.
            np_pad2_num_links (int): Links for the secondary dim.

        Keyword Args:
            padding_mode (str): "zeros" or "replicate". Defaults to "zeros".
            memory_config (ttnn.MemoryConfig, optional): Output memory config.

        Returns:
            ttnn.Tensor: The compact halo buffer (written in place).
        )doc",
        &ttnn::experimental::neighbor_pad_halo,
        nb::arg("input"),
        nb::arg("halo_buffer"),
        nb::arg("np_padding_h"),
        nb::arg("np_padding_w"),
        nb::arg("np_cluster_axis"),
        nb::arg("np_num_links"),
        nb::arg("np_topology"),
        nb::arg("h_neighbor_semaphore"),
        nb::arg("barrier_semaphore"),
        nb::arg("w_neighbor_semaphore"),
        nb::arg("np_pad_dim2"),
        nb::arg("np_pad2_left"),
        nb::arg("np_pad2_right"),
        nb::arg("np_pad2_cluster_axis"),
        nb::arg("np_pad2_num_links"),
        nb::kw_only(),
        nb::arg("padding_mode") = "zeros",
        nb::arg("memory_config") = nb::none());
}

void bind_halo_scatter(nb::module_& mod) {
    ttnn::bind_function<"halo_scatter", "ttnn.experimental.">(
        mod,
        R"doc(
        Local (no-fabric) repack for the persistent-padded activation pipeline.

        Allocates a padded buffer [B,T,H+2pH,W+2pW,C] and fills it in one pass: interior from
        interior_src (the unpadded activation) and border from the compact halo buffer
        [H-top | H-bot | W-left | W-right] produced by neighbor_pad_halo. Folds the old ttnn.pad
        (interior copy) + border scatter into one op; the next conv reads the result as a plain
        coalesced conv (pad=0).

        Args:
            compact_buffer (ttnn.Tensor): Compact halo buffer returned by neighbor_pad_halo (border source).
            interior_src (ttnn.Tensor): Unpadded activation [B,T,H,W,C] (interior source).
            np_padding_h (int): H-halo rows per side (must match the neighbor_pad_halo call).
            np_padding_w (int): W-halo columns per side.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory config.

        Returns:
            ttnn.Tensor: A newly-allocated padded buffer with interior + border filled.
        )doc",
        &ttnn::experimental::halo_scatter,
        nb::arg("compact_buffer"),
        nb::arg("interior_src"),
        nb::arg("np_padding_h"),
        nb::arg("np_padding_w"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
