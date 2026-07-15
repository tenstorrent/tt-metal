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
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_neighbor_pad_halo(nb::module_& mod) {
    ttnn::bind_function<"neighbor_pad_halo", "ttnn.experimental.">(
        mod,
        R"doc(
        Fabric halo neighbor-pad: exchanges the H+W boundary rows between H/W-parallel mesh neighbors.

        Default (compact) mode: writes only the exchanged halo into a pre-allocated compact DRAM
        buffer [H-top | H-bot | W-left | W-right] and returns it — pure transport, bounded by DRAM
        read + fabric bandwidth. Pass `padded_output` for fold mode, which additionally scatters the
        full padded result in the same dispatch (interior copy overlaps the exchange) and returns it.
        The caller must pre-allocate `halo_buffer` (and `padded_output`, in fold mode) at the correct size.

        Args:
            input (ttnn.Tensor): Activation tensor [B, T, H, W, C] row-major (interior; may itself
                carry padding — see input_pad_h/w).
            halo_buffer (ttnn.Tensor): Pre-allocated compact DRAM halo buffer. In compact mode this is
                the returned tensor; in fold mode it is the internal staging buffer.
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
            input_pad_h (int): H-padding already present on `input`; the reader strides over it and the
                halo geometry reduces to the interior. Defaults to 0.
            input_pad_w (int): W-padding already present on `input`. Defaults to 0.
            padded_output (ttnn.Tensor, optional): Fold-mode target buffer; enables fold mode. Defaults to None.
            border_only (bool): Fold mode only — scatter just the border, skip the interior copy. Defaults to False.
            logical_h (int): Logical H extent per device; rows at or beyond it are masked to zero in-kernel
                (0 = no masking). Defaults to 0.
            logical_w (int): Logical W extent per device; cols at or beyond it are masked to zero. Defaults to 0.

        Returns:
            ttnn.Tensor: `padded_output` in fold mode, else the compact halo buffer (written in place).
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
        nb::arg("memory_config") = nb::none(),
        nb::arg("input_pad_h") = 0,
        nb::arg("input_pad_w") = 0,
        nb::arg("padded_output") = nb::none(),
        nb::arg("border_only") = false,
        nb::arg("logical_h") = 0,
        nb::arg("logical_w") = 0);
}

}  // namespace ttnn::operations::experimental::ccl
