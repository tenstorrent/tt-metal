// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_conv3d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_conv3d/neighbor_pad_conv3d.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/neighbor_pad_conv3d_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

void bind_neighbor_pad_conv3d(nb::module_& mod) {
    ttnn::bind_function<"neighbor_pad_conv3d", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused NeighborPad (fabric-only H-halo) + Conv3d operation.

        Exchanges halo rows between H-parallel devices via a compact pre-allocated DRAM
        halo buffer, then performs Conv3d — all in a single device program dispatch.
        The caller must pre-allocate `halo_buffer` with the correct size (see NP design docs).

        The Python wrapper CCLManager.neighbor_pad_conv3d_fused in
        models/tt_dit/parallel/manager.py constructs the NpConv3dConfig and manages
        semaphore allocation; this binding exposes the raw C++ entry point.

        Args:
            input (ttnn.Tensor): Unpadded activation tensor [B, T, H, W, C] row-major.
            weight (ttnn.Tensor): Prepared conv3d weights [kD*kH*kW*C_in, C_out] tiled.
            bias (ttnn.Tensor, optional): Bias tensor [1, C_out] tiled.
            halo_buffer (ttnn.Tensor): Pre-allocated compact DRAM halo buffer.
            np_padding_h (int): H-halo rows per side (typically 1 for k=3).
            np_padding_w (int): W-halo columns per side (0 if not needed).
            np_cluster_axis (int): Mesh axis for H-parallel devices.
            np_num_links (int): Number of fabric links for NP.
            np_topology (ttnn.Topology): Fabric topology (Linear or Ring).
            h_neighbor_semaphore (ttnn.GlobalSemaphore): H-neighbor handshake semaphore.
            barrier_semaphore (ttnn.GlobalSemaphore): NP barrier semaphore.
            w_neighbor_semaphore (ttnn.GlobalSemaphore): W-neighbor handshake semaphore.
            np_pad_dim2 (int): Secondary (W-axis) padding dimension index; 0 = no 2D padding.
            np_pad2_left (int): Padding amount left for the secondary dim.
            np_pad2_right (int): Padding amount right for the secondary dim.
            np_pad2_cluster_axis (int): Cluster axis for the secondary dim (ignored when np_pad_dim2==0).
            np_pad2_num_links (int): Links for the secondary dim.
            conv_config (ttnn.NpConv3dConfig): NpConv3dConfig (a Conv3dConfig subclass; the
                fused-only flags are read only here) with halo flags set.
            output_channels (int): Number of output feature channels.
            kernel_size (List[int]): [kD, kH, kW].

        Keyword Args:
            stride (List[int]): [sD, sH, sW]. Defaults to [1, 1, 1].
            padding (List[int]): [pD, pH, pW] excluding halo. Defaults to [0, 0, 0].
            dilation (List[int]): [dD, dH, dW]. Defaults to [1, 1, 1].
            padding_mode (str): "zeros" or "replicate". Defaults to "zeros".
            groups (int): Depthwise group count. Defaults to 1.
            dtype (ttnn.DataType): Output dtype. Defaults to ttnn.bfloat16.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute config.
            memory_config (ttnn.MemoryConfig, optional): Output memory config.

        Returns:
            ttnn.Tensor: Output tensor [B, T_out, H_out, W_out, C_out].
        )doc",
        &ttnn::experimental::neighbor_pad_conv3d,
        nb::arg("input"),
        nb::arg("weight"),
        nb::arg("bias"),
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
        nb::arg("conv_config"),
        nb::arg("output_channels"),
        nb::arg("kernel_size"),
        nb::kw_only(),
        nb::arg("stride") = std::array<uint32_t, 3>{1u, 1u, 1u},
        nb::arg("padding") = std::array<uint32_t, 3>{0u, 0u, 0u},
        nb::arg("dilation") = std::array<uint32_t, 3>{1u, 1u, 1u},
        nb::arg("padding_mode") = "zeros",
        nb::arg("groups") = 1u,
        nb::arg("dtype") = tt::tt_metal::DataType::BFLOAT16,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

void bind_np_conv3d_config(nb::module_& mod) {
    using ttnn::experimental::prim::NpConv3dConfig;

    // Subclass of Conv3dConfig (already registered by bind_conv3d) so a config built/set in Python
    // works wherever a Conv3dConfig is accepted; these fields are read only by the fused op.
    auto py_np_config = nb::class_<NpConv3dConfig, ttnn::experimental::prim::Conv3dConfig>(
                            mod,
                            "NpConv3dConfig",
                            R"doc(
            Conv3dConfig plus the fused NeighborPad+Conv3d controls (halo buffer, progress
            semaphores, overlap scheme). Accepted anywhere a Conv3dConfig is.
            )doc")
                            .def(nb::init<>())
                            .def(
                                nb::init<
                                    DataType,
                                    Layout,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    std::array<uint32_t, 3>,
                                    uint32_t,
                                    CoreCoord>(),
                                nb::kw_only(),
                                nb::arg("weights_dtype") = DataType::BFLOAT16,
                                nb::arg("output_layout") = Layout::ROW_MAJOR,
                                nb::arg("T_out_block") = 1,
                                nb::arg("W_out_block") = 1,
                                nb::arg("H_out_block") = 1,
                                nb::arg("C_out_block") = 0,
                                nb::arg("C_in_block") = 0,
                                nb::arg("dilation") = std::array<uint32_t, 3>{1, 1, 1},
                                nb::arg("alignment") = 32,
                                nb::arg("compute_with_storage_grid_size") = nb::cast(CoreCoord{1, 1}));

    py_np_config.def_rw("h_halo_buffer_addr", &NpConv3dConfig::h_halo_buffer_addr, "");
    py_np_config.def_rw(
        "force_spatial_parallel",
        &NpConv3dConfig::force_spatial_parallel,
        "Pin t_out_parallel=1; fill grid with H/W parallelism (overlap-friendly, per-shape fallback).");
    py_np_config.def_rw(
        "halo_last",
        &NpConv3dConfig::halo_last,
        "Bulk-core two-phase: each conv core does its interior blocks first (overlap NP), then its "
        "boundary blocks after the NP gate. Hides NP without reserving cores or a 1-px reuse penalty.");
    py_np_config.def_rw(
        "region_progress_sem_addr",
        &NpConv3dConfig::region_progress_sem_addr,
        "Per-(region,link) progress GlobalSemaphore L1 addrs, indexed [region*num_links + link]; "
        "allocated/reset by the Python wrapper each call (not hashed).");
}

}  // namespace ttnn::operations::experimental::ccl
