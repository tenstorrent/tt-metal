// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_all_reduce_async(nb::module_& mod) {
    const auto* doc = R"doc(
        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices.  This operation requires a persistent
        fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-reduce operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-reduce operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_links (int, optional): Number of links to use for the all_reduce_async operation. Defaults to `None`, which indicates to the operation that it should choose. Note that this value will be ignored if there are fewer links available than requested.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Linear`.
            residual_tensor (ttnn.Tensor, optional): persistent-buffer overload only. When given, the reduction kernel returns `residual_tensor + sum(input_tensor over cluster_axis)` in one op (fused residual add). It must be L1 WIDTH_SHARDED with the output shard spec, the input dtype and the input padded shape. Defaults to `None` (plain all-reduce).
            one_shot (bool, optional): persistent-buffer overload only, 1D fabric. Reduce-scatter + all-gather in one op: the output cores are split into num_devices equal slices, each device unicasts every peer's slice of its partial straight into that peer's buffer, only the slice owner reduces (same kernel and chip order, so the result is bit-identical to the default), and the owner multicasts the reduced slice into every peer's output. About 2.5x fewer fabric packets on the busiest link than the default broadcast. Needs the output core count to be a multiple of num_devices, one tile row per shard, row-major shards. Takes precedence over live_rows_packing. Defaults to `False`.
            live_rows_packing (bool, optional): persistent-buffer overload only. When the input has <= 16 logical rows in a single 32-row bf16 tile row (width-sharded, one tile row per shard), send only the top half (faces 0/1) of each tile, four half-tiles per fabric packet. Rows 0..15 of the result are byte-identical to the default; rows 16..31 are undefined. Ineligible inputs keep the default path. Defaults to `False`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                    full_tensor,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                )
            >>> output = ttnn.experimental.all_reduce_async(input_tensor, topology=ttnn.Topology.Linear)

        )doc";

    ttnn::bind_function<"all_reduce_async", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                uint32_t,
                const std::vector<GlobalSemaphore>&,
                const std::vector<GlobalSemaphore>&,
                const std::vector<GlobalSemaphore>&,
                reduction_common::ReduceType,
                const std::optional<ttnn::MemoryConfig>&,
                ttnn::ccl::Topology,
                std::optional<size_t>,
                std::optional<tt::tt_metal::SubDeviceId>>(ttnn::experimental::all_reduce_async),
            nb::arg("input_tensor"),
            nb::arg("num_devices"),
            nb::arg("barrier_semaphores"),
            nb::arg("rs_global_semaphores"),
            nb::arg("ag_global_semaphores"),
            nb::arg("math_op"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                std::optional<std::uint32_t>,
                MeshDevice&,
                const std::optional<std::vector<GlobalSemaphore>>&,
                const std::optional<std::vector<GlobalSemaphore>>&,
                const std::optional<std::vector<GlobalSemaphore>>&,
                reduction_common::ReduceType,
                const std::optional<ttnn::MemoryConfig>&,
                std::optional<ttnn::ccl::Topology>,
                std::optional<size_t>,
                std::optional<tt::tt_metal::SubDeviceId>>(ttnn::experimental::all_reduce_async),
            nb::arg("input_tensor"),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("mesh_device"),
            nb::arg("barrier_semaphores") = nb::none(),
            nb::arg("rs_global_semaphores") = nb::none(),
            nb::arg("ag_global_semaphores") = nb::none(),
            nb::arg("math_op"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                ttnn::Tensor&,
                uint32_t,
                MeshDevice&,
                const GlobalSemaphore&,
                std::optional<const DataType>,
                const std::optional<ttnn::MemoryConfig>&,
                ttnn::ccl::Topology,
                std::optional<size_t>,
                std::optional<tt::tt_metal::SubDeviceId>,
                bool,
                bool,
                bool,
                const std::optional<ttnn::Tensor>&,
                bool,
                bool>(ttnn::experimental::all_reduce_async),
            nb::arg("input_tensor"),
            nb::arg("buffer_tensor"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_noc1_only") = false,
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("fp32_dest_acc") = false,
            nb::arg("residual_tensor") = nb::none(),
            nb::arg("live_rows_packing") = false,
            nb::arg("one_shot") = false));
}

}  // namespace ttnn::operations::experimental::ccl
