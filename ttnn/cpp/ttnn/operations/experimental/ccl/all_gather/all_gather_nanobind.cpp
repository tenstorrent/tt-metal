// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/operations/experimental/ccl/all_gather/all_gather.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

ttnn::Tensor all_gather_wrapper(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::experimental::all_gather(
        input_tensor, dim, memory_config, output_tensor, cluster_axis, subdevice_id, sub_core_grids);
}

}  // namespace

void bind_all_gather(nb::module_& mod) {
    const auto* all_gather_doc = R"doc(
        Performs an all-gather collective operation that gathers data from all devices into a new output tensor, concatenated along the specified :attr:`dim`. If the :attr:`input_tensor` has unaligned row-major pages or padded tiles on the gather :attr:`dim`, a slower composite all-gather implementation is used.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be gathered.
            dim (int): Dimension along which to concatenate.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to the input tensor's memory config.
            output_tensor (ttnn.Tensor, optional): Pre-allocated output tensor, can improve performance if provided. This must be allocated before invoking any op to avoid races. Defaults to None (op allocates a new output tensor).
            cluster_axis (int, optional): Axis on the 2D mesh device grid to gather along. Each of the non-cluster_axis dimensions perform independent all-gathers along the devices on the cluster_axis. Irrelevant for 1D mesh grids.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores. Defaults to the first subdevice on the mesh device.
            sub_core_grids (CoreRangeSet, optional): Restricts worker core selection to this sub-grid. Defaults to all cores on the chosen subdevice.

        Returns:
            ttnn.Tensor: The gathered tensor, with output_shape = input_shape for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] * num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices in the mesh.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> output = ttnn.experimental.all_gather(ttnn_tensor, dim=0)
            >>> print(output.shape)
            [8, 1, 32, 256]
        )doc";

    ttnn::bind_function<"all_gather", "ttnn.experimental.">(
        mod,
        all_gather_doc,
        ttnn::overload_t(
            &all_gather_wrapper,
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::experimental::ccl
