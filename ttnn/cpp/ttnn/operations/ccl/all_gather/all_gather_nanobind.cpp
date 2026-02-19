// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "all_gather.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void bind_all_gather(nb::module_& mod) {
    const auto* doc =
        R"doc(
        All-gather operation across devices along a selected dimension and optional cluster axis. If cluster axis is specified then we gather across the cluster axis, resulting in identical tensor shards across all devices along the cluster axis. If it is not specified, then we gather across all devices in the mesh. All-gather is a collective operation that gathers data from all devices into a new output tensor, concatenated along the specified `dim`. When cluster_axis is specified, each of the non-cluster_axis dimensions are performing independent all-gathers along the devices on the cluster axis. When the layout is row-major or we have tile padding on the gather dim, we use the composite all-gather implementation that falls back to all-broadcast.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be gathered.
            dim (int): Dimension along which to gather.

        Keyword Args:
            cluster_axis (int, optional): The axis on the mesh device to gather across. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor.
            num_links (int, optional): The number of links to use for the all-gather operation. Defaults to `None`, for which the number of links is determined automatically.
            topology (ttnn.Topology, optional): Fabric topology. Defaults to `None`.
            chunks_per_sync (int, optional): Hyperparameter.
            num_workers_per_link (int, optional): Hyperparameter.
            num_buffers_per_channel (int, optional): Hyperparameter.
            sub_core_grids (CoreRangeSet, optional): Specifies sub-core grid ranges for advanced core selection control. Default uses all the cores in the device.

        Returns:
            ttnn.Tensor: The gathered tensor, with output_shape = input_shape for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] * num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices along the mesh.

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
            >>> output = ttnn.all_gather(ttnn_tensor, dim=0)
            >>> print(output.shape)
            [8, 1, 32, 256]
        )doc";

    using OperationType = decltype(ttnn::all_gather);
    ttnn::bind_registered_operation(
        mod,
        ttnn::all_gather,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology,
               const std::optional<uint32_t> chunks_per_sync,
               const std::optional<uint32_t> num_workers_per_link,
               const std::optional<uint32_t> num_buffers_per_channel,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    subdevice_id,
                    memory_config,
                    optional_output_tensor,
                    num_links,
                    topology,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    sub_core_grids);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology") = nb::none(),
            nb::arg("chunks_per_sync") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

}  // namespace ttnn::operations::ccl
