// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "all_broadcast.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::ccl {

void bind_all_broadcast(nb::module_& mod) {
    const auto* doc =
        R"doc(
        All-broadcast operation across devices. This operation broadcasts data from all devices to all other devices in the mesh, returning a vector of tensors where each tensor contains the data from a corresponding device in the mesh. The output tensors are identical across all devices along the cluster axis if specified, otherwise they are identical across all devices in the mesh.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be broadcast.

        Keyword Args:
            cluster_axis (int, optional): The axis on the mesh device to broadcast across. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to input tensor memory config.
            num_links (int, optional): The number of links to use for the all-broadcast operation. Defaults to `1`.
            topology (ttnn.Topology, optional): Fabric topology. Defaults to `ttnn.Topology.Linear`.

        Returns:
            List[ttnn.Tensor]: A list of tensors, one from each device, where each tensor has the same shape as the input.

        Example:

            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                            torch.rand([1, 1, 32, 256]),
                            dtype=ttnn.bfloat16,
                            device=mesh_device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            >>> output = ttnn.all_broadcast(input_tensor)
            >>> # output is a list of 8 tensors, each with shape [1, 1, 32, 256]
    )doc";

    ttnn::bind_function<"all_broadcast">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::all_broadcast,
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear)));
}

}  // namespace ttnn::operations::ccl
