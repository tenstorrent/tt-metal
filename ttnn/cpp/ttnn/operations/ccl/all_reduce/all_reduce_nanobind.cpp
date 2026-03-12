// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "all_reduce.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void bind_all_reduce(nb::module_& mod) {
    const auto* doc =
        R"doc(
        All-reduce operation across devices with Sum reduction. If cluster axis is specified, the all-reduce is performed on tensor shards along the cluster axis, resulting in identical tensor shards across all devices along the cluster axis. If it is not specified, then we reduce across all devices in the mesh. All-reduce is a collective operation that reduces data from all devices using the Sum operation and returns the result to all devices.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be reduced.

        Keyword Args:
            cluster_axis (int, optional): The axis on the mesh device to reduce across. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            num_links (int, optional): Number of links to use for the all_reduce operation. Defaults to `None`.
            topology (ttnn.Topology, optional): Fabric topology. Defaults to `None`.

        Returns:
            ttnn.Tensor: The reduced tensor with the same shape as the input tensor. The output tensor is identical across all devices along the cluster axis if specified, otherwise it is identical across all devices in the mesh.

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
            >>> output = ttnn.all_reduce(ttnn_tensor)
            >>> print(output.shape)
            [1, 1, 32, 256]
        )doc";

    ttnn::bind_function<"all_reduce">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::all_reduce,
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology") = nb::none()));
}

}  // namespace ttnn::operations::ccl
