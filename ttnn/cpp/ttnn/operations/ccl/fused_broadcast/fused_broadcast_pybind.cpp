// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_broadcast_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/fused_broadcast/fused_broadcast.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

namespace {

template <typename fused_broadcast_operation_t>
void bind_fused_broadcast(pybind11::module& module, const fused_broadcast_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const fused_broadcast_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& root_coord,
               const MeshCoordinate& mesh_shape,
               const ttnn::ccl::Topology topology,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               uint32_t num_links) -> ttnn::Tensor {
                return self(input_tensor, root_coord, mesh_shape, topology, memory_config, subdevice_id, num_links);
            },
            py::arg("input_tensor"),
            py::arg("root_coord") = MeshCoordinate{1, 0},
            py::arg("mesh_shape") = MeshCoordinate{4, 2},
            py::kw_only(),
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("num_links") = 1});
}

}  // namespace

void py_bind_fused_broadcast(pybind11::module& module) {
    bind_fused_broadcast(
        module,
        ttnn::fused_broadcast,
        R"doc(
        Performs a fused TP P2P replicate + SP tree broadcast operation for MLA+MoE latency optimization.


        On a 4x2 mesh with root at (1,0):
        1. TP P2P replicate: (1,0) -> (1,1)
        2. SP broadcast: (1,0) -> col 0, (1,1) -> col 1 (parallel)

        Args:
            input_tensor (ttnn.Tensor): Input tensor to broadcast (must be on root device).
            root_coord (MeshCoordinate, optional): Root device coordinate. Defaults to MeshCoordinate(1, 0).
            mesh_shape (MeshCoordinate, optional): Mesh dimensions. Defaults to MeshCoordinate(4, 2) for 4x2 mesh.

        Keyword Args:
            topology (ttnn.ccl.Topology, optional): Communication topology. Defaults to ttnn.ccl.Topology.Ring.
            semaphore (ttnn.GlobalSemaphore, optional): Coordination semaphore for TP synchronization.
            barrier_semaphore (ttnn.GlobalSemaphore, optional): Barrier semaphore for SP synchronization.
            subdevice_id (tt.tt_metal.SubDeviceId, optional): Optional sub-device ID.

        Returns:
            ttnn.Tensor: Output tensor containing broadcasted data on all mesh devices.

        Note:
            This operation is specifically optimized for 4x2 mesh topologies and requires:
            - Root coordinate in compute row (1 or 2) for optimal latency
            - Input tensor must be allocated on the root device
            - Designed for the MLA+MoE ultra-latency variant (4-collective, 10 rounds)

        Example:
            >>> # Create input tensor on root device (1,0)
            >>> input_tensor = ttnn.from_torch(
            ...     torch.randn([1, 1, 32, 256], dtype=torch.bfloat16),
            ...     device=mesh_device.get_device(MeshCoordinate(1, 0)),
            ...     dtype=ttnn.bfloat16,
            ...     layout=ttnn.TILE_LAYOUT
            ... )
            >>> # Perform fused broadcast
            >>> output_tensor = ttnn.fused_broadcast(
            ...     input_tensor,
            ...     root_coord=MeshCoordinate(1, 0),
            ...     mesh_shape=MeshCoordinate(4, 2),
            ...     topology=ttnn.ccl.Topology.Ring
            ... )
        )doc");
}

}  // namespace ttnn::operations::ccl
