// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/ccl/broadcast/broadcast.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

void bind_broadcast(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Performs a broadcast operation from a sender device to all other mesh devices across a cluster axis.

        Args:
            input_tensor (ttnn.Tensor): Input tensor. The data residing on ``sender_coord`` is the data that gets broadcast.
            sender_coord (ttnn.MeshCoordinate): Coordinate of the sender device in the mesh.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            cluster_axis (int, optional): The mesh axis to broadcast across. Defaults to `None`, in which case the tensor must have a line (1D) topology.
            num_links (int, optional): Number of links to use for the broadcast operation. Defaults to `None`, for which the number of links is determined automatically.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to the input tensor memory config.
            topology (ttnn.Topology, optional): Fabric topology. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Linear`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.

        Returns:
            ttnn.Tensor: The output tensor, same shape as the input, holding the sender device's data on every device along the cluster axis.

        Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, BFLOAT8_B, FLOAT32
                  - TILE, ROW_MAJOR

            Broadcast is a data-movement collective and does not restrict the input dtype; the output preserves the input dtype.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: WIDTH_SHARDED, HEIGHT_SHARDED, BLOCK_SHARDED (each may reside in DRAM or L1; the op places no buffer-type restriction)
        )doc";

    ttnn::bind_function<"broadcast">(
        mod,
        doc,
        &ttnn::broadcast,
        nb::arg("input_tensor"),
        nb::arg("sender_coord"),
        nb::kw_only(),
        nb::arg("num_links") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("subdevice_id") = nb::none());
}

}  // namespace ttnn::operations::ccl
