// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
            use_l1_small_for_semaphores (bool, optional): If True, allocate internal global semaphores in L1_SMALL instead of L1 to reduce L1 fragmentation. Defaults to `False`.

        Returns:
            List[ttnn.Tensor]: A list of tensors, one from each device, where each tensor has the same shape as the input.

        Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, BFLOAT8_B, FLOAT32
                  - TILE, ROW_MAJOR

            All-broadcast is a data-movement collective and does not restrict the input dtype; each output tensor preserves the input dtype. When ``cluster_axis`` is not specified the tensor must have a line (1D) topology.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: WIDTH_SHARDED, HEIGHT_SHARDED, BLOCK_SHARDED (each may reside in DRAM or L1; the op places no buffer-type restriction)
    )doc";

    ttnn::bind_function<"all_broadcast">(
        mod,
        doc,
        &ttnn::all_broadcast,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
        nb::arg("use_l1_small_for_semaphores") = false);
}

}  // namespace ttnn::operations::ccl
