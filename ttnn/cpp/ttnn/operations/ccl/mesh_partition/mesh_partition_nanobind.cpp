// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mesh_partition.hpp"

namespace ttnn::operations::ccl {

void bind_mesh_partition(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Partitions the input tensor across the mesh such that each device has the i/num_devices-th partition of the input tensor along the specified dimension. This is the inverse of all_gather. When cluster axis is specified, we partition along the cluster axis.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int): the dimension to partition along.
            cluster_axis (int, optional): the cluster axis on the mesh. Defaults to `None`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: The partitioned tensor, with output_shape = input_shape for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] / num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices along the mesh.

        Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, BFLOAT8_B, FLOAT32
                  - TILE, ROW_MAJOR

            mesh_partition is a per-device slice (it does not use the fabric) and does not restrict the input dtype; the output preserves the input dtype. ``input_shape[dim]`` must be evenly divisible by the number of devices along the cluster axis.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: DRAM and L1
        )doc";

    ttnn::bind_function<"mesh_partition">(
        mod,
        doc,
        &ttnn::mesh_partition,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim"),
        nb::arg("cluster_axis") = nb::none(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::ccl
