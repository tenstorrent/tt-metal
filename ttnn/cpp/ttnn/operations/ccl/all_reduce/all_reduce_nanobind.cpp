// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

        Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, BFLOAT8_B, FLOAT32
                  - TILE, ROW_MAJOR

            The reduction is performed in BFLOAT16; BFLOAT8_B inputs are up-cast to BFLOAT16 for the reduction and cast back, and FLOAT32 takes a dedicated reduction path. Input must be rank 2 or greater. The output preserves the input dtype and shape.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: supported in DRAM or L1. A sharded input is converted to interleaved internally (its buffer type is preserved), and the result is written back according to the requested output ``memory_config``.
        )doc";

    ttnn::bind_function<"all_reduce">(
        mod,
        doc,
        &ttnn::all_reduce,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::none());
}

}  // namespace ttnn::operations::ccl
