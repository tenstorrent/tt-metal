// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_create_heads_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "llama_reduce_scatter_create_heads.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::ccl {

void bind_llama_rs_create_heads(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Reduce_scatter after FF1/3 for Llama70B.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                intermediate_packet_buffer (ttnn.Tensor): the intermediate packet buffer tensor.
                dim (number): the reduce dimension
                cross_device_semaphore (ttnn.GlobalSemaphore): the cross device semaphore.
                subdevice_id (ttnn.SubDeviceId): the subdevice id.
                cluster_axis (number): the cluster axis.
                mesh_device (ttnn.MeshDevice): the mesh device.
                topology (ttnn.Topology): Communication topology for the reduce-scatter stage.
                num_links (number, optional): the number of links. Defaults to `3`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.llama_rs_create_heads(
                                tt_input_tensors_list[i],
                                tt_intermediate_tensors_list[i],
                                dim,
                                ccl_semaphore_handles[i],
                                worker_sub_device_id,
                                cluster_axis=1,
                                mesh_device=mesh_device,
                                num_links=num_links,
                                memory_config=output_mem_config))doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::llama_rs_create_heads,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor").noconvert(),
            nb::arg("intermediate_packet_buffer").noconvert(),
            nb::arg("dim"),
            nb::arg("cross_device_semaphore"),
            nb::arg("subdevice_id"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("topology"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("num_heads"),
            nb::arg("num_kv_heads"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("qkv_memory_config") = nb::none(),
            nb::arg("use_noc1_only") = false,
            nb::arg("use_optimal_ccl_for_llama") = false});
}

}  // namespace ttnn::operations::experimental::ccl
