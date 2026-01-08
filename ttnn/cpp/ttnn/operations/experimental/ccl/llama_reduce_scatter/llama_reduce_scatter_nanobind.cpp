// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "llama_reduce_scatter.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::ccl {

void bind_llama_reduce_scatter(nb::module_& mod) {
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
                num_links (number, optional): the number of links. Defaults to `3`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                topology (ttnn.Topology, optional): Communication topology to use. Defaults to `ttnn.Topology.Linear`.
                use_noc1_only (bool, optional): Force NOC1-only transport. Defaults to `False`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.llama_reduce_scatter(
                                tt_input_tensors_list[i],
                                tt_intermediate_tensors_list[i],
                                dim,
                                ccl_semaphore_handles[i],
                                worker_sub_device_id,
                                cluster_axis=1,
                                mesh_device=mesh_device,
                                num_links=num_links,
                                memory_config=output_mem_config))doc";

    using OperationType = decltype(ttnn::experimental::llama_reduce_scatter);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::llama_reduce_scatter,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& intermediate_packet_buffer,
               uint32_t dim,
               const GlobalSemaphore& cross_device_semaphore,
               const tt::tt_metal::SubDeviceId& subdevice_id,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               tt::tt_fabric::Topology topology,
               bool use_noc1_only) {
                return self(
                    input_tensor,
                    intermediate_packet_buffer,
                    dim,
                    cross_device_semaphore,
                    subdevice_id,
                    cluster_axis,
                    mesh_device,
                    num_links,
                    memory_config,
                    topology,
                    use_noc1_only);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("intermediate_packet_buffer").noconvert(),
            nb::arg("dim"),
            nb::arg("cross_device_semaphore"),
            nb::arg("subdevice_id"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = tt::tt_fabric::Topology::Linear,
            nb::arg("use_noc1_only") = false});
}

}  // namespace ttnn::operations::experimental::ccl
