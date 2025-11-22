// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rs_matmul_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

// TODO: Update with all_gather_matmul docs
void bind_rs_matmul(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::llama_rs_matmul,
        R"doc(all_gather_matmul(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.all_gather_matmul(tensor, weight_tensor, dim=0, (0, 0))

        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::llama_rs_matmul)& self,
               const ttnn::Tensor& input_tensor,               // mm0 used
               const ttnn::Tensor& weight_tensor,              // mm1 used
               ttnn::Tensor& intermediate_packet_buffer,       // rs2
               int32_t dim,                                    // rs3
               const GlobalSemaphore& cross_device_semaphore,  // rs4
               const uint32_t cluster_axis,                    // rs 5
               const MeshDevice& mesh_device,                  // rs 6
               const uint32_t num_links,                       // rs 7 default 1
               const tt::tt_metal::SubDeviceId& subdevice_id,
               const std::optional<const ttnn::Tensor>& second_weight_tensor,
               const std::optional<const ttnn::Tensor>& rs_tensor,  // rs1
               tt::tt_fabric::Topology topology,
               const std::optional<ttnn::MemoryConfig>& memory_config_rs,  // rs 8 default std::nullopt
               const std::optional<ttnn::MemoryConfig>& memory_config_mm,  // mm4 used but default std::nullopt
               const std::optional<const ttnn::DeviceComputeKernelConfig>&
                   compute_kernel_config,                                   // mm8 used but default std::nullopt
               const std::optional<const GlobalCircularBuffer>& global_cb,  // mm12 used but default std::nullopt
               std::optional<const ttnn::CoreGrid>& core_grid,              // mm9 may use but default std::nullopt
               const bool transpose_a,                                      // mm2 set false
               const bool transpose_b,                                      // mm3 set false
               const std::optional<const DataType>& dtype,                  // mm5 set false
               const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // mm6 std::nullopt
               const std::optional<const std::string>& activation,                                  // mm7 set false
               const std::optional<const tt::tt_metal::Tile>& output_tile,                          // mm10 std::nullopt
               std::optional<Tensor>& optional_output_tensor,                                       // mm11 std::nullopt
               bool use_noc1_only

               ) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    weight_tensor,
                    intermediate_packet_buffer,
                    dim,
                    cross_device_semaphore,
                    cluster_axis,
                    mesh_device,
                    num_links,
                    subdevice_id,
                    second_weight_tensor,
                    rs_tensor,
                    topology,
                    memory_config_rs,
                    memory_config_mm,
                    compute_kernel_config,
                    global_cb,
                    core_grid,
                    transpose_a,
                    transpose_b,
                    dtype,
                    program_config,
                    activation,
                    output_tile,
                    optional_output_tensor,
                    use_noc1_only);
            },
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("intermediate_packet_buffer"),
            nb::arg("dim"),
            nb::arg("cross_device_semaphore"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("num_links"),
            nb::arg("subdevice_id"),
            nb::kw_only(),
            nb::arg("second_weight_tensor") = nb::none(),
            nb::arg("rs_tensor") = nb::none(),
            nb::arg("topology") = tt::tt_fabric::Topology::Linear,
            nb::arg("memory_config_rs") = nb::none(),
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("global_cb") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("use_noc1_only") = false});
}
}  // namespace ttnn::operations::experimental::ccl
