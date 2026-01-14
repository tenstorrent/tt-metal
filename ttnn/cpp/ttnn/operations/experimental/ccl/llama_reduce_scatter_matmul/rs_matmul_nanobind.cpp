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
        R"doc(
        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`topology` (ttnn.Topology): Communication topology for the reduce-scatter stage. Defaults to `ttnn.Topology.Linear`.
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
        ttnn::nanobind_arguments_t{
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
