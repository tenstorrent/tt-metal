// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/offset_cumsum/offset_cumsum.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail {
void bind_experimental_offset_cumsum_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Computes shifted cumulative sum of per-device expert histograms gathered across a cluster axis.

            Takes a 1D UINT32 tensor of shape [n_routed_experts] (per-device expert histogram),
            performs all_gather along the specified cluster_axis to produce a 2D tensor
            [num_devices, n_routed_experts], then computes a shifted prefix sum in integer arithmetic.

            Output is a 2D UINT32 tensor of shape [num_devices + 1, n_routed_experts] where row k
            contains the sum of rows 0..k-1 from the gathered input (row 0 is all zeros).

            Args:
                * :attr:`input_tensor`: 1D UINT32 tensor of expert counts [n_routed_experts].
                * :attr:`cluster_axis`: Axis along which to all_gather across devices.
                * :attr:`num_links`: Number of links for all_gather.
                * :attr:`memory_config`: Memory configuration for intermediate and output tensors.

        )doc";

    using OperationType = decltype(ttnn::offset_cumsum);
    bind_registered_operation(
        mod,
        ttnn::offset_cumsum,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               uint32_t cluster_axis,
               uint32_t num_links,
               const ttnn::MemoryConfig& memory_config) {
                return self(input_tensor, cluster_axis, num_links, memory_config);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("cluster_axis"),
            nb::arg("num_links"),
            nb::arg("memory_config")});
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail
