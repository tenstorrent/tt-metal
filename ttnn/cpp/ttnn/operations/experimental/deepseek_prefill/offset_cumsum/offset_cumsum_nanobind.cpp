// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "offset_cumsum.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail {
void bind_experimental_offset_cumsum_operation(nb::module_& mod) {
    ttnn::bind_function<"offset_cumsum", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Computes shifted cumulative sum of per-device expert histograms gathered across a cluster axis.

            Takes a 1D UINT32 tensor of shape [n_routed_experts] (per-device expert histogram),
            performs all_gather along the specified cluster_axis to produce a 2D tensor
            [num_devices, n_routed_experts], then computes a shifted prefix sum in integer arithmetic.

            Returns a tuple of two UINT32 tensors:
              - dispatch_offsets: shape [num_devices, n_routed_experts] where row k contains the
                sum of rows 0..k-1 from the gathered input (row 0 is all zeros).
              - total_counts_per_expert: shape [1, n_routed_experts] containing the total count
                across all devices for each expert (i.e. the sum of all input rows).

            Args:
                * :attr:`input_tensor`: 1D UINT32 tensor of expert counts [n_routed_experts].
                * :attr:`cluster_axis`: Axis along which to all_gather across devices.
                * :attr:`num_links`: Number of links for all_gather.
                * :attr:`memory_config`: Memory configuration for intermediate and output tensors.

        )doc",
        &offset_cumsum,
        nb::arg("input_tensor").noconvert(),
        nb::arg("cluster_axis"),
        nb::arg("num_links"),
        nb::arg("memory_config"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail
