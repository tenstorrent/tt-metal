// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "moe_routing_remap.hpp"
#include "moe_routing_remap_nanobind.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_moe_routing_remap(nb::module_& mod) {
    const auto* doc = R"doc(
        Splits the non-zero MoE routing weights among the devices along one axis of the mesh. The non-zero weights of
        `routing_weights_tensor`, in increasing expert order, form `expert_parallel_size` consecutive groups of
        ``non_zero_weight_size / expert_parallel_size`` weights. Device ``i`` along `cluster_axis`, counting from 0,
        keeps the weights of group ``i`` at their expert positions, and 0 for every other expert.

        For example, with 16 experts, ``non_zero_weight_size=8`` and 4 devices along `cluster_axis`
        (``expert_parallel_size=4``), each device keeps 2 weights:

        .. list-table::
           :header-rows: 1

           * - Tensor
             - Weights
           * - `routing_weights_tensor`, on every device
             - ``[0, 1, 0, 2, 0, 0, 3, 4, 0, 5, 0, 0, 6, 7, 0, 8]``
           * - Output of device 0 along `cluster_axis`
             - ``[0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
           * - Output of device 1 along `cluster_axis`
             - ``[0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]``
           * - Output of device 2 along `cluster_axis`
             - ``[0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 0]``
           * - Output of device 3 along `cluster_axis`
             - ``[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8]``

        On a single device, `expert_parallel_size` is 1 and the output equals `routing_weights_tensor`.

        Args:
            routing_weights_tensor (ttnn.Tensor): the routing weights, of shape ``[1, experts]`` with ``experts`` at most 65536, replicated on every device: one weight per expert. Exactly `non_zero_weight_size` weights must be non-zero, and the others ``0`` (``+0.0`` for BFLOAT16 and FLOAT32).
            non_zero_weight_size (int): the number of non-zero weights in `routing_weights_tensor`. Must be greater than 0, at most ``experts``, and a multiple of `expert_parallel_size`.
            expert_parallel_size (int): the number of groups to split the non-zero weights into. Must equal the number of devices along `cluster_axis` of the mesh (1 on a single device), and divide ``experts``.
            cluster_axis (int): the mesh axis to split the weights along: 0 or 1. The device at mesh coordinate ``(row, column)`` is device ``row`` along axis 0 and device ``column`` along axis 1.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. It must be interleaved. Defaults to `None`, which uses the memory configuration of `routing_weights_tensor`.
            optional_output_tensor (ttnn.Tensor, optional): Preallocated tensor for the output. It must have the shape, dtype and layout of `routing_weights_tensor`, and the output memory configuration that `memory_config` gives. Defaults to `None`.

        Returns:
            ttnn.Tensor: on each device, the weights of the device's group at their expert positions and 0 for the other experts, with the shape, dtype and layout of `routing_weights_tensor`.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - ROW_MAJOR

        Memory Support:
            - Interleaved: DRAM and L1
    )doc";

    ttnn::bind_function<"moe_routing_remap">(
        mod,
        doc,
        &ttnn::moe_routing_remap,
        nb::arg("routing_weights_tensor").noconvert(),
        nb::arg("non_zero_weight_size"),
        nb::arg("expert_parallel_size"),
        nb::arg("cluster_axis"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
