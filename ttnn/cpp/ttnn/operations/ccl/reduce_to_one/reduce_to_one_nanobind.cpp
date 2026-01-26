// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "reduce_to_one_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reduce_to_one.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace nb = nanobind;

namespace ttnn::operations::ccl {

void bind_reduce_to_one(nb::module_& module) {
    const auto* doc =
        R"doc(
        Performs a sum reduction across all devices in a 2x4 mesh, with the final result on the root device.

        The reduction follows a 3-level tree structure:
        - Level 1: Within each column, leaf nodes (rows 0, 3) send to intermediate nodes (rows 1, 2)
        - Level 2: Within each column, root2 (row 2) sends to column root (row 1)
        - Level 3: Column root of column 0 sends to final root in column 1

        Args:
            input_tensor (ttnn.Tensor): Input tensor to reduce. Must be sharded across 8 cores (2 columns x 4 rows).
            root_coord (MeshCoordinate): Coordinate of the final root device where the result will be stored.
            topology (Topology): The fabric topology to use for communication.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): Optional pre-allocated output tensor. Defaults to None.
            intermediate_tensor (ttnn.Tensor, optional): Optional pre-allocated intermediate tensor for receiving data. Defaults to None.

        Returns:
            ttnn.Tensor: The reduced tensor (sum of all input shards) on the root device.
        )doc";

    using OperationType = decltype(ttnn::reduce_to_one);
    ttnn::bind_registered_operation(
        module,
        ttnn::reduce_to_one,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& root_coord,
               const tt::tt_fabric::Topology topology,
               std::optional<ttnn::Tensor>& output_tensor,
               std::optional<ttnn::Tensor>& intermediate_tensor) {
                return self(input_tensor, root_coord, topology, output_tensor, intermediate_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("root_coord"),
            nb::arg("topology"),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("intermediate_tensor") = nb::none()});
}

}  // namespace ttnn::operations::ccl
