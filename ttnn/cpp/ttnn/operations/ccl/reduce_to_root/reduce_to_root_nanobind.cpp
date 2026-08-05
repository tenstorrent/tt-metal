// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "reduce_to_root_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "reduce_to_root.hpp"

namespace ttnn::operations::ccl {

void bind_reduce_to_root(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Reduce-to-root operation. Performs sdpa tree reduction across 4 devices and stores the output on the root device only.

            Args:
                input_tensor_l (ttnn.Tensor): the SDPA values (l) state tensor, sharded across the 4 devices.
                input_tensor_s (ttnn.Tensor): the SDPA running-sum (s) state tensor, sharded across the 4 devices.
                input_tensor_m (ttnn.Tensor): the SDPA running-max (m) state tensor, sharded across the 4 devices.
                root_coord (ttnn.MeshCoordinate): Coordinate of the root device. Should be (1, 0) for the 4-device setup.

            Keyword Args:
                scale_fp32 (float, optional): scale applied during the reduction. Defaults to `1.0`.
                topology (ttnn.Topology, optional): Fabric topology. Defaults to `ttnn.Topology.Linear`.
                output_tensor_l (ttnn.Tensor, optional): Preallocated output tensor for values. Defaults to `None`.
                output_tensor_s (ttnn.Tensor, optional): Preallocated output tensor for sum. Defaults to `None`.
                output_tensor_m (ttnn.Tensor, optional): Preallocated output tensor for max. Defaults to `None`.
                intermediate_tensor (ttnn.Tensor, optional): Preallocated intermediate tensor. Defaults to `None`.
                input_mux_cores (List[ttnn.CoreCoord], optional): the 4 mux core coordinates used for the reduction. Defaults to `None`.

            Returns:
                List[ttnn.Tensor]: the reduced (l, s, m) tensors, each with the same spec as the corresponding input. The results are valid only on the root device.

            Supported dtypes and layouts:

                .. list-table::
                    :header-rows: 1

                    * - Tensor
                      - Dtypes
                      - Layouts
                    * - input_tensor_l / _s / _m
                      - BFLOAT16
                      - TILE

                reduce_to_root operates on a fixed 4-device line topology with the root at (1, 0). All three input tensors must be sharded; each output preserves the spec of its corresponding input.

            Memory Support:
                - Sharded: required (L1)
            )doc";

    ttnn::bind_function<"reduce_to_root">(
        mod,
        doc,
        &ttnn::reduce_to_root,
        nb::arg("input_tensor_l").noconvert(),
        nb::arg("input_tensor_s").noconvert(),
        nb::arg("input_tensor_m").noconvert(),
        nb::arg("root_coord"),
        nb::kw_only(),
        nb::arg("scale_fp32") = 1.0f,
        nb::arg("topology").noconvert() = nb::cast(tt::tt_fabric::Topology::Linear),
        nb::arg("output_tensor_l") = nb::none(),
        nb::arg("output_tensor_s") = nb::none(),
        nb::arg("output_tensor_m") = nb::none(),
        nb::arg("intermediate_tensor") = nb::none(),
        nb::arg("input_mux_cores") = nb::none());

    mod.def(
        "reduce_to_root_tensor_spec",
        ttnn::reduce_to_root_tensor_spec,
        nb::arg("input_tensor_l"),
        nb::arg("input_tensor_s"),
        nb::arg("input_tensor_m"),
        nb::arg("root_coord"),
        nb::arg("scale_fp32"),
        nb::arg("topology"),
        nb::arg("input_mux_cores") = nb::none());
}
}  // namespace ttnn::operations::ccl
