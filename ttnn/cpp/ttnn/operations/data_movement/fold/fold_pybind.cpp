// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/data_movement/fold/fold_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(py::module& module) {
    bind_registered_operation(module, ttnn::fold,
    R"doc(
            Fold TT Tensor.
            Input tensor must be on TT accelerator device, in ROW_MAJOR.
            Output tensor will be on TT accelerator device, in ROW_MAJOR.
            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"
                "input", "Input tensor", "Tensor", "Tensor of shape [N, H, W, C]", "Yes"
                "stride_h", "Stride along the H-dimension", "int", "", "Yes"
                "stride_w", "Stride along the W-dimension", "int", "", "Yes"
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fold)& op, const ttnn::Tensor& input, uint32_t stride_h, uint32_t stride_w,
                bool use_transpose_as_fold, std::optional<std::vector<uint32_t>> output_shape, uint32_t pad_c, uint32_t pad_h, uint32_t pad_w, std::optional<CoreCoord> grid_size, std::optional<uint32_t> override_shard_height,
                const uint8_t& queue_id)
                -> ttnn::Tensor {
                return op(queue_id, input, stride_h, stride_w, use_transpose_as_fold, output_shape, pad_c, pad_h, pad_w, grid_size, override_shard_height);
            },
            py::arg("input"),
            py::arg("stride_h"),
            py::arg("stride_w"),
            py::arg("use_transpose_as_fold") = false,
            py::arg("output_shape") = std::nullopt,
            py::arg("pad_c") = 0,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("grid_size") = std::nullopt,
            py::arg("override_shard_height") = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = 0 });
}

}  // namespace ttnn::operations::data_movement
