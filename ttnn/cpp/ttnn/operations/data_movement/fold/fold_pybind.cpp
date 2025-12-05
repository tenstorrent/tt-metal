// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/data_movement/fold/fold_pybind.hpp"

#include <array>
#include <variant>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::fold,
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
            [](const decltype(ttnn::fold)& op,
               const ttnn::Tensor& input,
               uint32_t stride_h,
               uint32_t stride_w,
               bool use_transpose_as_fold,
               std::optional<ttnn::Shape> output_shape,
               std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>> padding,
               std::optional<CoreRangeSet> grid_size,
               std::optional<MemoryConfig> override_memory_config) -> ttnn::Tensor {
                return op(
                    input,
                    stride_h,
                    stride_w,
                    use_transpose_as_fold,
                    output_shape,
                    padding,
                    grid_size,
                    override_memory_config);
            },
            py::arg("input"),
            py::arg("stride_h"),
            py::arg("stride_w"),
            py::arg("use_transpose_as_fold") = false,
            py::arg("output_shape") = std::nullopt,
            py::arg("padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("grid_size") = std::nullopt,
            py::arg("override_memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement
