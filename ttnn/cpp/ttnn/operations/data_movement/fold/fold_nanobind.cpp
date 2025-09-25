// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fold/fold_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
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
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::fold)& op,
               const ttnn::Tensor& input,
               uint32_t stride_h,
               uint32_t stride_w,
               bool use_transpose_as_fold,
               std::optional<ttnn::Shape> output_shape,
               uint32_t pad_c,
               uint32_t pad_h,
               uint32_t pad_w,
               std::optional<CoreRangeSet> grid_size,
               std::optional<MemoryConfig> override_memory_config) -> ttnn::Tensor {
                return op(
                    input,
                    stride_h,
                    stride_w,
                    use_transpose_as_fold,
                    output_shape,
                    pad_c,
                    pad_h,
                    pad_w,
                    grid_size,
                    override_memory_config);
            },
            nb::arg("input"),
            nb::arg("stride_h"),
            nb::arg("stride_w"),
            nb::arg("use_transpose_as_fold") = false,
            nb::arg("output_shape") = nb::none(),
            nb::arg("pad_c") = 0,
            nb::arg("pad_h") = 0,
            nb::arg("pad_w") = 0,
            nb::arg("grid_size") = nb::none(),
            nb::arg("override_memory_config") = nb::none()});
}

}  // namespace ttnn::operations::data_movement
