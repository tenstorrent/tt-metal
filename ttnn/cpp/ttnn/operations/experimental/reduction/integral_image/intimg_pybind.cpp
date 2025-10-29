// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_pybind.hpp"
#include "intimg.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reduction::detail {
void bind_reduction_intimg_operation(py::module& module) {
    auto docstring =
        R"doc(
            Compute the summed-area table (integral image) per channel.

            This op takes a single 4D tensor and returns, for every pixel (w, h) in each
            channel, the sum of all values in the rectangle [0..w, 0..h], inclusive.

            Constraints
            ----------
            * x must be 4D with shape: [1, W, H, C]
            - Axis 0: batches (must be exactly 1)
            - Axis 1: width_of_images (W)
            - Axis 2: height_of_images (H)
            - Axis 3: number_of_channels (C) == number of images
            * Layout: 'tile' (required)
            * dtype: bfloat16 (required)
            * Only one input tensor is accepted.

            Returns
            -------
            Tensor
                A 4D tensor with the same shape/layout/dtype as `x`, where each element
                at (0, w, h, c) equals the inclusive sum of x over [0:w, 0:h] in channel c.

            Notes
            -----
            * Summation is inclusive in both width and height.
            * Each channel is processed independently.
            * NaNs (if present) haven't been tested.

            Raises
            ------
            RuntimeError
                If shape is not 4D, batch != 1, layout != 'tile', or dtype != bfloat16.
            TypeError
                If `x` is not a tensor type accepted by this binding.

            Example
            -------
            >>> # x: [1, 4, 3, 2] -> 1 batch, width=4, height=3, channels=2
            >>> y = ttnn.experimental.intimg(x)
            >>> y.shape
            (1, 4, 3, 2)
        )doc";

    using OperationType = decltype(ttnn::experimental::intimg);
    bind_registered_operation(
        module,
        ttnn::experimental::intimg,
        docstring,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor) -> Tensor { return self(input_tensor); },
            py::arg("input_tensor").noconvert(),
            py::kw_only()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
