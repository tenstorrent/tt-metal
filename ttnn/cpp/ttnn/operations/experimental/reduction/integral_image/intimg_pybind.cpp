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
        - x must be 4D with shape: [1, W, H, C]
          - Axis 0: batches (must be exactly 1)
          - Axis 1: width_of_images (W)
          - Axis 2: height_of_images (H)
          - Axis 3: number_of_channels (C) == number of images
        - Layout: 'tile' (required)
        - dtype: bfloat16 (required)
        - Only one input tensor is accepted.

        Returns
        -------
        Tensor
            A 4D tensor with the same shape/layout/dtype as `x`, where each element
            at (0, w, h, c) equals the inclusive sum of x over [0:w, 0:h] in channel c.

        Supported dtypes and layout for input tensor values:

        .. list-table::
            :header-rows: 1

            * - Dtypes
                - Layouts
            * - BFLOAT16
                - TILE

        Memory Support:
            - Interleaved: DRAM and L1

        Example:

        .. code-block:: python

            import ttnn
            import torch

            # Create a compliant tensor
            input_tensor = torch.Tensor([3, 1, 2])

            # Convert tensor to ttnn format
            input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

            # Golden tensor
            golden_intimg_tensor_ttnn = ttnn.cumsum(ttnn.cumsum(input_tensor_ttnn, 1), 2)

            # Integral image (still experimental)
            intimg_tensor_ttnn = ttnn.experimental.intimg(input_tensor_ttnn)

            # Compare PCC
            assert_with_pcc(golden_intimg_tensor_ttnn, intimg_tensor_ttnn, pcc=0.999)

        Notes
        -----
        - This feature is still *experimental*.
        - Summation is inclusive in both width and height.
        - Each channel is processed independently.

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
