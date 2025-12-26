// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "image_rotate.hpp"

namespace ttnn::operations::image_rotate {

namespace py = pybind11;

void py_bind_image_rotate(py::module& module) {
    const auto doc = R"doc(
        Rotates an image tensor by an arbitrary angle around a specified center point using bilinear interpolation.

        The image_rotate operation performs spatial transformation by rotating each pixel position
        around a rotation center. Areas outside the rotated image are filled with a configurable
        fill value.

        Args:
            input_tensor (ttnn.Tensor): Input image tensor of shape (N, H, W, C) in NHWC format
            angle (float): Rotation angle in degrees. Positive values rotate counter-clockwise

        Keyword Args:
            center (Optional[Tuple[float, float]]): Rotation center point (cx, cy) in pixel coordinates.
                                                    Default: ((W-1)/2, (H-1)/2) - image center
            fill (float): Fill value for areas outside the rotated image. Default: 0.0
            expand (bool): If True, return error. Only False is supported (same output dimensions). Default: False
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Default: DRAM_INTERLEAVED

        Returns:
            ttnn.Tensor: Rotated image tensor of shape (N, H, W, C) - same as input

        Example:
            >>> # Create input tensor (N=1, H=256, W=256, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 256, 256, 32), device=device)
            >>>
            >>> # Rotate 45 degrees counter-clockwise around image center
            >>> output = ttnn.image_rotate(input_tensor, 45.0)
            >>> print(output.shape)  # [1, 256, 256, 32]
            >>>
            >>> # Rotate 90 degrees clockwise (negative angle)
            >>> output_cw = ttnn.image_rotate(input_tensor, -90.0)
            >>>
            >>> # Rotate around custom center with white fill
            >>> output_custom = ttnn.image_rotate(input_tensor, 30.0, center=(128, 128), fill=1.0)
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::image_rotate,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("angle"),
            py::kw_only(),
            py::arg("center") = std::nullopt,
            py::arg("fill") = 0.0f,
            py::arg("expand") = false,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::image_rotate
