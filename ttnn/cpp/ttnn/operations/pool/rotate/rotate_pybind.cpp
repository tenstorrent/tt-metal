// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "rotate.hpp"

namespace ttnn::operations::rotate {

namespace py = pybind11;

void py_bind_rotate(py::module& module) {
    const auto* const doc = R"doc(
        Rotates a tensor by an arbitrary angle around a specified center point using configurable interpolation.

        The rotate operation performs spatial transformation by rotating each pixel position
        around a rotation center. Areas outside the rotated tensor are filled with a configurable
        fill value.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (N, H, W, C) in NHWC format
            angle (float): Rotation angle in degrees. Positive values rotate counter-clockwise

        Keyword Args:
            center (Optional[Tuple[float, float]]): Rotation center point as (x, y) in pixel coordinates,
                                                    where x is the horizontal/width coordinate and y is
                                                    the vertical/height coordinate. Default: tensor center
                                                    at ((W-1)/2, (H-1)/2)
            fill (float): Fill value for areas outside the rotated tensor. Default: 0.0
            expand (bool): If True, return error. Only False is supported (same output dimensions). Default: False
            interpolation_mode (str): Interpolation method - only "nearest" (sharp, faster) is supported. Default: "nearest"
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Default: DRAM_INTERLEAVED

        Returns:
            ttnn.Tensor: Rotated tensor of shape (N, H, W, C) - same as input

        Example:
            >>> # Create input tensor (N=1, H=256, W=256, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 256, 256, 32), device=device)
            >>>
            >>> # Rotate 45 degrees counter-clockwise with nearest interpolation (default)
            >>> output = ttnn.rotate(input_tensor, 45.0)
            >>> print(output.shape)  # [1, 256, 256, 32]
            >>>
            >>> # Rotate 90 degrees clockwise with nearest interpolation (faster)
            >>> output_cw = ttnn.rotate(input_tensor, -90.0, interpolation_mode="nearest")
            >>>
            >>> # Rotate around custom center (x=128, y=64) with white fill
            >>> output_custom = ttnn.rotate(input_tensor, 30.0, center=(128, 64), fill=1.0, interpolation_mode="nearest")
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::rotate,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("angle"),
            py::kw_only(),
            py::arg("center") = std::nullopt,
            py::arg("fill") = 0.0f,
            py::arg("expand") = false,
            py::arg("interpolation_mode") = "nearest",
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::rotate
