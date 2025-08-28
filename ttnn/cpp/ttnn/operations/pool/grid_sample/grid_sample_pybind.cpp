// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "grid_sample.hpp"

namespace ttnn::operations::grid_sample {

namespace detail {
namespace py = pybind11;

void bind_grid_sample(py::module& module) {
    const auto doc = R"doc(
        grid_sample(input_tensor: ttnn.Tensor, grid: ttnn.Tensor, *, mode: str = "bilinear", padding_mode: str = "zeros", use_precomputed_grid: bool = False, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs grid sampling on the input tensor using the provided sampling grid.

        Grid sample uses bilinear interpolation to sample input values at arbitrary
        grid locations. This is commonly used for spatial transformations, image warping,
        and implementing spatial transformer networks.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (N, H_in, W_in, C) - channel last format
            grid (ttnn.Tensor): Sampling grid of shape (N, H_out, W_out, 2) containing
                               normalized coordinates in range [-1, 1]. The last dimension
                               contains (x, y) coordinates where:
                               * x=-1 corresponds to leftmost input column
                               * x=+1 corresponds to rightmost input column
                               * y=-1 corresponds to topmost input row
                               * y=+1 corresponds to bottommost input row

        Keyword Args:
            mode (str): Interpolation mode. Currently only "bilinear" is supported.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            use_precomputed_grid (bool): Whether to use precomputed grid coordinates. Currently only False is supported.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation.

        Returns:
            ttnn.Tensor: Output tensor of shape (N, H_out, W_out, C) - channel last format

        Example:
            >>> # Create input tensor (N=1, H=4, W=4, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 4, 4, 32), device=device)

            >>> # Create identity grid (should return input unchanged)
            >>> theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float)
            >>> grid = torch.nn.functional.affine_grid(theta, (1, 3, 4, 4), align_corners=False)
            >>> grid_tensor = ttnn.from_torch(grid, device=device)

            >>> # Apply grid sample
            >>> output = ttnn.grid_sample(input_tensor, grid_tensor)
            >>> print(output.shape)  # [1, 4, 4, 32]
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::grid_sample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("grid"),
            py::kw_only(),
            py::arg("mode") = "bilinear",
            py::arg("padding_mode") = "zeros",
            py::arg("use_precomputed_grid") = false,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace detail

void py_bind_grid_sample(pybind11::module& module) { detail::bind_grid_sample(module); }

}  // namespace ttnn::operations::grid_sample
