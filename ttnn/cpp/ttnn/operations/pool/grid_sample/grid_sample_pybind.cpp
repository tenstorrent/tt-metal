// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "grid_sample.hpp"
#include "grid_sample_prepare_grid.hpp"

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
            grid (ttnn.Tensor): Sampling grid with two possible formats:
                               * Standard mode (use_precomputed_grid=False): Shape (N, H_out, W_out, 2) containing
                                 normalized coordinates in range [-1, 1]. The last dimension contains (x, y) coordinates where:
                                 - x=-1 corresponds to leftmost input column, x=+1 to rightmost input column
                                 - y=-1 corresponds to topmost input row, y=+1 to bottommost input row
                               * Precomputed mode (use_precomputed_grid=True): Shape (N, H_out, W_out, 6) containing
                                 precomputed pixel coordinates and bilinear interpolation weights from ttnn.prepare_grid_sample_grid()

        Keyword Args:
            mode (str): Interpolation mode. Currently only "bilinear" is supported.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            use_precomputed_grid (bool): Whether to use precomputed grid coordinates.
                                   When False (default): grid should be normalized coordinates in [-1, 1]
                                   When True: grid should be preprocessed using ttnn.prepare_grid_sample_grid()
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation.

        Returns:
            ttnn.Tensor: Output tensor of shape (N, H_out, W_out, C) - channel last format

        Example:
            >>> # Create input tensor (N=1, H=4, W=4, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 4, 4, 32), device=device)

            >>> # Create identity grid (should return input unchanged)
            >>> theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float)
            >>> grid = torch.nn.functional.affine_grid(theta, (1, 32, 4, 4), align_corners=False)
            >>> grid_tensor = ttnn.from_torch(grid.to(torch.bfloat16), device=device)

            >>> # Method 1: Standard grid sampling
            >>> output = ttnn.grid_sample(input_tensor, grid_tensor)
            >>> print(output.shape)  # [1, 4, 4, 32]

            >>> # Method 2: Using precomputed grid for better performance
            >>> # First, create float32 grid on host for preprocessing
            >>> grid_float32 = ttnn.from_torch(grid, dtype=ttnn.float32)
            >>> input_shape = [1, 4, 4, 32]  # [N, H, W, C] format
            >>>
            >>> # Precompute grid coordinates and bilinear weights
            >>> prepared_grid = ttnn.prepare_grid_sample_grid(
            ...     grid_float32, input_shape, padding_mode="zeros", output_dtype=ttnn.bfloat16
            ... )
            >>> # Move prepared grid to device
            >>> prepared_grid = ttnn.to_device(prepared_grid, device)
            >>>
            >>> # Apply grid sample with precomputed grid
            >>> output_precomputed = ttnn.grid_sample(input_tensor, prepared_grid, use_precomputed_grid=True)
            >>> print(output_precomputed.shape)  # [1, 4, 4, 32]
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

void bind_prepare_grid_sample_grid(py::module& module) {
    // Bind prepare_grid_sample_grid function
    module.def(
        "prepare_grid_sample_grid",
        prepare_grid_sample_grid,
        py::arg("grid"),
        py::arg("input_shape"),
        py::kw_only(),
        py::arg("padding_mode") = "zeros",
        py::arg("output_dtype") = std::nullopt,
        R"doc(
        prepare_grid_sample_grid(grid: ttnn.Tensor, input_shape: List[int], *, padding_mode: str = "zeros", output_dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Precomputes grid sample data for optimized kernel execution.

        This function takes a normalized grid tensor and precomputes the pixel coordinates
        and bilinear interpolation weights needed for grid sampling.

        Args:
            grid (ttnn.Tensor): Grid tensor of shape (N, H_out, W_out, 2) with normalized coordinates in [-1, 1]
            input_shape (List[int]): Input tensor dimensions [N, H_in, W_in, C] in NHWC format

        Keyword Args:
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            output_dtype (ttnn.DataType, optional): Data type for the output tensor. Default: bfloat16

        Returns:
            ttnn.Tensor: Precomputed grid tensor of shape (N, H_out, W_out, 6) where:
                        - [:, :, :, 0]: North-west height coordinate (as integer stored in bfloat16)
                        - [:, :, :, 1]: North-west width coordinate (as integer stored in bfloat16)
                        - [:, :, :, 2]: Weight for north-west pixel
                        - [:, :, :, 3]: Weight for north-east pixel
                        - [:, :, :, 4]: Weight for south-west pixel
                        - [:, :, :, 5]: Weight for south-east pixel

        Example:
            >>> # Create a normalized grid with coordinates in [-1, 1] range
            >>> torch_grid = torch.randn(1, 8, 8, 2) * 0.8  # Keep within valid range
            >>> grid = ttnn.from_torch(torch_grid, dtype=ttnn.float32)
            >>> input_shape = [1, 32, 32, 64]  # N, H, W, C format
            >>>
            >>> # Precompute grid for optimized sampling
            >>> precomputed_grid = ttnn.prepare_grid_sample_grid(
            ...     grid, input_shape, padding_mode="zeros", output_dtype=ttnn.bfloat16
            ... )
            >>> print(precomputed_grid.shape)  # [1, 8, 8, 6]
            >>>
            >>> # The output contains:
            >>> # precomputed_grid[:, :, :, 0] = North-west height coordinates
            >>> # precomputed_grid[:, :, :, 1] = North-west width coordinates
            >>> # precomputed_grid[:, :, :, 2] = North-west pixel weight
            >>> # precomputed_grid[:, :, :, 3] = North-east pixel weight
            >>> # precomputed_grid[:, :, :, 4] = South-west pixel weight
            >>> # precomputed_grid[:, :, :, 5] = South-east pixel weight
            >>>
            >>> # Use with grid_sample for better performance on repeated operations
            >>> precomputed_grid_device = ttnn.to_device(precomputed_grid, device)
            >>> output = ttnn.grid_sample(input_tensor, precomputed_grid_device, use_precomputed_grid=True)
        )doc");
}

}  // namespace detail

void py_bind_grid_sample(py::module& module) {
    detail::bind_grid_sample(module);
    detail::bind_prepare_grid_sample_grid(module);
}

}  // namespace ttnn::operations::grid_sample
