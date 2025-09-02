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
        grid_sample(input_tensor: ttnn.Tensor, grid: ttnn.Tensor, *, mode: str = "bilinear", padding_mode: str = "zeros", use_precomputed_grid: bool = False, extend_channels: bool = False, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs grid sampling on the input tensor using the provided sampling grid.

        Grid sample uses bilinear interpolation to sample input values at arbitrary
        grid locations. This is commonly used for spatial transformations, image warping,
        and implementing spatial transformer networks.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (N, H_in, W_in, C) - channel last format
            grid (ttnn.Tensor): Sampling grid with flexible batching support:
                               * Standard mode (use_precomputed_grid=False):
                                 - Shape (N, H_out, W_out, 2*K) where K is the grid batching factor
                                 - Contains K sets of normalized coordinates in range [-1, 1]
                                 - Each coordinate pair (x, y): x=-1 (leftmost), x=+1 (rightmost), y=-1 (topmost), y=+1 (bottommost)
                                 - When K=1: standard single coordinate per location (maps 1:1 to PyTorch F.grid_sample behavior)
                                 - When K>1: multiple coordinate sets batched per grid location
                               * Precomputed mode (use_precomputed_grid=True):
                                 - Shape (N, H_out, W_out, 6*K) containing K sets of precomputed data
                                 - Each set has 6 elements: pixel coordinates and bilinear interpolation weights
                                 - Generated using ttnn.prepare_grid_sample_grid() for K=1, then ttnn.reshape() for K>1

        Keyword Args:
            mode (str): Interpolation mode. Currently only "bilinear" is supported.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            use_precomputed_grid (bool): Whether to use precomputed grid coordinates.
                                   When False (default): grid should be normalized coordinates in [-1, 1]
                                   When True: grid should be preprocessed using ttnn.prepare_grid_sample_grid()
            extend_channels (bool): Controls how grid batching factor K affects output dimensions:
                                   When True: extend channels (legacy behavior) - output shape (N, H_out, W_out, C*K)
                                   When False (default): extend W dimension - output shape (N, H_out, W_out*K, C)
                                   This flag decouples channel extension from grid batching behavior.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation.

        Returns:
            ttnn.Tensor: Output tensor shape depends on extend_channels flag:
                        - When extend_channels=False (default): (N, H_out, W_out*K, C) - W dimension extended
                        - When extend_channels=True: (N, H_out, W_out, C*K) - channels extended (legacy behavior)
                        Where K is the grid batching factor. When K=1, both behaviors produce the same output shape.

        Example:
            >>> # Create input tensor (N=1, H=4, W=4, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 4, 4, 32), device=device)

            >>> # Example 1: Standard single grid (K=1) - both extend_channels values produce same result
            >>> theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float)
            >>> grid = torch.nn.functional.affine_grid(theta, (1, 32, 4, 4), align_corners=False)
            >>> grid_tensor = ttnn.from_torch(grid.to(torch.bfloat16), device=device)
            >>> output_default = ttnn.grid_sample(input_tensor, grid_tensor)  # extend_channels=False (default)
            >>> output_channels = ttnn.grid_sample(input_tensor, grid_tensor, extend_channels=True)
            >>> print(output_default.shape)  # [1, 4, 4, 32] - same for both when K=1
            >>> print(output_channels.shape) # [1, 4, 4, 32] - same for both when K=1

            >>> # Example 2: Batched grid (K=4) - demonstrates different behaviors
            >>> # Create grid with 4 coordinate sets per location: shape (1, 4, 4, 8) where K=4
            >>> batched_grid = torch.randn(1, 4, 4, 8) * 0.5  # K=4 coordinate sets
            >>> batched_grid_tensor = ttnn.from_torch(batched_grid.to(torch.bfloat16), device=device)
            >>>
            >>> # extend_channels=False (default): W dimension extended
            >>> output_w_extend = ttnn.grid_sample(input_tensor, batched_grid_tensor)
            >>> print(output_w_extend.shape)  # [1, 4, 16, 32] - W extended from 4 to 16 (4*K)
            >>>
            >>> # extend_channels=True: channels extended (legacy behavior)
            >>> output_c_extend = ttnn.grid_sample(input_tensor, batched_grid_tensor, extend_channels=True)
            >>> print(output_c_extend.shape)  # [1, 4, 4, 128] - channels extended from 32 to 128 (32*K)

            >>> # Example 3: Using precomputed grid for better performance
            >>> grid_float32 = ttnn.from_torch(grid, dtype=ttnn.float32)
            >>> input_shape = [1, 4, 4, 32]  # [N, H, W, C] format
            >>> prepared_grid = ttnn.prepare_grid_sample_grid(
            ...     grid_float32, input_shape, padding_mode="zeros", output_dtype=ttnn.bfloat16
            ... )
            >>> prepared_grid = ttnn.to_device(prepared_grid, device)
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
            py::arg("extend_channels") = false,
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
                               Note: This function only supports unbatched grids (batching factor K=1).
                               For grid batching (K>1), use ttnn.reshape() to convert the output to
                               (N, H_out, W_out//K, 6*K) format for use with ttnn.grid_sample().
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
