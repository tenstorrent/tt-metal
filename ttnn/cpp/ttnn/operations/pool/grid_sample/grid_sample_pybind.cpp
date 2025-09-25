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
        grid_sample(input_tensor: ttnn.Tensor, grid: ttnn.Tensor, *, mode: str = "bilinear", padding_mode: str = "zeros", use_precomputed_grid: bool = False, batch_output_channels: bool = False, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs grid sampling on the input tensor using the provided sampling grid.

        Grid sample uses bilinear interpolation to sample input values at arbitrary
        grid locations. This is commonly used for spatial transformations, image warping,
        and implementing spatial transformer networks.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (N, H_in, W_in, C) - channel last format
            grid (ttnn.Tensor): Sampling grid with flexible batching support:
                               * Standard mode (use_precomputed_grid=False):
                                 - Shape (N, H_grid, W_grid, 2*K) where K is the grid batching factor
                                 - Contains K sets of normalized coordinates in range [-1, 1] packed into the last dimension
                                 - Each coordinate pair (x, y): x=-1 (leftmost), x=+1 (rightmost), y=-1 (topmost), y=+1 (bottommost)
                                 - Data type: BFLOAT16 or FLOAT32 (FLOAT32 provides higher precision for coordinate calculations)
                                 - When K=1: standard single coordinate per location (maps 1:1 to PyTorch F.grid_sample behavior)
                                 - When K>1: K coordinate sets are packed per spatial location, typically created by reshaping
                                   a larger grid from (N, H_grid, W_grid*K, 2) to (N, H_grid, W_grid, 2*K), where the desired grid shape would be W_grid*K
                               * Precomputed mode (use_precomputed_grid=True):
                                 - Shape (N, H_grid, W_grid, 6*K) containing K sets of precomputed data packed into the last dimension
                                 - Each set has 6 elements: pixel coordinates and bilinear interpolation weights
                                 - Data type: BFLOAT16 only (precomputed grids must be BFLOAT16)
                                 - When K=1: standard precomputed grid
                                 - When K>1: K precomputed sets are packed per spatial location, typically created by reshaping
                                   a larger precomputed grid from (N, H_grid, W_grid*K, 6) to (N, H_grid, W_grid, 6*K), where the desired grid shape would be W_grid*K
                                 - Generated using ttnn.prepare_grid_sample_grid() for K=1, then ttnn.reshape() for K>1, both being done on host side

        Keyword Args:
            mode (str): Interpolation mode. Currently only "bilinear" is supported.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            use_precomputed_grid (bool): Whether to use precomputed grid coordinates.
                                   When False (default): grid should be normalized coordinates in [-1, 1]
                                   When True: grid should be preprocessed using ttnn.prepare_grid_sample_grid()
            batch_output_channels (bool): Controls how grid batching factor K affects output dimensions:
                                    When False (default): extend W dimension - output shape (N, H_grid, W_grid*K, C)
                                     The K coordinate sets produce K spatial outputs, expanding the width dimension
                                   When True: batch output channels - output shape (N, H_grid, W_grid, C*K)
                                     The K coordinate sets produce K channel groups, expanding the channel dimension
                                   Setting this argument to True requires for K (grid batching factor) to be larger than one
                                   Note: K doesn't disappear when batch_output_channels=False, it just gets distributed to the width dimension
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the operation.

        Returns:
            ttnn.Tensor: Output tensor shape depends on batch_output_channels flag:
                        - When batch_output_channels=False (default): (N, H_grid, W_grid*K, C) - W dimension extended
                        - When batch_output_channels=True: (N, H_grid, W_grid, C*K) - channels batched
                        Where K is the grid batching factor.

        Example:
            >>> # Create input tensor (N=1, H=4, W=4, C=32) - channel last format
            >>> input_tensor = ttnn.from_torch(torch.randn(1, 4, 4, 32), device=device)

            >>> # Example 1: Standard single grid (K=1) - both batch_output_channels values produce same result
            >>> theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float)
            >>> grid = torch.nn.functional.affine_grid(theta, (1, 32, 4, 4), align_corners=False)
            >>> grid_tensor = ttnn.from_torch(grid.to(torch.bfloat16), device=device)
            >>> output_default = ttnn.grid_sample(input_tensor, grid_tensor)  # batch_output_channels=False (default)
            >>> print(output_default.shape)  # [1, 4, 4, 32] - same for both when K=1

            >>> # Example 2: Grid batching (K=4) - demonstrates proper reshaping workflow
            >>> # Step 1: Create natural grid as you normally would (like in PyTorch)
            >>> K = 4  # Grid batching factor
            >>> natural_grid = torch.randn(1, 4, 16, 2) * 0.5  # Natural shape: (N, H_grid, W_grid*K, 2)
            >>>
            >>> # Step 2: Reshape for optimization - pack K coordinate sets into last dimension
            >>> W_grid = 16 // K  # W_grid = 4
            >>> batched_grid = natural_grid.view(1, 4, W_grid, 2*K)  # Reshaped: (1, 4, 4, 8)
            >>> batched_grid_tensor = ttnn.from_torch(batched_grid.to(torch.bfloat16), device=device)
            >>>
            >>> # batch_output_channels=False (default): W dimension extended
            >>> output_w_extend = ttnn.grid_sample(input_tensor, batched_grid_tensor)
            >>> print(output_w_extend.shape)  # [1, 4, 16, 32] - W extended from 4 to 16 (W_grid*K)
            >>>
            >>> # batch_output_channels=True: channels batched
            >>> output_c_extend = ttnn.grid_sample(input_tensor, batched_grid_tensor, batch_output_channels=True)
            >>> print(output_c_extend.shape)  # [1, 4, 4, 128] - channels batched from 32 to 128 (K*C)

            >>> # Example 3: Using FLOAT32 grid for higher precision
            >>> grid_float32 = ttnn.from_torch(grid.to(torch.float32), dtype=ttnn.float32, device=device)
            >>> output_float32 = ttnn.grid_sample(input_tensor, grid_float32)  # Higher precision coordinates
            >>> print(output_float32.shape)  # [1, 4, 4, 32]

            >>> # Example 4: Using precomputed grid for better performance
            >>> grid_float32_host = ttnn.from_torch(grid, dtype=ttnn.float32)
            >>> input_shape = [1, 4, 4, 32]  # [N, H, W, C] format
            >>> prepared_grid = ttnn.prepare_grid_sample_grid(
            ...     grid_float32_host, input_shape, padding_mode="zeros", output_dtype=ttnn.bfloat16
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
            py::arg("batch_output_channels") = false,
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
