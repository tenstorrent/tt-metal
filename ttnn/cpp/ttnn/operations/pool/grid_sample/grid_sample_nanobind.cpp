// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "grid_sample.hpp"
#include "grid_sample_prepare_grid.hpp"


namespace ttnn::operations::grid_sample {

namespace {

void bind_grid_sample_op(nb::module_& mod) {
    const auto* const doc = R"doc(
        Performs grid sampling on the input tensor using the provided sampling grid.

        Grid sample uses bilinear interpolation to sample input values at arbitrary
        grid locations. This is commonly used for spatial transformations, image warping,
        and implementing spatial transformer networks.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (N, H_in, W_in, C) - channel last format
            grid (ttnn.Tensor): Sampling grid with flexible batching support.

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
            mode (str): Interpolation mode.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            align_corners (bool): Whether to align corners when mapping normalized coordinates to pixel indices.
            use_precomputed_grid (bool): Whether to use precomputed grid coordinates.

                When False (default): grid should be normalized coordinates in [-1, 1]

                When True: grid should be preprocessed using ttnn.prepare_grid_sample_grid()

            batch_output_channels (bool): Controls how grid batching factor K affects output dimensions.

                When False (default): extend W dimension - output shape (N, H_grid, W_grid*K, C).
                The K coordinate sets produce K spatial outputs, expanding the width dimension.

                When True: batch output channels - output shape (N, H_grid, W_grid, C*K).
                The K coordinate sets produce K channel groups, expanding the channel dimension.

                Setting this argument to True requires for K (grid batching factor) to be larger than one.
                Note: K doesn't disappear when batch_output_channels=False, it just gets distributed to the width dimension.

            memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the operation.

        Returns:
            ttnn.Tensor: Output tensor shape depends on batch_output_channels flag.

                - When batch_output_channels=False (default): (N, H_grid, W_grid*K, C) - W dimension extended
                - When batch_output_channels=True: (N, H_grid, W_grid, C*K) - channels batched

                Where K is the grid batching factor.
        )doc";

    mod.def(
        "grid_sample",
        &ttnn::grid_sample,
        doc,
        nb::arg("input_tensor"),
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("mode") = nb::str("bilinear"),
        nb::arg("padding_mode") = nb::str("zeros"),
        nb::arg("align_corners") = false,
        nb::arg("use_precomputed_grid") = false,
        nb::arg("batch_output_channels") = false,
        nb::arg("memory_config") = nb::none());
}

void bind_prepare_grid_sample_grid(nb::module_& mod) {
    // Bind prepare_grid_sample_grid function
    mod.def(
        "prepare_grid_sample_grid",
        prepare_grid_sample_grid,
        nb::arg("grid"),
        nb::arg("input_shape"),
        nb::kw_only(),
        nb::arg("mode") = nb::str("bilinear"),
        nb::arg("padding_mode") = nb::str("zeros"),
        nb::arg("align_corners") = false,
        nb::arg("output_dtype") = nb::none(),
        R"doc(
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
            mode (str): Nearest or bilinear operation. Currently only "bilinear" is supported.
            padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
            align_corners (bool): Whether to align corners when sampling (default: false)
            output_dtype (ttnn.DataType, optional): Data type for the output tensor. Default: bfloat16

        Returns:
            ttnn.Tensor: Precomputed grid tensor of shape (N, H_out, W_out, 6)
        )doc");
}

}  // namespace

void bind_grid_sample(nb::module_& mod) {
    bind_grid_sample_op(mod);
    bind_prepare_grid_sample_grid(mod);
}

}  // namespace ttnn::operations::grid_sample
