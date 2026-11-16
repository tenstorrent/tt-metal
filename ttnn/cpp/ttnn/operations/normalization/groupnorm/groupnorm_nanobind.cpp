// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <tuple>
#include <cstdint>

#include "ttnn/device.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "groupnorm.hpp"
#include "groupnorm_grid_utils.hpp"
#include "groupnorm_input_mask.hpp"

namespace ttnn::operations::normalization::detail {

namespace {
void bind_normalization_group_norm_operation(nb::module_& mod) {
    const auto* doc = R"doc(
                Computes group_norm over :attr:`input_tensor`.
                See `Group Normalization <https://arxiv.org/abs/1803.08494>`_ for more details.

                .. math::
                    \text{group_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

                Where:
                    - :math:`\mu` and :math:`\sigma^2` are the mean and variance of the input tensor, respectively
                    - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
                    - :math:`\epsilon` is a small constant.

                GroupNorm traditionally operates by splitting the input tensor's channels into groups, and then computing the mean and variance of each group.
                This implementation is slightly different, in that it forms the groups using the tensor's last dimension.
                Concretely, the input tensor is expected to have a shape of [N, 1, H*W, C], where C is the dimension along which the groups are formed.

                TTNN provides utility functions to help prepare this op's inputs for different types of input tensors:
                    - When using sharded input tensors, :func:`ttnn.determine_expected_group_norm_sharded_config_and_grid_size` can provide the appropriate memory configuration and grid size.
                    - When using interleaved (DRAM) input tensors, :func:`ttnn.determine_expected_group_norm_dram_grid_size` can provide the appropriate grid size.
                    - :func:`ttnn.dram_group_norm_params_from_torch` is a convenience function that prepares the weight, bias, and input mask from PyTorch tensors for interleaved inputs.
                    - :func:`ttnn.get_group_norm_cores_across_channel` returns the number of cores that split the channel axis for a given memory layout, grid, and shard orientation. This value must be passed consistently to :func:`ttnn.create_group_norm_input_mask` and :func:`ttnn.create_group_norm_weight_bias_rm`. For HEIGHT_SHARDED inputs this is always 1; for BLOCK_SHARDED inputs it is ``core_grid.x`` when using ROW_MAJOR shard orientation, or ``core_grid.y`` when using COL_MAJOR.
                    - :func:`ttnn.create_group_norm_input_mask` creates the appropriate input mask for a given tensor dimension and group size.
                    - :func:`ttnn.create_group_norm_weight_bias_rm` converts the weight and bias tensors into appropriately padded and tiled inputs

                See the sharded example in this document for more details on how to properly prepare the op's inputs using these functions.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword args:
                num_groups (int): Number of groups to split the tensor's channels into.
                epsilon (float): Defaults to 1e-12.
                input_mask (ttnn.Tensor, optional): Defaults to `None`. When processing the inputs, the mask is used to only look at the elements of the current group.
                weight (ttnn.Tensor, optional): Gamma (scale) parameter for the affine transformation. When omitted, no scaling is applied. Defaults to `None`.
                bias (ttnn.Tensor, optional): Beta (shift) parameter for the affine transformation. When omitted, no shift is applied. Defaults to `None`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                dtype (ttnn.DataType, optional): Output data type. When provided, it must equal the :attr:`input_tensor` dtype; a mismatch is rejected. Defaults to `None` (output dtype matches the input).
                core_grid (CoreGrid, optional): Defaults to `None`.
                inplace (bool, optional): Defaults to `True`.
                output_layout (ttnn.Layout, optional): Defaults to `None`.
                num_out_blocks (int, optional): For non-sharded (interleaved) inputs, splits the per-core output height (``block_h``, in tiles) into ``num_out_blocks`` chunks so each iteration uses less SRAM, at the cost of performance. Ignored for sharded inputs. Should only be set if needed to relieve SRAM pressure. Accepted explicit values are ``-1`` (use the built-in auto-heuristic) or a chunk count in range ``[1, block_h]``. Defaults to `None`, whose meaning depends on :attr:`core_grid` (non-sharded inputs only): when :attr:`core_grid` is also `None` (auto-selected), ``num_out_blocks`` is determined automatically using the same auto-heuristic as ``-1``, and passing an explicit ``num_out_blocks`` in that case is rejected. When :attr:`core_grid` is provided and ``num_out_blocks`` is `None` (default), ``num_out_blocks`` defaults to ``1`` (no chunking).
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration for the op. Defaults to `None`.
                negative_mask (ttnn.Tensor, optional): Defaults to `None`. Can be used only in row-major sharded input/output tensors. Created with ttnn.create_group_norm_input_negative_mask. Used to reduce the number of CB's used in the sharded version of the kernel. When no tensor is passed the op synthesizes the mask in L1 if and only if the program would otherwise not fit in L1.
                use_welford (bool, optional): Defaults to `False`. If `True`, stable two-pass statistics with FP32 accumulation are used to compute the mean and variance. For non-tile-aligned ``H*W``, the operation falls back to the tile-reduction path so padding rows can be excluded correctly. The argument name is retained for compatibility.

            Returns:
                ttnn.Tensor: the output tensor.

            Note:
                The supported input data types and layouts:

                .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16, FLOAT32
                      - TILE, ROW_MAJOR

                ROW_MAJOR input is supported only for sharded inputs. An interleaved
                (non-sharded) input must be in TILE layout; convert it first with
                ``ttnn.to_layout(input, ttnn.TILE_LAYOUT)`` (and convert the output
                back with ``ttnn.to_layout`` if a ROW_MAJOR result is required).

                .. list-table:: weight (gamma) and bias (beta)
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16, FLOAT32
                      - ROW_MAJOR

                weight (gamma) and bias (beta) must share the same dtype.

                .. list-table:: input_mask
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16, BFLOAT8_B
                      - TILE

                The output dtype matches the :attr:`input_tensor` dtype (BFLOAT16 or FLOAT32); an explicit :attr:`dtype` must equal it. Both the layout and the memory configuration also match the :attr:`input_tensor`.

            Memory Support:
              - Interleaved: DRAM and L1
              - Sharded (L1): Height and Block sharded

            Limitations:
              - :attr:`input_tensor` is a 4D tensor of shape [N, 1, H*W, C] and is allocated on the device
              - For the :attr:`input_tensor`, C must be a multiple of the tile size (32) and divide evenly into :attr:`num_groups`.
              - For a TILE-layout :attr:`input_tensor`, the per-sample H*W need not be a multiple of the tile size (32); the fused kernel corrects for the tile-padding rows it reduces over. Residual error comes from the bfloat16 cancellation in ``Var - K*E[x]^2``, so it grows with *both* the padding fraction ``K = padded_HW / logical_HW - 1`` and the input's mean-to-spread ratio ``E[x]^2 / Var`` -- the latter dominates. Note ``K`` is not monotonic in H*W; it spikes just above every tile boundary, so H*W=33 (``K=0.94``) is worse than H*W=50 (``K=0.28``). Measured: with near-zero-mean inputs every shape tested stays inside the op's tolerance, including ``K=3`` (H*W=8) at 0.067 versus an aligned-control 0.041; with a large mean (uniform[0,1), ``E[x]^2/Var`` ~ 3) only H*W <= 16 (``K >= 1``) exceeds it. See #50682.
              - For a ROW_MAJOR :attr:`input_tensor`, H*W must still be a multiple of the tile size (32); non-multiples are rejected rather than silently approximated.
              - For the :attr:`input_mask`, C must match the number of groups, H must match a tile's height, and W must be a multiple of a tile's width.
              - If :attr:`core_grid` is not provided, it is inferred from the sharded input's shard grid or, for interleaved/DRAM input, via the same logic as :func:`ttnn.determine_expected_group_norm_dram_grid_size`. The inferred grid may not be optimal; pass :attr:`core_grid` explicitly to override it. To prepare a sharded input ahead of time and obtain a matching grid in one step, see :func:`ttnn.determine_expected_group_norm_sharded_config_and_grid_size`.
              - :attr:`inplace` is not supported for TILE-layout inputs and requires input and output layouts to be identical.
              - When generating inputs (e.g. weight, bias) for block sharded tensors, the number of cores in a column should draw upon core.x rather than core.y.
              - When generating inputs (e.g. weight, bias) for height sharded tensors, the number of cores in a column should be 1 rather than core.y.
              - Width-sharding is not supported
        )doc";

    ttnn::bind_function<"group_norm">(
        mod,
        doc,
        &ttnn::group_norm,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("num_groups"),
        nb::arg("epsilon") = 1e-12,
        nb::arg("input_mask") = nb::none(),
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("inplace") = true,
        nb::arg("output_layout") = nb::none(),
        nb::arg("num_out_blocks") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("negative_mask") = nb::none(),
        nb::arg("use_welford") = false);
    mod.def(
        "create_group_norm_input_mask",
        [](std::int64_t num_channel,
           std::int64_t num_groups,
           std::int64_t num_cores_across_channel,
           DataType data_type,
           std::int64_t tile_height,
           std::int64_t tile_width,
           std::int64_t rows_in_last_tile) {
            return create_group_norm_input_mask(
                num_channel,
                num_groups,
                num_cores_across_channel,
                data_type,
                tile_height,
                tile_width,
                rows_in_last_tile);
        },
        nb::arg("num_channel"),
        nb::arg("num_groups"),
        nb::arg("num_cores_across_channel"),
        nb::arg("data_type") = DataType::BFLOAT16,
        nb::arg("tile_height") = 32,
        nb::arg("tile_width") = 32,
        nb::arg("rows_in_last_tile") = 0,
        R"doc(
            C++ implementation of create_group_norm_input_mask.
            Returns a ttnn.Tensor of shape [1, num_groups, 32, 32*block_wt], dtype=ttnn.DataType.BFLOAT16.

            rows_in_last_tile (= ``logical_hw % 32``) is for non-tile-aligned H*W. It appends a
            second set of groups -- shape becomes [1, 2*num_groups, 32, 32*block_wt] -- identical
            to the first but with rows >= rows_in_last_tile zeroed, which group_norm selects on
            each batch's final row-tile. Leave at 0 for tile-aligned H*W; without it, group_norm
            derives the second set with a device-side multiply+concat on every call.
        )doc");
    mod.def(
        "create_group_norm_input_negative_mask",
        [](std::int64_t num_channel,
           std::int64_t num_groups,
           std::int64_t num_cores_across_channel,
           DataType data_type,
           std::int64_t tile_height,
           std::int64_t tile_width) {
            return create_group_norm_input_negative_mask(
                num_channel, num_groups, num_cores_across_channel, data_type, tile_height, tile_width);
        },
        nb::arg("num_channel"),
        nb::arg("num_groups"),
        nb::arg("num_cores_across_channel"),
        nb::arg("data_type") = DataType::BFLOAT16,
        nb::arg("tile_height") = 32,
        nb::arg("tile_width") = 32,
        R"doc(
            C++ implementation of create_group_norm_input_negative_mask.
            Returns a ttnn.Tensor of shape [1, num_groups, 32, 32*block_wt], dtype=ttnn.DataType.BFLOAT16.
        )doc");
    mod.def(
        "_compute_num_virtual_cols",
        [](std::uint32_t grid_x, int num_groups, std::uint32_t num_channels) -> std::uint32_t {
            return compute_num_virtual_cols(grid_x, num_groups, num_channels);
        },
        nb::arg("grid_x"),
        nb::arg("num_groups"),
        nb::arg("num_channels"),
        R"doc(
            Compute the number of virtual columns for DRAM group-norm.
            Finds the largest nvc <= min(grid_x, num_groups) such that
            (num_channels / nvc) % TILE_SIZE == 0 and num_groups % nvc == 0.
            Returns 0 if no valid value exists.
        )doc");
    mod.def(
        "_find_expected_dram_grid",
        [](std::uint32_t max_x,
           std::uint32_t max_y,
           std::uint32_t num_channels,
           int num_groups,
           std::uint32_t input_nhw,
           std::uint32_t num_batches) -> ttnn::CoreGrid {
            auto result = find_expected_dram_grid(max_x, max_y, num_channels, num_groups, input_nhw, num_batches);
            if (!result.has_value()) {
                throw std::runtime_error(
                    "Cannot find a valid DRAM group-norm grid for num_channels=" + std::to_string(num_channels) +
                    ", num_groups=" + std::to_string(num_groups) + ", input_nhw=" + std::to_string(input_nhw) +
                    ", num_batches=" + std::to_string(num_batches) + ", max_grid=(" + std::to_string(max_x) + ", " +
                    std::to_string(max_y) + ")");
            }
            return result.value();
        },
        nb::arg("max_x"),
        nb::arg("max_y"),
        nb::arg("num_channels"),
        nb::arg("num_groups"),
        nb::arg("input_nhw"),
        nb::arg("num_batches") = 1,
        R"doc(
            Find the largest valid CoreGrid within (max_x, max_y) bounds
            for DRAM interleaved group-norm. Raises if no valid grid exists.
            num_batches ensures uniform multicast group sizes across batches.
        )doc");
    mod.def(
        "determine_expected_group_norm_sharded_config_and_grid_size",
        [](ttnn::MeshDevice* device,
           std::uint32_t num_channels,
           int num_groups,
           std::uint32_t input_nhw,
           bool is_height_sharded,
           bool is_row_major) {
            TT_FATAL(
                device != nullptr,
                "determine_expected_group_norm_sharded_config_and_grid_size: device must not be null.");
            const auto grid = device->compute_with_storage_grid_size();
            auto result = ttnn::operations::normalization::determine_expected_group_norm_sharded_config_and_grid_size(
                grid, num_channels, num_groups, input_nhw, is_height_sharded, is_row_major);
            return std::make_tuple(result.memory_config, result.core_grid);
        },
        nb::kw_only(),
        nb::arg("device"),
        nb::arg("num_channels"),
        nb::arg("num_groups"),
        nb::arg("input_nhw"),
        nb::arg("is_height_sharded"),
        nb::arg("is_row_major") = false,
        R"doc(
    Derive sharded memory config and grid for group norm.

    - num_channels must be divisible by num_groups and 32 (tile width).
    - input_nhw is N*H*W in logical units; padded to core multiples.
    - If is_height_sharded: shard along NHW only; channels per core is all C.
      Otherwise: shard across channels and NHW (BLOCK_SHARDED).
    - is_row_major toggles shard shape orientation.

    Keyword-only arguments. Uses ``device.compute_with_storage_grid_size()``.

    Returns: ``(MemoryConfig, CoreGrid)`` for L1 height- or block-sharded group norm.
        )doc");
}
}  // namespace

void bind_normalization_group_norm(nb::module_& mod) { bind_normalization_group_norm_operation(mod); }

}  // namespace ttnn::operations::normalization::detail
