// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_unshuffle_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_pixel_unshuffle(nb::module_& mod) {
    const auto* doc = R"doc(
        pixel_unshuffle(input, downscale_factor, *, memory_config=None, output_layout=None) -> ttnn.Tensor

        Rearranges elements in a tensor of shape ``[N, C, H, W]`` to a tensor of
        shape ``[N, C * r^2, H / r, W / r]`` where ``r = downscale_factor``.

        Reverses the effect of :func:`pixel_shuffle`. Equivalent to
        ``torch.nn.functional.pixel_unshuffle(input, downscale_factor)``.

        Channel ordering matches PyTorch:
        ``output[n, c*r^2 + rh*r + rw, h', w'] = input[n, c, h'*r+rh, w'*r+rw]``.

        **Input layout:** Both ``ROW_MAJOR`` and ``TILE`` are accepted. TILE tensors are
        automatically untilized to ROW_MAJOR before the kernel runs.

        **Input memory:** DRAM or L1, interleaved or sharded.  Sharded input tensors
        are converted to DRAM-interleaved internally.

        **Output memory:** Controlled by ``memory_config``.  Supports DRAM, L1-interleaved,
        and sharded L1 (HEIGHT_SHARDED / BLOCK_SHARDED).

        **Output layout:** ROW_MAJOR by default.  Pass ``output_layout=ttnn.TILE_LAYOUT``
        to receive a TILE tensor directly.

        Args:
            input (ttnn.Tensor): 4D tensor ``[N, C, H, W]`` on device.
                H and W must be divisible by ``downscale_factor``.
                Supported dtypes: ``bfloat16``, ``float32``, ``uint16``, ``int32``.
            downscale_factor (int): Spatial reduction factor ``r`` (must be ≥ 1).

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory config.
                Defaults to the input tensor's memory config (after sharded → interleaved
                normalisation).  Pass a sharded MemoryConfig for sharded L1 output.
            output_layout (ttnn.Layout, optional): Output tensor layout.
                ``None`` or ``ROW_MAJOR_LAYOUT`` (default) → ROW_MAJOR output.
                ``TILE_LAYOUT`` → TILE output (kernel output is tilized before returning).

        Returns:
            ttnn.Tensor: Output tensor ``[N, C * r^2, H / r, W / r]``.

        Examples::

            # Basic usage — ROW_MAJOR DRAM output
            >>> x = ttnn.from_torch(torch.randn(1, 1, 1536, 1536), dtype=ttnn.bfloat16,
            ...                     layout=ttnn.TILE_LAYOUT, device=device)
            >>> y = ttnn.pixel_unshuffle(x, downscale_factor=4)
            >>> y.shape
            [1, 16, 384, 384]

            # TILE output
            >>> y_tile = ttnn.pixel_unshuffle(x, 4, output_layout=ttnn.TILE_LAYOUT)
            >>> y_tile.layout
            Layout.TILE

            # Sharded L1 output
            >>> shard_spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            >>> sharded_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
            >>> y_sharded = ttnn.pixel_unshuffle(x, 4, memory_config=sharded_cfg)
    )doc";

    ttnn::bind_function<"pixel_unshuffle">(
        mod,
        doc,
        &ttnn::pixel_unshuffle,
        nb::arg("input"),
        nb::arg("downscale_factor"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_layout") = nb::none());
}

}  // namespace ttnn::operations::data_movement
