// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_shuffle_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/data_movement/pixel_shuffle/pixel_shuffle.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_pixel_shuffle(nb::module_& mod) {
    const auto* doc = R"doc(
        pixel_shuffle(input, upscale_factor, *, memory_config=None, output_layout=None, channel_order=PixelUnshuffleChannelOrder.CHANNEL_MAJOR, channels_last=False, padded_channels=None) -> ttnn.Tensor

        Rearranges elements in a tensor of shape ``[N, C * r^2, H, W]`` to a tensor of shape
        ``[N, C, H * r, W * r]`` where ``r = upscale_factor``: the inverse of :func:`pixel_unshuffle`,
        equivalent to ``torch.nn.functional.pixel_shuffle(input, upscale_factor)`` for the default
        channel order.

        ``channel_order`` is the packing :func:`pixel_unshuffle` used, so
        ``pixel_shuffle(pixel_unshuffle(x, r, channel_order=o), r, channel_order=o) == x``:
        ``CHANNEL_MAJOR``: ``output[n, c, h*r+rh, w*r+rw] = input[n, c*r^2 + rh*r + rw, h, w]`` (PyTorch);
        ``SPATIAL_MAJOR``: ``output[n, c, h*r+rh, w*r+rw] = input[n, rh*(r*C) + rw*C + c, h, w]`` (ONNX DepthToSpace, DCR).

        **Input layout:** ``ROW_MAJOR`` or ``TILE`` (TILE is untilized first). **Input memory:** DRAM or L1,
        interleaved or sharded. **Output layout:** ROW_MAJOR, or TILE with ``output_layout=ttnn.TILE_LAYOUT``.
        **Output memory:** ``memory_config`` (default: the input's memory config when interleaved, else DRAM).

        **channels_last:** the activation is NHWC on both sides, the layout a height-sharded conv2d produces and
        consumes: input ``[N, H, W, C * r^2]`` ROW_MAJOR HEIGHT_SHARDED in L1 (for example the row-major output of
        a 1x1 matmul), output ``[N, H * r, W * r, padded_channels]`` ROW_MAJOR HEIGHT_SHARDED in L1
        (``memory_config`` with shard ``[pixels_per_core, padded_channels]``), the ``C`` channels of every pixel
        contiguous and zero-padded up to ``padded_channels`` (default: ``C`` rounded up to the L1 alignment).
        It is the inverse of ``pixel_unshuffle(..., channels_last=True)`` and turns a 1x1 matmul
        ``C_in -> C * r^2`` into a stride-``r`` transposed conv with an ``r x r`` kernel (SPATIAL_MAJOR:
        ``W_1x1[(dy*r+dx)*C + co, ci] = W_t[ci, co, dy, dx]``).

        Args:
            input (ttnn.Tensor): 4D tensor on device, ``[N, C * r^2, H, W]`` (or ``[N, H, W, C * r^2]`` with
                ``channels_last``). The channel count must be divisible by ``r^2``.
                Supported dtypes: ``bfloat16``, ``float32``, ``uint16``, ``int32``.
            upscale_factor (int): Spatial upscale factor ``r`` (>= 1).

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory config.
            output_layout (ttnn.Layout, optional): ``ROW_MAJOR_LAYOUT`` (default) or ``TILE_LAYOUT``.
            channel_order (ttnn.PixelUnshuffleChannelOrder, optional): ``CHANNEL_MAJOR`` (default, PyTorch) or
                ``SPATIAL_MAJOR`` (ONNX DepthToSpace DCR).
            channels_last (bool, optional): NHWC in and out as described above. Default ``False``.
            padded_channels (int, optional): channels_last only: width of the output sticks (>= ``C``, a multiple
                of the L1 alignment in elements); the extra channels are zero.

        Returns:
            ttnn.Tensor: ``[N, C, H * r, W * r]`` (or ``[N, H * r, W * r, padded_channels]`` with ``channels_last``).

        Examples::

            >>> x = ttnn.from_torch(torch.randn(1, 16, 384, 384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> y = ttnn.pixel_shuffle(x, upscale_factor=4)
            >>> y.shape
            [1, 1, 1536, 1536]

            # conv_transpose2d(k=2, s=2) on a height-sharded NHWC activation: 1x1 matmul (row-major output) + pixel_shuffle
            >>> y = ttnn.matmul(x_nhwc, w_1x1, program_config=cfg_untilize_out, ...)          # [1, 1, H*W, C*4] row-major
            >>> up = ttnn.pixel_shuffle(ttnn.reshape(y, [1, H, W, C * 4]), 2, memory_config=hs_out,
            ...                         channel_order=ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR, channels_last=True)
            >>> up.shape
            [1, 2*H, 2*W, C]
    )doc";

    ttnn::bind_function<"pixel_shuffle">(
        mod,
        doc,
        &ttnn::pixel_shuffle,
        nb::arg("input"),
        nb::arg("upscale_factor"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_layout") = nb::none(),
        nb::arg("channel_order") = ttnn::PixelUnshuffleChannelOrder::CHANNEL_MAJOR,
        nb::arg("channels_last") = false,
        nb::arg("padded_channels") = nb::none());
}

}  // namespace ttnn::operations::data_movement
