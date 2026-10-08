// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modulated_deform_im2col_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/modulated_deform_im2col/modulated_deform_im2col.hpp"

namespace ttnn::operations::experimental::modulated_deform_im2col::detail {

void bind_modulated_deform_im2col(nb::module_& mod) {
    const auto* doc =
        R"doc(
        The column matrix of a modulated deformable convolution (DCNv2): the input
        sampled bilinearly at every offset kernel position and scaled by the mask,
        laid out so that one matmul with the (K_h*K_w*C, C_out) weight finishes the
        convolution.

        For output pixel m = (b, h_o, w_o), kernel position kk = k_h*K_w + k_w and
        channel c::

            y = h_o*stride_h - pad_h + k_h*dilation_h + offset[b, h_o, w_o, 2*kk]
            x = w_o*stride_w - pad_w + k_w*dilation_w + offset[b, h_o, w_o, 2*kk + 1]
            out[m, kk*C + c] = mask[b, h_o, w_o, kk] * bilinear(input[b, :, :, c], y, x)

        Corners outside the input contribute zero, as in mmcv and torchvision.

        Args:
            * :attr:`input`: (B, H, W, C) ROW_MAJOR bfloat16, C a multiple of 32
            * :attr:`offset`: (B, H_out, W_out, 2*K*K) ROW_MAJOR bfloat16, in pixels,
              (y, x) per kernel position as in DCNv2
            * :attr:`mask`: (B, H_out, W_out, K*K) ROW_MAJOR bfloat16, after its sigmoid
            * :attr:`kernel_size`: (K_h, K_w)
            * :attr:`stride`, :attr:`padding`, :attr:`dilation`: (h, w) pairs
            * :attr:`memory_config`: output memory config, interleaved

        Returns:
            (1, 1, B*H_out*W_out, K*K*C) TILE bfloat16, columns ordered (kk, c).
        )doc";

    ttnn::bind_function<"modulated_deform_im2col", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::modulated_deform_im2col,
        nb::arg("input"),
        nb::arg("offset"),
        nb::arg("mask"),
        nb::kw_only(),
        nb::arg("kernel_size"),
        nb::arg("stride") = std::array<uint32_t, 2>{1, 1},
        nb::arg("padding") = std::array<uint32_t, 2>{0, 0},
        nb::arg("dilation") = std::array<uint32_t, 2>{1, 1},
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::modulated_deform_im2col::detail
