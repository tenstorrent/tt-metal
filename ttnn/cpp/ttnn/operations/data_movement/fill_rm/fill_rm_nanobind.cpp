// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "fill_rm.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_fill_rm_op(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Creates an ``[N, C, H, W]`` ROW_MAJOR BFLOAT16 tensor on the device of `any`. In each ``H x W`` plane, the
            top-left ``hOnes x wOnes`` block is set to `val_hi`, and every other element is set to `val_lo`.
            For example, with ``hOnes=1`` and ``wOnes=2``, the top-left corner of each plane is::

                hi hi lo ...
                lo lo lo ...
                ...

            Args:
                N (int): batch size of the output tensor. Must be greater than 0.
                C (int): number of channels of the output tensor. Must be greater than 0.
                H (int): height of the output tensor. Use a multiple of 32.
                W (int): width of the output tensor. Use a multiple of 32.
                hOnes (int): number of rows, from the top of each ``H x W`` plane, in the high-value region. At most `H`.
                wOnes (int): number of columns, from the left of each ``H x W`` plane, in the high-value region. At most `W`.
                any (ttnn.Tensor): a BFLOAT16 tensor on device. Only its device and memory configuration are used.
                val_hi (float): value for the high-value region. It is rounded to BFLOAT16.
                val_lo (float): value for every other element. It is rounded to BFLOAT16.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output tensor. It must be interleaved. Defaults to `None`, which uses the memory configuration of `any`.

            Returns:
                ttnn.Tensor: the output tensor, of shape ``[N, C, H, W]``, in ROW_MAJOR layout with BFLOAT16 dtype.

            Note:
                Supported dtypes and layouts for `any`:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16
                     - TILE, ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: not supported
        )doc";

    ttnn::bind_function<"fill_rm">(
        mod,
        doc,
        &ttnn::fill_rm,
        nb::arg("N"),
        nb::arg("C"),
        nb::arg("H"),
        nb::arg("W"),
        nb::arg("hOnes"),
        nb::arg("wOnes"),
        nb::arg("any"),
        nb::arg("val_hi"),
        nb::arg("val_lo"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

void bind_fill_ones_rm_op(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Same as ``ttnn.fill_rm``, with `val_hi` set to 1 and `val_lo` set to 0: creates an ``[N, C, H, W]`` ROW_MAJOR
            BFLOAT16 tensor whose top-left ``hOnes x wOnes`` block in each ``H x W`` plane is 1, and every other element 0.

            Args:
                N (int): batch size of the output tensor. Must be greater than 0.
                C (int): number of channels of the output tensor. Must be greater than 0.
                H (int): height of the output tensor. Use a multiple of 32.
                W (int): width of the output tensor. Use a multiple of 32.
                hOnes (int): number of rows, from the top of each ``H x W`` plane, in the high-value region. At most `H`.
                wOnes (int): number of columns, from the left of each ``H x W`` plane, in the high-value region. At most `W`.
                any (ttnn.Tensor): a BFLOAT16 tensor on device. Only its device and memory configuration are used.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output tensor. It must be interleaved. Defaults to `None`, which uses the memory configuration of `any`.

            Returns:
                ttnn.Tensor: the output tensor, of shape ``[N, C, H, W]``, in ROW_MAJOR layout with BFLOAT16 dtype.

            Note:
                Supported dtypes and layouts for `any`:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16
                     - TILE, ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: not supported
        )doc";

    ttnn::bind_function<"fill_ones_rm">(
        mod,
        doc,
        &ttnn::fill_ones_rm,
        nb::arg("N"),
        nb::arg("C"),
        nb::arg("H"),
        nb::arg("W"),
        nb::arg("hOnes"),
        nb::arg("wOnes"),
        nb::arg("any"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace

void bind_fill_rm(nb::module_& mod) {
    bind_fill_rm_op(mod);
    bind_fill_ones_rm_op(mod);
}

}  // namespace ttnn::operations::data_movement
