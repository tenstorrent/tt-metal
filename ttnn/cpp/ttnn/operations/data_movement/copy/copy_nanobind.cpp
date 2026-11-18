// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "copy.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_copy(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Copies the values of `input_a` into `input_b` and returns `input_b`. `input_b` is modified in place.

            Args:
                input_a (ttnn.Tensor): the tensor to copy from, on device.
                input_b (ttnn.Tensor): the tensor to copy into, on the device of `input_a`, with the same shape and layout. It keeps its dtype and memory configuration.

            Returns:
                ttnn.Tensor: `input_b`, holding the values of `input_a`.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                     - TILE, ROW_MAJOR
                   * - BFLOAT8_B, BFLOAT4_B
                     - TILE

                TILE tensors must have 32x32 tiles, with the last two dimensions padded only up to the next multiple
                of 32.

                `input_b` can have a dtype other than that of `input_a` when both are in TILE layout: a ``BFLOAT8_B`` or
                ``BFLOAT4_B`` `input_a` can be copied into a ``FLOAT32`` or ``BFLOAT16`` `input_b`, and a ``BFLOAT16``
                `input_a` into a ``FLOAT32`` `input_b` when its values are not NaN.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1
                - ND sharded: L1 and DRAM
                - `input_a` and `input_b` can have different memory configurations
        )doc";

    ttnn::bind_function<"copy">(
        mod,
        doc,
        nb::overload_cast<const ttnn::Tensor&, const ttnn::Tensor&>(&ttnn::copy),
        nb::arg("input_a").noconvert(),
        nb::arg("input_b").noconvert());
}

void bind_assign(nb::module_& mod) {
    const auto* doc =
        R"doc(
    Returns a new tensor which is a new copy of input tensor. Alternatively, copies input tensor ``input_a`` to ``input_b`` if their shapes and memory layouts match, and returns input_b tensor.

    Input tensor must have BFLOAT16 data type.
    Output tensor will have BFLOAT16 data type.

    Args:
        input_tensor (ttnn.Tensor): Tensor assign is applied to. Tensor of shape [W, Z, Y, X]. Used by the overload that takes ``memory_config`` (keyword-only, required), ``dtype`` and ``output_tensor``.
        input_a (ttnn.Tensor): Tensor to copy from. Tensor of shape [W, Z, Y, X]. Used by the overload that copies ``input_a`` to ``input_b``.
        input_b (ttnn.Tensor): Tensor to copy to. Tensor of shape [W, Z, Y, X]. Only if performing assign from ``input_a``.

    )doc";

    ttnn::bind_function<"assign">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::MemoryConfig&,
                std::optional<const ttnn::DataType>,
                const std::optional<ttnn::Tensor>&>(&ttnn::assign),
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config"),
            nb::arg("dtype") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttnn::Tensor&>(&ttnn::assign),
            nb::arg("input_a").noconvert(),
            nb::arg("input_b").noconvert()));
}

}  // namespace ttnn::operations::data_movement::detail
