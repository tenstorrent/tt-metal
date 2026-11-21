// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "reshape.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"  // for ttsl::SmallVector<int32_t>

namespace ttnn::operations::data_movement {

namespace {

// Python binding overload: takes W, Z, Y, X as separate parameters
ttnn::Tensor reshape_on_device_wrapper(
    const ttnn::Tensor& input_tensor,
    int W,
    int Z,
    int Y,
    int X,
    const std::optional<MemoryConfig>& memory_config_arg) {
    return reshape_on_device(input_tensor, ttsl::SmallVector<int32_t>{W, Z, Y, X}, memory_config_arg);
}

}  // namespace

void bind_reshape(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Reshapes `input_tensor` to ``[W, Z, Y, X]``, keeping its elements in the same row-major order.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, of rank 4, on device. A TILE input must use 32x32 tiles.
                W (int): the first dimension of the output.
                Z (int): the second dimension of the output.
                Y (int): the third dimension of the output.
                X (int): the last dimension of the output.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. It must be interleaved. When `X` equals the last dimension of the input, it must be `None` or the input's memory configuration. Defaults to `None`, which uses the input's memory configuration.

            Returns:
                ttnn.Tensor: the output tensor, of shape ``[W, Z, Y, X]``, with the input's dtype and layout. When `X` equals the last dimension of the input, the output is a view that shares the input's device buffer.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32
                     - TILE, ROW_MAJOR

                ``W * Z * Y * X`` must equal the number of elements of the input. One of `W`, `Z`, `Y` and `X` can be
                -1: it is then inferred from that number and the other three.

                For a TILE input, the last two dimensions of the input and of the output must be multiples of 32.

                For a ROW_MAJOR input whose last dimension differs from `X`:

                - When both are multiples of 8, the larger must be a multiple of the smaller. An input in DRAM then
                  needs a last dimension that is a multiple of 32 for ``BFLOAT16``, or of 16 for ``FLOAT32``.
                - Otherwise, the input must be ``BFLOAT16``.

            Memory Support:
                - Interleaved: DRAM and L1
        )doc";

    ttnn::bind_function<"reshape_on_device">(
        mod,
        doc,
        nb::overload_cast<const ttnn::Tensor&, int, int, int, int, const std::optional<ttnn::MemoryConfig>&>(
            &reshape_on_device_wrapper),
        nb::arg("input_tensor"),
        nb::arg("W"),
        nb::arg("Z"),
        nb::arg("Y"),
        nb::arg("X"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
