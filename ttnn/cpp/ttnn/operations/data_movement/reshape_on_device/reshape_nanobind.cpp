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

        Returns a tensor with the new shape ``[W, Z, Y, X]``.

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.arange(4.)
            W = 1
            Z = 1
            Y = 2
            X = 2
            output_tensor = torch.reshape(input_tensor, (W, Z, Y, X))

        Whether data moves depends on ``X``:

        * If ``X`` equals the input's padded last dim (for TILE, its last dim rounded up to a multiple of 32), the
          result is a zero-cost view that shares the input's buffer and keeps its memory configuration;
          ``memory_config`` is ignored.
        * Otherwise the data is copied into a new tensor on device.
        * A ROW_MAJOR copy in which the input's or the new last dim is not a multiple of 8 is instead done on host and
          written back to device. It needs a ``BFLOAT16`` input.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, of rank 4. A TILE layout tensor must use the standard 32x32 tile.
            W (int): the new dim 0. One of ``W``, ``Z``, ``Y`` and ``X`` can be ``-1``, in which case it is inferred from the volume.
            Z (int): the new dim 1.
            Y (int): the new dim 2.
            X (int): the new dim 3.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor when data is copied. Defaults to `None`, which uses the input tensor's memory configuration.

        Returns:
            ttnn.Tensor: the rank 4 output tensor, with the input's dtype and layout. It is a view of the input when ``X`` matches the input's padded last dim, and a new tensor otherwise.

        Note:
            Supported dtypes and layouts when data is copied:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

            A view, where ``X`` matches the input's padded last dim, does not move data and accepts any dtype.

            A TILE copy needs the input and output to have the same tile-padded volume; reshapes that change the
            tile padding, such as ``(1, 1, 1, 1024)`` to ``(1, 1, 32, 32)``, are not supported. Use :func:`ttnn.reshape`
            for those.

            A ROW_MAJOR copy on device needs the input's and the new last dim to be multiples of 8, with one of the two
            row sizes, in bytes, a multiple of the other.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: not supported when data is copied; use :func:`ttnn.reshape` for sharded tensors

        Example:

            >>> tensor = ttnn.from_torch(torch.rand((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.reshape_on_device(tensor, 1, 1, 64, 32)
            >>> print(output.shape)
            ttnn.Shape([1, 1, 64, 32])

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
