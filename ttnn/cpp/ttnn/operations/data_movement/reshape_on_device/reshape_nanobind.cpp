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

        Returns a tensor with the new shape ``[W, Z, Y, X]``. For a TILE input with tile padding, padding values can
        become output values, on the view path as well as the copy path, so its logical elements are not always kept
        in row-major order. Use :func:`ttnn.reshape` to reshape the logical elements of a padded TILE tensor.

        Whether data moves depends on ``X``:

        * If ``X`` equals the input's padded last dim (for TILE, its last dim rounded up to a multiple of 32), the
          result is a zero-cost view, provided the padded output fits in the input's buffer; otherwise an error is
          raised, as for TILE ``[1, 1, 32, 64]`` to ``[1, 2, 16, 64]``, whose padded output is twice as large. The
          view shares the input's buffer and keeps its memory layout, buffer type and
          shard grid, with shard metadata adapted to the new shape as in :func:`ttnn.view`; ``memory_config`` is
          ignored. For an ND sharded input the view must also keep the padded shape; other
          reshapes of an ND sharded input raise an error, so use :func:`ttnn.reshape` for those.
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

            A TILE copy reshapes the tile-padded data, not just the logical elements, and needs the input and output
            to have the same tile-padded volume. When the input has tile padding, padding values can therefore end
            up as logical output values: ``(1, 1, 32, 48)`` to ``(1, 1, 48, 32)`` runs, but the output does not
            hold the input's logical elements in row-major order. Reshapes that change the tile-padded volume, such as ``(1, 1, 1, 1024)`` to
            ``(1, 1, 32, 32)``, are not supported. Use :func:`ttnn.reshape` to reshape logical data with padding.

            A ROW_MAJOR copy on device needs the input's and the new last dim to be multiples of 8, with one of the two
            row sizes, in bytes, a multiple of the other.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: supported for a view. A copy on device needs interleaved input and output; the ``BFLOAT16`` ROW_MAJOR copy done on host can read a sharded input when ``memory_config`` is interleaved. Use :func:`ttnn.reshape` for other sharded copies
            - ND sharded: only as a view that keeps the padded shape

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
