// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/small_vector_caster.hpp"  // NOLINT - for nanobind SmallVector binding support.

#include "tilize_with_val_padding.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_tilize_with_val_padding(nb::module_& mod) {
    // The BFLOAT16 NaN rule in this docstring tracks tenstorrent/tt-metal#31406, and the FLOAT32 and INT32 pad_value
    // rules track tenstorrent/tt-metal#44604 (an int32_t PadValue); drop each once its issue is fixed.
    const auto* doc =
        R"doc(
            Changes the layout of the input tensor from ROW_MAJOR to TILE, padding it to `output_tensor_shape` with
            `pad_value`.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, on device, in ROW_MAJOR layout.
                output_tensor_shape (ttnn.Shape or List[int]): the padded shape of the output. Its last two dimensions are multiples of 32. For an input of rank 2 or more, it has the input's rank, its dimensions before the last two equal the input's, and its last two are at least the input's. For an input of rank 0 or 1, it has two dimensions, and its last is at least the input's size. For an input of rank 5 or more, the last two dimensions must be the input's rounded up to multiples of 32. When the last dimension is more than 1024, the second-to-last must be the input's rounded up to a multiple of 32.
                pad_value (float or int): value of the padding. For ``FLOAT32`` tensors it must be a float, such as ``1.0`` rather than ``1``. ``BFLOAT16`` rounds it to the nearest ``BFLOAT16`` value. For ``INT32``, ``UINT32`` and ``UINT16`` tensors it must be an int that the dtype can hold, and a negative ``INT32`` value must be at least ``-2**24``.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the input tensor's memory configuration. A sharded output, including the default one for a sharded input, needs a shard shape that is a multiple of 32x32. When the input and the output are WIDTH_SHARDED in L1, only the second-to-last dimension is padded and ``sub_core_grids`` is `None`, the output keeps the input's core grid.
                dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input dtype, or gives ``FLOAT32`` for a ``FP8_E4M3`` input. A ``BFLOAT16`` or ``FLOAT32`` input can give a ``BFLOAT16``, ``FLOAT32``, ``BFLOAT8_B`` or ``BFLOAT4_B`` output. For an integer input it must be `None` or the input's dtype.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which lets the operation choose its cores.

            Returns:
                ttnn.Tensor: the output tensor, in TILE layout with 32x32 tiles. Its shape is the input's, and its padded shape is `output_tensor_shape`. For an input with no elements, `output_tensor_shape` must have no elements either, and it is the output's shape.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                     - ROW_MAJOR

                ``BFLOAT16`` input values must not be NaN.

                On Blackhole, a ``FP8_E4M3`` ROW_MAJOR input is also supported. Its output defaults to ``FLOAT32``, and a
                ``dtype`` must be a floating-point type other than ``FP8_E4M3``. The last dimension of `output_tensor_shape`
                must be more than 32, and `pad_value` must be a float, which is first converted to ``FP8_E4M3``.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
                - Sharded output: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
                - A WIDTH_SHARDED or BLOCK_SHARDED input needs a shard width, in bytes, that is a multiple of the buffer's alignment
                - ND sharded: L1 input and output
        )doc";

    ttnn::bind_function<"tilize_with_val_padding">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            const ttnn::Shape&,
            ttnn::PadValue,
            const std::optional<MemoryConfig>&,
            std::optional<DataType>,
            bool,
            const std::optional<CoreRangeSet>&>(&ttnn::tilize_with_val_padding),
        nb::arg("input_tensor"),
        nb::arg("output_tensor_shape"),
        nb::arg("pad_value"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("sub_core_grids") = nb::none());
}

void bind_tilize_with_zero_padding(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Changes data layout of input tensor to TILE. Pads to the nearest multiple of TILE width/height with zero value.

            Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 or UINT32 data type.

            Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 or UINT32 data type.

            Args:
                * :attr:`input_tensor`: Input Tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor.
                * :attr:`dtype`: Data type of the output tensor.
                * :attr:`use_multicore`: Whether to use multicore.
        )doc";

    ttnn::bind_function<"tilize_with_zero_padding">(
        mod,
        doc,
        &ttnn::tilize_with_zero_padding,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
