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
    const auto* doc =
        R"doc(
            Changes the layout of the input tensor from ROW_MAJOR to TILE, padding it to `output_tensor_shape` with `pad_value`.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, on device.
                output_tensor_shape (ttnn.Shape or List[int]): the padded shape of the output. It must have the input's rank and be at least the input's size in every dimension, and its last two dimensions must be multiples of 32. The output keeps the input's logical shape.
                pad_value (float or int): value to pad with. For ``FLOAT32`` tensors, pass a float: an int is currently written as raw bits rather than converted.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`, which uses the input tensor's memory configuration.
                dtype (ttnn.DataType, optional): Data type of the output tensor. Defaults to `None`, which keeps the input dtype.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which uses the whole device.

            Returns:
                ttnn.Tensor: the output tensor, in TILE layout with the standard 32x32 tile.

            Note:
                Supported dtypes and layouts for the input tensor:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                     - ROW_MAJOR

                ``dtype`` can convert the output, for example from ``BFLOAT16`` input to ``BFLOAT8_B`` output. A TILE input
                is returned unchanged.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input in L1 or DRAM; the output can be interleaved or sharded
                - ND sharded: input and output
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
