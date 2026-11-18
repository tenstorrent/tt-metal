// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_nanobind.hpp"

#include <optional>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include "copy.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace {
std::string get_binary_doc_string(
    std::string_view op_name,
    std::string_view input_a,
    std::string_view input_b,
    fmt::format_string<std::string_view&, std::string_view&> op_desc) {
    return fmt::format(
        R"doc(
        {0}

        Both input tensors must be of equal shape.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{2}", "First tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "{3}", "Second tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        fmt::format(op_desc, input_a, input_b),
        op_name,
        input_a,
        input_b);
}
}  // namespace

namespace ttnn::operations::data_movement::detail {

void bind_copy(nb::module_& mod) {
    auto doc = get_binary_doc_string(
        "copy",
        "input_a",
        "input_b",
        R"doc(Copies the elements from ``{0}`` into ``{1}``. ``{1}`` is modified in place.)doc");

    ttnn::bind_function<"copy">(
        mod,
        doc.c_str(),
        nb::overload_cast<const ttnn::Tensor&, const ttnn::Tensor&>(&ttnn::copy),
        nb::arg("input_a").noconvert(),
        nb::arg("input_b").noconvert());
}

void bind_assign(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Copies `input_tensor` into a new tensor with the memory configuration `memory_config`, or into
            `output_tensor` when one is given.

            A second overload, ``ttnn.assign(input_a, input_b)``, copies `input_a` into `input_b` and returns `input_b`,
            as ``ttnn.copy`` does. `input_b` must be on the device of `input_a`, with the same shape and layout, and it
            sets the output's dtype and memory configuration.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, on device.

            Keyword Args:
                memory_config (ttnn.MemoryConfig): Memory configuration of the output.
                dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input's dtype.
                output_tensor (ttnn.Tensor, optional): Preallocated tensor for the output, on the input's device, with the input's shape and layout, the memory configuration `memory_config` and the output's dtype. Defaults to `None`, which allocates a new tensor.

            Returns:
                ttnn.Tensor: the output tensor, holding the input's values with the input's shape and layout. It is `output_tensor` when one is given.

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

                An output dtype other than the input's needs a TILE input: a ``BFLOAT8_B`` or ``BFLOAT4_B`` input can
                give a ``FLOAT32`` or ``BFLOAT16`` output, and a ``BFLOAT16`` input a ``FLOAT32`` output when its
                values are not NaN.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1
                - ND sharded: L1 and DRAM
                - The output can have a memory configuration other than the input's
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
