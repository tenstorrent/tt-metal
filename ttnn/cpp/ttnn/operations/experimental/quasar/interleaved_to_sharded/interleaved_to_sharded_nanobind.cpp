// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "interleaved_to_sharded.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_interleaved_to_sharded(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts an interleaved tensor to a sharded tensor (Quasar / Metal 2.0).

        Args:
            input_tensor (ttnn.Tensor): the interleaved input tensor.
            sharded_memory_config (ttnn.MemoryConfig): sharded memory configuration for the output.

        Keyword Args:
            output_dtype (ttnn.DataType, optional): data type of the output tensor. Defaults to the input dtype.
            keep_l1_aligned (bool, optional): keep the shard L1-aligned. Defaults to `False`.
            preallocated_output (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the sharded output tensor.
        )doc";

    ttnn::bind_function<"interleaved_to_sharded", "ttnn.experimental.quasar.">(
        mod,
        doc,
        // Overload 1: explicit grid / shard_shape / scheme / orientation.
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::variant<CoreCoord, CoreRangeSet>&,
                std::array<uint32_t, 2>,
                TensorMemoryLayout,
                tt::tt_metal::ShardOrientation,
                const std::optional<DataType>&,
                const std::optional<bool>&>(&ttnn::operations::experimental::quasar::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false),
        // Overload 2: sharded MemoryConfig -- mirrors ttnn.interleaved_to_sharded(x, sharded_memory_config).
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const MemoryConfig&,
                const std::optional<DataType>&,
                const std::optional<bool>&,
                const std::optional<ttnn::Tensor>&>(&ttnn::operations::experimental::quasar::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("sharded_memory_config"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false,
            nb::arg("preallocated_output") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
