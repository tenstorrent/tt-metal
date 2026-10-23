// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "to_memory_config_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_to_memory_config(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a tensor to the specified memory configuration (sharded/interleaved layout conversion),
        dispatching to the quasar reshard / interleaved_to_sharded / sharded_to_interleaved backends.

        Args:
            * :attr:`tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (MemoryConfig): target memory config
            * :attr:`dtype` (ttnn.DataType, optional): output dtype
            * :attr:`output_tensor` (ttnn.Tensor, optional): preallocated output tensor
        )doc";

    ttnn::bind_function<"to_memory_config", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::to_memory_config,
        nb::arg("tensor").noconvert(),
        nb::arg("memory_config"),
        nb::arg("dtype") = nb::none(),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
