// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "untilize_codegen.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

void bind_untilize_codegen(nb::module_& mod) {
    const auto* doc = R"doc(
        PoC codegen untilize: tri-RISC (reader + pack_untilize compute + writer)
        TILE -> ROW_MAJOR untilize over a 4D bfloat16 interleaved tensor, wrapped in
        a cached DeviceOperation. Interleaved tile-row path only.

        Args:
            input_tensor (ttnn.Tensor): 4D TILE bfloat16 interleaved tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): defaults to input's.
    )doc";

    ttnn::bind_function<"untilize_codegen">(
        mod,
        doc,
        &ttnn::untilize_codegen,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
