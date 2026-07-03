// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_codegen_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "repeat_codegen.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

void bind_repeat_codegen(nb::module_& mod) {
    const auto* doc = R"doc(
        PoC codegen repeat: dual-RISC higher-dim repeat over a 4D TILE bfloat16
        interleaved tensor, wrapped in a cached DeviceOperation.

        Args:
            input_tensor (ttnn.Tensor): 4D TILE bfloat16 tensor.
            dim (int): repeat dimension (0, 1, or 2).
            repetitions (int): number of repeats along dim.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): defaults to input's.
    )doc";

    ttnn::bind_function<"repeat_codegen">(
        mod,
        doc,
        &ttnn::repeat_codegen,
        nb::arg("input_tensor"),
        nb::arg("dim"),
        nb::arg("repetitions"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
