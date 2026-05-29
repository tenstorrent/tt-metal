// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_repro_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/examples/memory_repro/memory_repro.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

void bind_memory_repro_operation(nb::module_& mod) {
    ttnn::bind_function<"memory_repro">(
        mod,
        R"doc(memory_repro(input_tensor: ttnn.Tensor) -> ttnn.Tensor

Minimal repro op for per-core CB allocation. Launches a blank dataflow kernel on
cores (0,0) and (1,0) with CBs of different sizes (1.0 MiB and 750 KiB).)doc",
        &ttnn::memory_repro,
        nb::arg("input_tensor"));
}

}  // namespace ttnn::operations::examples
