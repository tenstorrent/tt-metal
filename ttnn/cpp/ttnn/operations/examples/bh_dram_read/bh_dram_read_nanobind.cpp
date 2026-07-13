// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/examples/bh_dram_read/bh_dram_read.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

void bind_bh_dram_read_operation(nb::module_& mod) {
    ttnn::bind_function<"bh_dram_read">(
        mod,
        R"doc(bh_dram_read(input_tensor: ttnn.Tensor) -> None

Places one worker core per DRAM bank; each core reads the input tensor's pages
that reside in its assigned bank and discards them. Read-only; returns nothing.)doc",
        &ttnn::bh_dram_read,
        nb::arg("input_tensor"));
}

}  // namespace ttnn::operations::examples
