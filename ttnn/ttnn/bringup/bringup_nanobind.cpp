// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Python bindings of the derived bring-up ops (INDEX.md): ttnn._ttnn.operations.bringup, re-exported as
// ttnn.bringup. fork_op.py adds one declaration and one call per fork between the markers.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// BEGIN FORKED OPS (fork_op.py)
// END FORKED OPS (fork_op.py)

namespace ttnn::bringup {

void py_module([[maybe_unused]] nb::module_& mod) {  // unused until the first fork registers
    // BEGIN FORKED OPS (fork_op.py)
    // END FORKED OPS (fork_op.py)
}

}  // namespace ttnn::bringup
