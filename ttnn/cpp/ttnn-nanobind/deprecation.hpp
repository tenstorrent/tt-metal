// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include <nanobind/nanobind.h>

namespace ttnn {

// See .github/instructions/ttnn-python-deprecation.instructions.md ("Shared helper").
inline void warn_deprecated(std::string_view what, std::string_view instead) {
    namespace nb = nanobind;
    // Bindings registered through ttnn::bind_function run with the GIL released.
    nb::gil_scoped_acquire gil;
    // Imported on every call (a sys.modules lookup) and never cached in a static, whose destructor would run after
    // the interpreter has shut down.
    nb::module_::import_("ttnn._deprecation")
        .attr("warn_deprecated")(nb::str(what.data(), what.size()), nb::str(instead.data(), instead.size()));
}

}  // namespace ttnn
