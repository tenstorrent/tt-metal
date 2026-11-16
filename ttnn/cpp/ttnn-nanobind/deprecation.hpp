// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include <nanobind/nanobind.h>

namespace ttnn {

// Emits the TTNN Python deprecation warning (a FutureWarning attributed to the user's code) by calling the Python
// helper ttnn._deprecation.warn_deprecated. Call it on the first line of a binding's body, before doing any work for
// the deprecated function or argument. See .github/instructions/ttnn-python-deprecation.instructions.md.
//
// When a warnings filter turns the warning into an error, this throws nanobind::python_error, so the binding stops
// and the FutureWarning reaches the Python caller. Message formatting and stack walking live only in the Python
// helper.
//
// Only binding code may call this; the TTNN C++ library never emits Python warnings.
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
