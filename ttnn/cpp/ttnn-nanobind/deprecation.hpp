// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include <nanobind/nanobind.h>

namespace ttnn {

// See .github/instructions/ttnn-python-deprecation.instructions.md ("Shared helper").
// An empty earliest_removal_date means no date. Otherwise it is a YYYY-MM-DD date.
inline void warn_deprecated(
    std::string_view what, std::string_view instead, std::string_view earliest_removal_date = {}) {
    namespace nb = nanobind;
    // Bindings registered through ttnn::bind_function run with the GIL released.
    nb::gil_scoped_acquire gil;
    nb::object earliest_removal_date_arg =
        earliest_removal_date.empty() ? nb::object(nb::none())
                                      : nb::object(nb::str(earliest_removal_date.data(), earliest_removal_date.size()));
    // Imported on every call (a sys.modules lookup) and never cached in a static, whose destructor would run after
    // the interpreter has shut down.
    nb::module_::import_("ttnn._deprecation")
        .attr("warn_deprecated")(
            nb::str(what.data(), what.size()),
            nb::str(instead.data(), instead.size()),
            nb::arg("earliest_removal_date") = earliest_removal_date_arg);
}

}  // namespace ttnn
