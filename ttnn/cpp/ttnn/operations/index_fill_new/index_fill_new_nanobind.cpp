// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_new_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "index_fill_new.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::index_fill_new {

void bind_index_fill_new_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Descriptor-based port of :func:`ttnn.index_fill` (migration scaffolding for #42392).

        Identical arguments and semantics to ``ttnn.index_fill``; only the host-side program
        construction differs (``ProgramDescriptor`` instead of the legacy ``CachedProgram``
        factory). Temporary: removed once the port replaces the original operation.
    )doc";

    ttnn::bind_function<"index_fill_new">(
        mod,
        doc,
        &ttnn::index_fill_new,
        nb::arg("input"),
        nb::arg("dim"),
        nb::arg("index"),
        nb::arg("value"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::index_fill_new
