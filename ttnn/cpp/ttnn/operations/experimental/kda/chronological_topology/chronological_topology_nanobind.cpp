// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_topology_nanobind.hpp"
#include "chronological_topology.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::experimental::kda::chronological_topology::detail {
void bind_chronological_topology(nb::module_& mod) {
    ttnn::bind_function<"chronological_topology", "ttnn.experimental.kda.">(
        mod,
        "Derive fixed-size KDA control and selection records from device start metadata.",
        &ttnn::experimental::kda::chronological_topology,
        nb::arg("start").noconvert(),
        nb::arg("rank").noconvert(),
        nb::arg("sp_size"),
        nb::arg("local_rows"),
        nb::arg("batch_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"));
}
}  // namespace ttnn::operations::experimental::kda::chronological_topology::detail
