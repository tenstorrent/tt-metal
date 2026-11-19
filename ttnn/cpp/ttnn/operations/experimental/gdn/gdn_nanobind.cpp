// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/gdn/gdn_decode_step/gdn_decode_step_nanobind.hpp"
#include "ttnn/operations/experimental/gdn/gdn_spec_step/gdn_spec_step_nanobind.hpp"

namespace ttnn::operations::experimental::gdn::detail {

void bind_gdn(nb::module_& mod) {
    auto gdn_module = mod.def_submodule("gdn", "Experimental gated delta net (GDN) operations");
    gdn_decode_step::detail::bind_gdn_decode_step(gdn_module);
    gdn_spec_step::detail::bind_gdn_spec_step(gdn_module);
}

}  // namespace ttnn::operations::experimental::gdn::detail
