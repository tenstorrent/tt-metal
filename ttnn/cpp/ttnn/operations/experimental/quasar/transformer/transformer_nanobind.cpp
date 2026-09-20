// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_nanobind.hpp"

#include <nanobind/nanobind.h>


namespace ttnn::operations::experimental::quasar::transformer {

void bind_transformer(nb::module_& mod) {
    auto m_transformer = mod.def_submodule("transformer", "Quasar (metal 2.0) transformer operations");

}

}  // namespace ttnn::operations::experimental::quasar::transformer
