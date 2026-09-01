// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/quasar/transformer/sdpa_decode/sdpa_decode_nanobind.hpp"

namespace ttnn::operations::experimental::quasar::transformer {

void bind_transformer(nb::module_& mod) {
    auto m_transformer = mod.def_submodule("transformer", "Quasar (metal 2.0) transformer operations");

    // sdpa_decode (paged / MLA / sharded / sliding-window / attention-sink decode variants).
    // SDPAProgramConfig / PagedCacheGeometryOverride are NOT re-registered here: they are bound
    // once on the main ttnn.transformer module (transformer_nanobind.cpp) and resolve at call time
    // within the same extension module, so re-registering would throw on duplicate type binding.
    bind_sdpa_decode(m_transformer);
}

}  // namespace ttnn::operations::experimental::quasar::transformer
