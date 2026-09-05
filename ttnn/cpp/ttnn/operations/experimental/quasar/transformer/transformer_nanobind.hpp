// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::quasar::transformer {

namespace nb = nanobind;

// Creates the `transformer` submodule under ttnn.experimental.quasar and binds the Quasar
// (metal 2.0) transformer ops. Mirrors ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp
// so future quasar transformer ports (sdpa prefill, ...) have a home here.
void bind_transformer(nb::module_& mod);

}  // namespace ttnn::operations::experimental::quasar::transformer
