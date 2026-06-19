// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::quasar {

namespace nb = nanobind;

// Creates the `quasar` submodule under ttnn.experimental and binds the Quasar (metal 2.0) ops.
void bind_quasar(nb::module_& mod);

}  // namespace ttnn::operations::experimental::quasar
