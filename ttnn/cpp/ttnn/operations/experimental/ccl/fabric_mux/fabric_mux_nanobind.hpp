// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::fabric_mux {

namespace nb = nanobind;

void bind_fabric_mux(nb::module_& experimental_module);

}  // namespace ttnn::operations::experimental::fabric_mux
