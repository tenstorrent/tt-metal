// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::creation {

namespace nb = nanobind;
void bind_creation_operations(nb::module_& mod);

}  // namespace ttnn::operations::creation
