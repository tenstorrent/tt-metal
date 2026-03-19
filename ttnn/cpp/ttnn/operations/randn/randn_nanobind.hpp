// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::randn {
namespace nb = nanobind;
void bind_randn_operation(nb::module_& mod);
}  // namespace ttnn::operations::randn
