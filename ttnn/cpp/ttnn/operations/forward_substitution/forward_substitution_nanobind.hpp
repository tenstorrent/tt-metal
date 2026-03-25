// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::forward_substitution {

namespace nb = nanobind;
void bind_forward_substitution_operation(nb::module_& mod);
}  // namespace ttnn::operations::forward_substitution
