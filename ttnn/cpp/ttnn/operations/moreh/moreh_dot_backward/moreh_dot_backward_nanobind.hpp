// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {

namespace nb = nanobind;
void bind_moreh_dot_backward_operation(nb::module_& mod);

}  // namespace ttnn::operations::moreh::moreh_dot_backward
