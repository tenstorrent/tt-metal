// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {

namespace nb = nanobind;
void bind_moreh_abs_pow_operation(nb::module_& mod);

}  // namespace ttnn::operations::moreh::moreh_abs_pow
