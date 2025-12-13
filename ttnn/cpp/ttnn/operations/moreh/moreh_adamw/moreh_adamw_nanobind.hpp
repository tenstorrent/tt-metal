// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

namespace nb = nanobind;

void bind_moreh_adamw_operation(nb::module_& mod);

void py_module(nb::module_& mod);

}  // namespace ttnn::operations::moreh::moreh_adamw
