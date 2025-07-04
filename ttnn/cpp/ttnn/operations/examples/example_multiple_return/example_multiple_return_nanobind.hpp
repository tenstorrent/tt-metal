// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::examples {

namespace nb = nanobind;
void bind_example_multiple_return_operation(nb::module_& mod);

}  // namespace ttnn::operations::examples
