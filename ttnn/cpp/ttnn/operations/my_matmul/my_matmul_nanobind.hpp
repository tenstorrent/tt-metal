// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::my_matmul {
namespace nb = nanobind;
void bind_my_matmul_operation(nb::module_& mod);
}  // namespace ttnn::operations::my_matmul
