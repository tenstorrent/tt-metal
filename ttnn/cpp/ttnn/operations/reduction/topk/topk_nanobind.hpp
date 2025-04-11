// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;
void bind_reduction_topk_operation(nb::module_& mod);
}
