// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount::detail {

namespace nb = nanobind;

void bind_experimental_masked_bincount_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount::detail
