// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::transformer {

namespace nb = nanobind;
void bind_split_query_key_value_and_split_heads(nb::module_& mod);

}  // namespace ttnn::operations::transformer
