// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_concat_heads_decode(pybind11::module& module);

}  // namespace ttnn::operations::experimental::transformer::detail
