// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_softmax(pybind11::module& module);

}  // namespace ttnn::operations::normalization::detail
