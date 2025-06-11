// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::cnn::detail {

void bind_convert_to_chw(pybind11::module& module);

}  // namespace ttnn::operations::experimental::cnn::detail
