// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::distributed {

class PyISocket;

void py_isocket_module_types(pybind11::module& m);

}  // namespace ttnn::distributed
