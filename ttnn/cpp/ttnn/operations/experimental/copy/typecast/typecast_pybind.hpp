// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::copy::detail  {

void py_bind_typecast(pybind11::module& m);

}  // namespace ttnn::operations::experimental::copy::detail
