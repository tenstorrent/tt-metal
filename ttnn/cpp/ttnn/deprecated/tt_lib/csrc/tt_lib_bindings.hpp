// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "tt_metal/host_api.hpp"

namespace py = pybind11;

namespace tt {
void bind_deprecated(py::module m);

}
