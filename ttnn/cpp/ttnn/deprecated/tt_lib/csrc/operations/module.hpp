// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "primary/module.hpp"
#include "ttnn/operation_history.hpp"
#include "tt_numpy/functions.hpp"

namespace py = pybind11;


namespace tt {
namespace operations {

void py_module(py::module& m_operations) {
    auto m_primary = m_operations.def_submodule("primary", "Primary operations");
    primary::py_module(m_primary);

#ifdef DEBUG
    m_operations.def(
        "dump_operation_history_to_csv",
        &tt::tt_metal::operation_history::dump_to_csv,
        "Dump Operation History to a CSV File");
    m_operations.def(
        "dump_operation_history_to_json",
        &tt::tt_metal::operation_history::dump_to_json,
        "Dump Operation History to a JSON File");
    m_operations.def("clear_operation_history", &tt::tt_metal::operation_history::clear, "Clear Operation History");
#endif
}

}  // namespace operations

}  // namespace tt
