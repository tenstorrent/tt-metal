// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "new_examples_pybind.hpp"

#include <pybind11/pybind11.h>
#include <boost/python.hpp>

namespace ttnn::operations::new_examples {

class NewExampleOperation {
public:
    static int Operation(int a, int b) { return a + b; }

    static std::string Operation(std::string a, std::string b) { return a + b; }
};

void py_module(py::module& module) {
    using namespace boost::python;
    // First overload - integer addition
    module.attr("operation_type") = "ttnn_lightweight";
    module.attr("python_fully_qualified_name") = "ttnn.new_examples";
    module.def("operation", static_cast<int (*)(int, int)>(&NewExampleOperation::Operation));
    module.def("operation", static_cast<std::string (*)(std::string, std::string)>(&NewExampleOperation::Operation));
}

}  // namespace ttnn::operations::new_examples
