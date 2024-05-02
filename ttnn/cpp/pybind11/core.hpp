// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/core.hpp"

namespace py = pybind11;

namespace ttnn {
namespace core {
void py_module(py::module& module) {
    auto py_config = py::class_<ttnn::Config>(module, "Config")
                         .def_readwrite("cache_path", &ttnn::Config::cache_path)
                         .def_readwrite("model_cache_path", &ttnn::Config::model_cache_path)
                         .def_readwrite("tmp_dir", &ttnn::Config::tmp_dir)
                         .def_readwrite("enable_model_cache", &ttnn::Config::enable_model_cache)
                         .def_readwrite("enable_fast_runtime_mode", &ttnn::Config::enable_fast_runtime_mode)
                         .def_readwrite("throw_exception_on_fallback", &ttnn::Config::throw_exception_on_fallback)
                         .def_readwrite("enable_logging", &ttnn::Config::enable_logging)
                         .def_readwrite("enable_graph_report", &ttnn::Config::enable_graph_report)
                         .def_readwrite("enable_detailed_buffer_report", &ttnn::Config::enable_detailed_buffer_report)
                         .def_readwrite("enable_detailed_tensor_report", &ttnn::Config::enable_detailed_tensor_report)
                         .def_readwrite("enable_comparison_mode", &ttnn::Config::enable_comparison_mode)
                         .def_readwrite("comparison_mode_pcc", &ttnn::Config::comparison_mode_pcc)
                         .def_readwrite("root_report_path", &ttnn::Config::root_report_path)
                         .def_readwrite("report_name", &ttnn::Config::report_name)
                         .def("__repr__", [](const ttnn::Config& config) { return fmt::format("{}", config); });

    module.def("get_memory_config", &ttnn::get_memory_config);
    module.def("set_printoptions", &ttnn::set_printoptions, py::kw_only(), py::arg("profile"));
}

}  // namespace core
}  // namespace ttnn
