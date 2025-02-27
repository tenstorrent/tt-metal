// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "ttnn/core.hpp"
#include "tt-metalium/lightmetal_binary.hpp"

namespace py = pybind11;

namespace ttnn {
namespace core {

void py_module_types(py::module& module) { py::class_<ttnn::Config>(module, "Config"); }

void py_module(py::module& module) {
    auto py_config = static_cast<py::class_<ttnn::Config>>(module.attr("Config"));
    py_config.def(py::init<const ttnn::Config&>()).def("__repr__", [](const ttnn::Config& config) {
        return fmt::format("{}", config);
    });
    reflect::for_each<ttnn::Config::attributes_t>([&py_config](auto I) {
        py_config.def_property(
            std::string{reflect::member_name<I, ttnn::Config::attributes_t>()}.c_str(),
            &ttnn::Config::get<I>,
            &ttnn::Config::set<I>);
    });
    py_config.def_property_readonly("report_path", &ttnn::Config::get<"report_path">);

    py::class_<LightMetalBinary>(module, "LightMetalBinary")
        .def(py::init<>())
        .def(py::init<std::vector<uint8_t>>())
        .def("get_data", &LightMetalBinary::get_data)
        .def("set_data", &LightMetalBinary::set_data)
        .def("size", &LightMetalBinary::size)
        .def("is_empty", &LightMetalBinary::is_empty)
        .def("save_to_file", &LightMetalBinary::save_to_file)
        .def_static("load_from_file", &LightMetalBinary::load_from_file);

    module.def("get_memory_config", &ttnn::get_memory_config);
    module.def("light_metal_begin_capture", &LightMetalBeginCapture);
    module.def("light_metal_end_capture", &LightMetalEndCapture);

    module.def(
        "set_printoptions",
        &ttnn::set_printoptions,
        py::kw_only(),
        py::arg("profile"),
        R"doc(

        Set print options for tensor output.

        Keyword Args:
            profile (const std::string): the profile to use for print options.

        Returns:
            `None`: modifies print options.

        Examples:
            >>> ttnn.set_printoptions(profile="short")
        )doc");

    module.def("dump_stack_trace_on_segfault", &ttnn::core::dump_stack_trace_on_segfault);
}

}  // namespace core
}  // namespace ttnn
