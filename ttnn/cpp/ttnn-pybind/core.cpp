// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <reflect>

#include "ttnn/core.hpp"
#include "tt-metalium/lightmetal_binary.hpp"
#include "tt-metalium/lightmetal_replay.hpp"
#include "tt-metalium/mesh_device.hpp"

namespace ttnn::core {

void py_module_types(py::module& module) { py::class_<ttnn::Config>(module, "Config"); }

void py_module(py::module& module) {
    using tt::tt_metal::LightMetalBeginCapture;
    using tt::tt_metal::LightMetalBinary;
    using tt::tt_metal::LightMetalEndCapture;
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

    py::class_<tt::tt_metal::LightMetalReplay>(module, "LightMetalReplay")
        .def_static(
            "create",
            [](LightMetalBinary binary, distributed::MeshDevice* device = nullptr) {
                return std::make_unique<tt::tt_metal::LightMetalReplay>(std::move(binary), device);
            },
            py::arg("binary"),
            py::arg("device") = nullptr)
        .def("run", &tt::tt_metal::LightMetalReplay::run);

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

}  // namespace ttnn::core
