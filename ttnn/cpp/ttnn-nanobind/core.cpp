// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <reflect>

#include "tt-metalium/lightmetal_binary.hpp"
#include "tt-metalium/lightmetal_replay.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "ttnn-nanobind/nanobind_helpers.hpp"
#include "ttnn/config.hpp"
#include "ttnn/core.hpp"
#include "ttnn/common/guard.hpp"

namespace ttnn::core {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::Config>(mod, "Config");
}

void py_module(nb::module_& mod) {
    using tt::tt_metal::LightMetalBeginCapture;
    using tt::tt_metal::LightMetalBinary;
    using tt::tt_metal::LightMetalEndCapture;
    auto py_config = static_cast<nb::class_<ttnn::Config>>(mod.attr("Config"));
    py_config
        .def(nb::init<const ttnn::Config&>())
        .def("__repr__", [](const ttnn::Config& config) {
            return fmt::format("{}", config);
        });
    reflect::for_each<ttnn::Config::attributes_t>([&py_config](auto I) {
       py_config.def_prop_rw(
            std::string{reflect::member_name<I, ttnn::Config::attributes_t>()}.c_str(),
            &ttnn::Config::get<I>,
            &ttnn::Config::set<I>);
    });
    py_config.def_prop_ro("report_path", &ttnn::Config::get<"report_path">);

    nb::class_<LightMetalBinary>(mod, "LightMetalBinary")
        .def(nb::init<>())
        .def(nb::init<std::vector<uint8_t>>())
        .def("get_data", &LightMetalBinary::get_data)
        .def("set_data", &LightMetalBinary::set_data)
        .def("size", &LightMetalBinary::size)
        .def("is_empty", &LightMetalBinary::is_empty)
        .def("save_to_file", &LightMetalBinary::save_to_file)
        .def_static("load_from_file", &LightMetalBinary::load_from_file);

    nb::class_<tt::tt_metal::LightMetalReplay>(mod, "LightMetalReplay")
        .def_static(
            "create",
            [](LightMetalBinary binary, distributed::MeshDevice* device = nullptr) {
                return nbh::make_unique<tt::tt_metal::LightMetalReplay>(std::move(binary), device);
            },
            nb::arg("binary"),
            nb::arg("device") = nullptr)
        .def("run", &tt::tt_metal::LightMetalReplay::run);

    mod.def("get_memory_config", &ttnn::get_memory_config);
    mod.def("light_metal_begin_capture", &LightMetalBeginCapture);
    mod.def("light_metal_end_capture", &LightMetalEndCapture);

    mod.def(
        "set_printoptions",
        &ttnn::set_printoptions,
        nb::kw_only(),
        nb::arg("profile"),
        R"doc(

        Set print options for tensor output.

        Keyword Args:
            profile (const std::string): the profile to use for print options.

        Returns:
            `None`: modifies print options.

        Examples:
            >>> ttnn.set_printoptions(profile="short")
        )doc");

    mod.def("dump_stack_trace_on_segfault", &ttnn::core::dump_stack_trace_on_segfault);

    mod.def("get_current_command_queue_id_for_thread", &ttnn::core::get_current_command_queue_id_for_thread);
    mod.def("push_current_command_queue_id_for_thread", &ttnn::core::push_current_command_queue_id_for_thread);
    mod.def("pop_current_command_queue_id_for_thread", &ttnn::core::pop_current_command_queue_id_for_thread);
}

}  // namespace ttnn::core
