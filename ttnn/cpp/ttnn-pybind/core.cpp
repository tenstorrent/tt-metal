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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <reflect>

#include "ttnn/core.hpp"
#include "ttnn/common/guard.hpp"
#include "tt-metalium/lightmetal_binary.hpp"
#include "tt-metalium/lightmetal_replay.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_stl/caseless_comparison.hpp"

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
        [](const std::string& profile, const py::object& sci_mode, const py::object& precision) {
            ttnn::TensorPrintProfile profile_enum =
                enchantum::cast<ttnn::TensorPrintProfile>(profile, ttsl::ascii_caseless_comp).value();

            ttnn::SciMode sci_mode_enum = ttnn::SciMode::Default;
            if (!sci_mode.is_none()) {
                if (py::isinstance<py::bool_>(sci_mode)) {
                    sci_mode_enum = sci_mode.cast<bool>() ? ttnn::SciMode::Enable : ttnn::SciMode::Disable;
                } else if (py::isinstance<py::str>(sci_mode)) {
                    auto cmp = [](const auto& a, const auto& b) -> bool {
                        return ttsl::ascii_caseless_comp(std::string_view(a), std::string_view(b));
                    };
                    const std::string sci_mode_str = sci_mode.cast<std::string>();
                    if (cmp(sci_mode_str, "true")) {
                        sci_mode_enum = ttnn::SciMode::Enable;
                    } else if (cmp(sci_mode_str, "false")) {
                        sci_mode_enum = ttnn::SciMode::Disable;
                    } else if (cmp(sci_mode_str, "none") || cmp(sci_mode_str, "default")) {
                        sci_mode_enum = ttnn::SciMode::Default;
                    } else {
                        throw std::invalid_argument("sci_mode must be None, bool, or str (true, false, default)");
                    }
                } else {
                    throw std::invalid_argument("sci_mode must be None, bool, or str (true, false, default)");
                }
            }

            int precision_value = 4;
            if (!precision.is_none()) {
                if (py::isinstance<py::int_>(precision)) {
                    precision_value = precision.cast<int>();
                } else {
                    throw std::invalid_argument("precision must be None or int");
                }
            }

            ttnn::set_printoptions(profile_enum, sci_mode_enum, precision_value);
        },
        py::kw_only(),
        py::arg("profile"),
        py::arg("sci_mode") = py::none(),
        py::arg("precision") = py::none(),
        R"doc(

        Set print options for tensor output.

        Keyword Args:
            profile (const std::string): the profile to use for print options.
            sci_mode (Optional[str]): scientific notation mode. Can be None (auto-detect),
                                      True/False (force enable/disable), or "default" (auto-detect).
            precision (Optional[int]): number of digits after decimal point for floating point values.

        Returns:
            `None`: modifies print options.

        Examples:
            >>> ttnn.set_printoptions(profile="short")
            >>> ttnn.set_printoptions(profile="short", sci_mode=True)
            >>> ttnn.set_printoptions(profile="short", sci_mode=None)
            >>> ttnn.set_printoptions(profile="short", precision=6)
        )doc");

    module.def("dump_stack_trace_on_segfault", &ttnn::core::dump_stack_trace_on_segfault);

    module.def("get_current_command_queue_id_for_thread", &ttnn::core::get_current_command_queue_id_for_thread);
    module.def("push_current_command_queue_id_for_thread", &ttnn::core::push_current_command_queue_id_for_thread);
    module.def("pop_current_command_queue_id_for_thread", &ttnn::core::pop_current_command_queue_id_for_thread);
}

}  // namespace ttnn::core
