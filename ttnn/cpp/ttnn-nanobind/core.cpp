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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <reflect>

#include "tt-metalium/lightmetal_binary.hpp"
#include "tt-metalium/lightmetal_replay.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_stl/caseless_comparison.hpp"
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
        auto name_str = std::string{reflect::member_name<I, ttnn::Config::attributes_t>()};
        if (name_str != "report_name") {
            py_config.def_prop_rw(name_str.c_str(), &ttnn::Config::get<I>, &ttnn::Config::set<I>);
        }
    });
    // Override report_name to accept None (std::optional) in addition to str/PathLike
    py_config.def_prop_rw(
        "report_name",
        [](const ttnn::Config& cfg) { return cfg.get<"report_name">(); },
        [](ttnn::Config& cfg, nb::object obj) {
            if (obj.is_none()) {
                cfg.set<"report_name">(std::optional<std::filesystem::path>{});
            } else {
                auto p = nb::cast<std::filesystem::path>(obj);
                cfg.set<"report_name">(std::optional<std::filesystem::path>{p});
            }
        });
    py_config.def_prop_ro("report_path", &ttnn::Config::get<"report_path">);

    // Explicit helper that clears the optional report_name
    mod.def(
        "clear_report_name",
        [](ttnn::Config& cfg) { cfg.set<"report_name">(std::optional<std::filesystem::path>{}); },
        nb::arg("config"));

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
        [](const std::string& profile, const nb::object& sci_mode, const nb::object& precision) {
            ttnn::TensorPrintProfile profile_enum =
                enchantum::cast<ttnn::TensorPrintProfile>(profile, ttsl::ascii_caseless_comp).value();

            ttnn::SciMode sci_mode_enum = ttnn::SciMode::Default;
            if (!sci_mode.is_none()) {
                if (nb::isinstance<nb::bool_>(sci_mode)) {
                    sci_mode_enum = nb::cast<bool>(sci_mode) ? ttnn::SciMode::Enable : ttnn::SciMode::Disable;
                } else if (nb::isinstance<nb::str>(sci_mode)) {
                    auto cmp = [](const auto& a, const auto& b) -> bool {
                        return ttsl::ascii_caseless_comp(std::string_view(a), std::string_view(b));
                    };
                    const std::string sci_mode_str = nb::cast<std::string>(sci_mode);
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
                if (nb::isinstance<nb::int_>(precision)) {
                    precision_value = nb::cast<int>(precision);
                } else {
                    throw std::invalid_argument("precision must be None or int");
                }
            }

            ttnn::set_printoptions(profile_enum, sci_mode_enum, precision_value);
        },
        nb::kw_only(),
        nb::arg("profile"),
        nb::arg("sci_mode") = nb::none(),
        nb::arg("precision") = nb::none(),
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

    mod.def("dump_stack_trace_on_segfault", &ttnn::core::dump_stack_trace_on_segfault);

    mod.def("get_current_command_queue_id_for_thread", &ttnn::core::get_current_command_queue_id_for_thread);
    mod.def("push_current_command_queue_id_for_thread", &ttnn::core::push_current_command_queue_id_for_thread);
    mod.def("pop_current_command_queue_id_for_thread", &ttnn::core::pop_current_command_queue_id_for_thread);
}

}  // namespace ttnn::core
