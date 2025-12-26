// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>  // size_t
#include <set>
#include <string>
#include <type_traits>  // is_same_v, decay
#include <utility>      // index_sequence, forward

#include <fmt/format.h>
#include <reflect>

#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tracy/Tracy.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "ttnn/graph/graph_processor.hpp"

// Forward declaration for database functions
namespace ttnn::database {
void insert_operation(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::string& operation_name,
    std::optional<double> duration_ms);
void insert_devices(
    const std::filesystem::path& report_path, const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);
void insert_buffers(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);
void insert_buffer_pages(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);
void insert_captured_graph(
    const std::filesystem::path& report_path, uint64_t operation_id, const nlohmann::json& captured_graph);
void insert_input_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors);
void insert_output_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors);
void insert_stack_trace(const std::filesystem::path& report_path, uint64_t operation_id);
uint64_t get_next_operation_id();
}  // namespace ttnn::database

namespace ttnn {
namespace decorators {

using Tensors = tt::tt_metal::operation::Tensors;
using OptionalTensors = tt::tt_metal::operation::OptionalTensors;
using OptionalConstTensors = tt::tt_metal::operation::OptionalConstTensors;

namespace detail {

// Helper to extract devices from operation arguments
template <typename T>
void collect_devices_from_arg(std::set<tt::tt_metal::distributed::MeshDevice*>& devices, const T& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
        if (arg.storage_type() == StorageType::DEVICE && arg.is_allocated()) {
            devices.insert(arg.device());
        }
    } else if constexpr (std::is_same_v<std::decay_t<T>, tt::tt_metal::distributed::MeshDevice*>) {
        if (arg != nullptr) {
            devices.insert(arg);
        }
    } else if constexpr (std::is_same_v<std::decay_t<T>, tt::tt_metal::distributed::MeshDevice&>) {
        devices.insert(&arg);
    }
    // Other types are ignored
}

template <typename... Args>
std::vector<tt::tt_metal::distributed::MeshDevice*> extract_devices(const Args&... args) {
    std::set<tt::tt_metal::distributed::MeshDevice*> device_set;
    (collect_devices_from_arg(device_set, args), ...);
    return std::vector<tt::tt_metal::distributed::MeshDevice*>(device_set.begin(), device_set.end());
}

// Helper to collect tensors from arguments
template <typename T>
void collect_tensors_from_arg(std::vector<Tensor>& tensors, const T& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
        tensors.push_back(arg);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::optional<Tensor>>) {
        if (arg.has_value()) {
            tensors.push_back(arg.value());
        }
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::vector<Tensor>>) {
        for (const auto& t : arg) {
            tensors.push_back(t);
        }
    }
    // Other types are ignored
}

template <typename... Args>
std::vector<Tensor> extract_input_tensors(const Args&... args) {
    std::vector<Tensor> tensors;
    (collect_tensors_from_arg(tensors, args), ...);
    return tensors;
}

// Helper to extract output tensors from return value
template <typename T>
std::vector<Tensor> extract_output_tensors(const T& output) {
    std::vector<Tensor> tensors;
    if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
        tensors.push_back(output);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::optional<Tensor>>) {
        if (output.has_value()) {
            tensors.push_back(output.value());
        }
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::vector<Tensor>>) {
        tensors = output;
    }
    return tensors;
}

// Get "add" from "ttnn::add"
static std::string base_name(const std::string& cpp_fully_qualified_name) {
    auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
    return last_token;
}

// Convert "ttnn::add" to "add_t"
inline std::string class_name(const std::string& cpp_fully_qualified_name) {
    return base_name(cpp_fully_qualified_name) + "_t";
}

// Convert "ttnn::add" to "ttnn.add"
inline std::string python_fully_qualified_name(const std::string& cpp_fully_qualified_name) {
    auto replace = [](const std::string& input, const std::string& from, const std::string& to) {
        if (from.empty()) {
            return input;
        }
        auto output = input;
        size_t start = 0;
        while ((start = output.find(from, start)) != std::string::npos) {
            output.replace(start, from.length(), to);
            start += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
        };
        return output;
    };
    return replace(cpp_fully_qualified_name, "::", ".");
}

}  // namespace detail

// Primitive operations map directly to device operations
template <typename operation_t>
concept PrimitiveOperationConcept = device_operation::DeviceOperationConcept<operation_t>;

// Composite operation allows any code to be executed
template <typename operation_t>
concept CompositeOperationConcept = !PrimitiveOperationConcept<operation_t>;

template <typename Op, typename... Args>
concept HasInvoke = requires {
    { Op::invoke(std::declval<Args>()...) };
};

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
struct registered_operation_t {
    static constexpr auto is_primitive = PrimitiveOperationConcept<operation_t>;

    // Get "add" from "ttnn::add"
    std::string base_name() const { return detail::base_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "add_t"
    std::string class_name() const { return detail::class_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "ttnn.add"
    std::string python_fully_qualified_name() const {
        return detail::python_fully_qualified_name(std::string{cpp_fully_qualified_name});
    }

    template <typename... Args>
        requires(HasInvoke<operation_t, Args && ...>)
    auto operator()(Args&&... args) const {
        return traced_invoke(std::forward<Args>(args)...);
    }

private:
    template <typename... args_t>
    auto traced_invoke(args_t&&... args) const {
        log_debug(tt::LogOp, "Started C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});

        // Check if logging is enabled
        const bool logging_enabled =
            !ttnn::CONFIG.get<"enable_fast_runtime_mode">() && ttnn::CONFIG.get<"enable_logging">();
        const auto report_path = ttnn::CONFIG.get<"report_path">();
        const bool should_log = logging_enabled && report_path.has_value();

        uint64_t operation_id = 0;
        std::vector<tt::tt_metal::distributed::MeshDevice*> devices;

        std::vector<Tensor> input_tensors;

        if (should_log) {
            operation_id = ttnn::database::get_next_operation_id();
            devices = detail::extract_devices(args...);
            input_tensors = detail::extract_input_tensors(args...);

            // Synchronize devices before operation
            for (auto* device : devices) {
                tt::tt_metal::distributed::Synchronize(device, std::nullopt);
            }

            // Pre-operation database inserts
            ttnn::database::insert_operation(
                report_path.value(), operation_id, python_fully_qualified_name(), std::nullopt);

            ttnn::database::insert_stack_trace(report_path.value(), operation_id);
            ttnn::database::insert_input_tensors(report_path.value(), operation_id, input_tensors);
        }

        tt::tt_metal::GraphTracker::instance().track_function_start(cpp_fully_qualified_name, args...);

        // Begin graph capture
        ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto output = invoke(std::forward<args_t>(args)...);
        auto end_time = std::chrono::high_resolution_clock::now();

        // End graph capture
        auto captured_graph = ttnn::graph::GraphProcessor::end_graph_capture();

        tt::tt_metal::GraphTracker::instance().track_function_end(output);
        log_debug(tt::LogOp, "Finished invoking C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});

        if (should_log) {
            // Synchronize devices after operation
            for (auto* device : devices) {
                tt::tt_metal::distributed::Synchronize(device, std::nullopt);
            }

            // Calculate duration in milliseconds
            auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            // Extract output tensors
            auto output_tensors = detail::extract_output_tensors(output);

            // Post-operation database inserts
            ttnn::database::insert_devices(report_path.value(), devices);
            ttnn::database::insert_operation(
                report_path.value(), operation_id, python_fully_qualified_name(), duration_ms);
            ttnn::database::insert_output_tensors(report_path.value(), operation_id, output_tensors);
            ttnn::database::insert_buffers(report_path.value(), operation_id, devices);

            if (ttnn::CONFIG.get<"enable_detailed_buffer_report">()) {
                ttnn::database::insert_buffer_pages(report_path.value(), operation_id, devices);
            }

            if (!captured_graph.is_null()) {
                ttnn::database::insert_captured_graph(report_path.value(), operation_id, captured_graph);
            }
        }

        return output;
    }

    template <typename... args_t>
        requires PrimitiveOperationConcept<operation_t>
    auto invoke(args_t&&... args) const {
        static_assert(
            requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
            "Primitive Operation must implement invoke() method to be invoked.");
        auto [operation_attributes, tensors_args] = operation_t::invoke(std::forward<decltype(args)>(args)...);
        return ttnn::device_operation::detail::invoke<operation_t>(operation_attributes, tensors_args);
    }

    template <typename... args_t>
        requires(CompositeOperationConcept<operation_t>)
    auto invoke(args_t&&... args) const {
        return invoke_composite(std::forward<args_t>(args)...);
    }

    template <typename... args_t>
    auto invoke_composite(args_t&&... args) const {
        return operation_t::invoke(std::forward<decltype(args)>(args)...);
    }
};

template <reflect::fixed_string cpp_fully_qualified_name>
struct operation_name_key_t {
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>);
};

template <typename operation_t>
struct operation_key_t {
    friend consteval auto get(operation_key_t<operation_t>);
};

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, auto operation>
struct set_operation_t : std::true_type {
    friend consteval auto get(operation_key_t<operation_t>) { return operation; }
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>) { return operation; }
};

constexpr reflect::fixed_string prim_namespace = "ttnn::prim";

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
consteval void assert_operation_in_correct_namespace() {
    if constexpr (PrimitiveOperationConcept<operation_t>) {
        if constexpr (cpp_fully_qualified_name.size() > prim_namespace.size()) {
            constexpr auto namespace_substring =
                tt::stl::reflection::fixed_string_substring<0, prim_namespace.size()>(cpp_fully_qualified_name);
            static_assert(
                tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace),
                "Primitive operations must be in the `ttnn::prim` namespace.");
        } else {
#ifndef DISABLE_NAMESPACE_STATIC_ASSERT
            static_assert(false, "Primitive operations must be in the `ttnn::prim` namespace.");
#endif
        }
    } else {
        if constexpr (cpp_fully_qualified_name.size() > prim_namespace.size()) {
            constexpr auto namespace_substring =
                tt::stl::reflection::fixed_string_substring<0, prim_namespace.size()>(cpp_fully_qualified_name);
            static_assert(
                not tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace),
                "Composite operations must not be in the `ttnn::prim` namespace. You may have forgotten to implement "
                "one of: validate_on_program_cache_hit, validate_on_program_cache_miss, create_output_tensors, or "
                "select_program_factory.");
        }
    }
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation_impl() {
    assert_operation_in_correct_namespace<cpp_fully_qualified_name, operation_t>();
    constexpr auto operation = registered_operation_t<cpp_fully_qualified_name, operation_t>{};
    static_assert(
        not requires(operation_name_key_t<cpp_fully_qualified_name> key) { get(key); },
        "Operation with this `cpp_fully_qualified_name` was already registered. Please use a different name.");
    static_assert(
        not requires(operation_key_t<operation_t> key) { get(key); },
        "Operation with this `operation_t` was already registered. Please use a different type.");
    static_assert(set_operation_t<cpp_fully_qualified_name, operation_t, operation>::value);
    return operation;
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation() {
    return register_operation_impl<cpp_fully_qualified_name, operation_t>();
}

}  // namespace decorators

using ttnn::decorators::register_operation;

}  // namespace ttnn
