// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/per_device_run_args.hpp"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <variant>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::device_operation::per_device_run_args {

using tt::tt_metal::experimental::ProgramRunArgs;
using KernelRunArgs = ProgramRunArgs::KernelRunArgs;

size_t KeyHash::operator()(const Key& key) const noexcept {
    size_t h = std::hash<std::string>{}(key.kernel) * 31 + std::hash<std::string>{}(key.arg);
    if (key.node.has_value()) {
        h ^= ((static_cast<size_t>(key.node->x) << 16) ^ key.node->y ^ (size_t{1} << 31)) * 0x9E3779B97F4A7C15ull;
    }
    return h;
}

namespace {

std::string describe(const Key& key) {
    return key.node.has_value() ? fmt::format("'{}' of kernel '{}' on node {}", key.arg, key.kernel, *key.node)
                                : fmt::format("common '{}' of kernel '{}'", key.arg, key.kernel);
}

const KernelRunArgs* kernel_args_of(const ProgramRunArgs& args, const std::string& kernel) {
    const auto it = std::find_if(args.kernel_run_args.begin(), args.kernel_run_args.end(), [&](const auto& k) {
        return k.kernel.get() == kernel;
    });
    return it == args.kernel_run_args.end() ? nullptr : &*it;
}

// Whether `kernel_args` lists the key's arg on the key's node (or as a common arg).
bool lists(const KernelRunArgs& kernel_args, const Key& key) {
    if (key.node.has_value()) {
        const auto scalar = kernel_args.runtime_arg_values.get(key.arg);
        if (scalar.has_value() && scalar->contains(*key.node)) {
            return true;
        }
        const auto array = kernel_args.runtime_array_arg_values.get(key.arg);
        return array.has_value() && array->contains(*key.node);
    }
    return kernel_args.common_runtime_arg_values.contains(key.arg) ||
           kernel_args.common_runtime_array_arg_values.contains(key.arg);
}

}  // namespace

ArgSchema arg_schema_of(const tt::tt_metal::experimental::ProgramSpec& spec) {
    ArgSchema schema;
    for (const auto& kernel : spec.kernels) {
        auto& args = schema[kernel.unique_id.get()];
        const auto& rta = kernel.runtime_arg_schema;
        args.runtime.insert(rta.runtime_arg_names.begin(), rta.runtime_arg_names.end());
        args.common.insert(rta.common_runtime_arg_names.begin(), rta.common_runtime_arg_names.end());
        for (const auto& array : rta.runtime_array_args) {
            args.runtime_arrays.emplace(array.name, array.size);
        }
        for (const auto& array : rta.common_runtime_array_args) {
            args.common_arrays.emplace(array.name, array.size);
        }
    }
    for (const auto& work_unit : spec.work_units) {
        const auto nodes = std::visit(
            [](const auto& target) -> tt::tt_metal::CoreRangeSet {
                using T = std::decay_t<decltype(target)>;
                if constexpr (std::is_same_v<T, tt::tt_metal::CoreRangeSet>) {
                    return target;
                } else if constexpr (std::is_same_v<T, tt::tt_metal::CoreRange>) {
                    return tt::tt_metal::CoreRangeSet(target);
                } else {
                    return tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(target, target));
                }
            },
            work_unit.target_nodes);
        for (const auto& kernel : work_unit.kernels) {
            auto& args = schema[kernel.get()];
            args.nodes = args.nodes.merge(nodes);
        }
    }
    return schema;
}

KeySet validate_and_collect_keys(
    const ttnn::MeshCoordinateRange& range, std::span<const DeviceRunArgs> per_device, const ArgSchema& schema) {
    KeySet keys;
    for (size_t i = 0; i < per_device.size(); ++i) {
        const auto& [device, values] = per_device[i];
        TT_FATAL(range.contains(device), "Per-device run args: device {} is not in range {}", device, range);
        for (size_t j = 0; j < i; ++j) {
            TT_FATAL(per_device[j].device != device, "Per-device run args: device {} is listed twice", device);
        }
        TT_FATAL(
            values.tensor_args.empty() && values.dfb_run_overrides.empty() &&
                values.advanced_options.sram_arena_args.empty() && values.advanced_options.prefetcher_pipe_args.empty(),
            "Per-device run args of device {} hold named kernel run args only (no tensor args, DFB overrides or "
            "advanced program args)",
            device);
        for (const auto& kernel_args : values.kernel_run_args) {
            const std::string& kernel = kernel_args.kernel.get();
            const auto declared_it = schema.find(kernel);
            TT_FATAL(declared_it != schema.end(), "Per-device run args of device {}: no kernel '{}'", device, kernel);
            const auto& declared = declared_it->second;
            TT_FATAL(
                kernel_args.advanced_options.runtime_varargs.empty() &&
                    kernel_args.advanced_options.common_runtime_varargs.empty(),
                "Per-device run args of kernel '{}' on device {}: varargs cannot be set per device",
                kernel,
                device);
            auto check_node = [&](const std::string& arg, const tt::tt_metal::CoreCoord& node) {
                TT_FATAL(
                    declared.nodes.contains(node),
                    "Per-device run args of device {}: kernel '{}' does not run on node {} ('{}')",
                    device,
                    kernel,
                    node.str(),
                    arg);
            };
            auto check_length = [&](const auto& arrays, const std::string& arg, size_t length) {
                const auto it = arrays.find(arg);
                TT_FATAL(
                    it != arrays.end(),
                    "Per-device run args of device {}: kernel '{}' declares no array '{}'",
                    device,
                    kernel,
                    arg);
                TT_FATAL(
                    length == it->second,
                    "Per-device run args of device {}: array '{}' of kernel '{}' has {} values, not {}",
                    device,
                    arg,
                    kernel,
                    length,
                    it->second);
            };
            for (const auto& [arg, per_node] : kernel_args.runtime_arg_values) {
                TT_FATAL(
                    declared.runtime.contains(arg),
                    "Per-device run args of device {}: kernel '{}' declares no runtime arg '{}'",
                    device,
                    kernel,
                    arg);
                for (const auto& [node, value] : per_node) {
                    check_node(arg, node);
                    keys.insert({kernel, arg, node});
                }
            }
            for (const auto& [arg, per_node] : kernel_args.runtime_array_arg_values) {
                for (const auto& [node, value] : per_node) {
                    check_length(declared.runtime_arrays, arg, value.size());
                    check_node(arg, node);
                    keys.insert({kernel, arg, node});
                }
            }
            for (const auto& [arg, value] : kernel_args.common_runtime_arg_values) {
                TT_FATAL(
                    declared.common.contains(arg),
                    "Per-device run args of device {}: kernel '{}' declares no common runtime arg '{}'",
                    device,
                    kernel,
                    arg);
                keys.insert({kernel, arg, std::nullopt});
            }
            for (const auto& [arg, value] : kernel_args.common_runtime_array_arg_values) {
                check_length(declared.common_arrays, arg, value.size());
                keys.insert({kernel, arg, std::nullopt});
            }
        }
    }
    return keys;
}

void check_base_covers(
    const ProgramRunArgs& base, const KeySet& prev, const KeySet& current, const ttnn::MeshCoordinateRange& range) {
    auto check = [&](const Key& key, const char* when) {
        const auto* kernel_args = kernel_args_of(base, key.kernel);
        TT_FATAL(
            kernel_args != nullptr && lists(*kernel_args, key),
            "Per-device run args of range {}: the base run args must list {}, which {} (list every "
            "arg set per device in the base run args, on every call)",
            range,
            describe(key),
            when);
    };
    for (const auto& key : current) {
        check(key, "this call sets per device");
    }
    for (const auto& key : prev) {
        if (!current.contains(key)) {
            check(key, "the previous call set per device");
        }
    }
}

ProgramRunArgs overlay(const ProgramRunArgs& base, const ProgramRunArgs& values) {
    ProgramRunArgs out = base;
    for (const auto& kernel_values : values.kernel_run_args) {
        auto it = std::find_if(out.kernel_run_args.begin(), out.kernel_run_args.end(), [&](const auto& k) {
            return k.kernel == kernel_values.kernel;
        });
        if (it == out.kernel_run_args.end()) {
            out.kernel_run_args.push_back(KernelRunArgs{.kernel = kernel_values.kernel});
            it = std::prev(out.kernel_run_args.end());
        }
        for (const auto& [arg, per_node] : kernel_values.runtime_arg_values) {
            auto& target = it->runtime_arg_values[arg];
            for (const auto& [node, value] : per_node) {
                target[node] = value;
            }
        }
        for (const auto& [arg, per_node] : kernel_values.runtime_array_arg_values) {
            auto& target = it->runtime_array_arg_values[arg];
            for (const auto& [node, value] : per_node) {
                target[node] = value;
            }
        }
        for (const auto& [arg, value] : kernel_values.common_runtime_arg_values) {
            it->common_runtime_arg_values[arg] = value;
        }
        for (const auto& [arg, value] : kernel_values.common_runtime_array_arg_values) {
            it->common_runtime_array_arg_values[arg] = value;
        }
    }
    return out;
}

const ProgramRunArgs* values_of(std::span<const DeviceRunArgs> per_device, const ttnn::MeshCoordinate& device) {
    const auto it =
        std::find_if(per_device.begin(), per_device.end(), [&](const auto& entry) { return entry.device == device; });
    return it == per_device.end() ? nullptr : &it->values;
}

namespace {
thread_local bool force_split_flag = false;
}  // namespace

ScopedForceSplit::ScopedForceSplit(tt::tt_metal::distributed::MeshDevice& mesh_device) :
    mesh_device_(mesh_device), previous_(force_split_flag) {
    mesh_device_.clear_program_cache();
    force_split_flag = true;
}

ScopedForceSplit::~ScopedForceSplit() {
    force_split_flag = previous_;
    mesh_device_.clear_program_cache();
}

bool force_split() { return force_split_flag; }

}  // namespace ttnn::device_operation::per_device_run_args
