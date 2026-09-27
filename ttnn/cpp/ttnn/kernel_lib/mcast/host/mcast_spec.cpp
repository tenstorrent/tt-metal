// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_spec_common.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <type_traits>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {
namespace {
namespace m2 = tt::tt_metal::experimental;
namespace wire = dataflow_kernel_lib::mcast_wire;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;

std::string spec_name(std::string_view prefix, std::string_view field) {
    return std::string(prefix) + TT_MCAST_SPEC_STRING(TT_MCAST_SPEC_STEM) + std::string(field);
}

CoreRangeSet node_ranges(const m2::Nodes& nodes) {
    return std::visit(
        [](const auto& value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, CoreCoord>) {
                return CoreRangeSet(CoreRange(value, value));
            } else {
                return CoreRangeSet(value);
            }
        },
        nodes);
}

CoreRangeSet placement(const m2::ProgramSpec& spec, const m2::KernelSpecName& kernel) {
    std::vector<CoreRange> ranges;
    for (const auto& unit : spec.work_units) {
        const auto count = std::count(unit.kernels.begin(), unit.kernels.end(), kernel);
        TT_FATAL(count <= 1, "Duplicate multicast kernel in a work unit");
        if (count) {
            auto nodes = node_ranges(unit.target_nodes);
            for (const auto& range : nodes.ranges()) {
                for (const auto& previous : ranges) {
                    TT_FATAL(!previous.intersects(range), "Multicast kernel has overlapping work-unit placement");
                }
                ranges.push_back(range);
            }
        }
    }
    TT_FATAL(!ranges.empty(), "Multicast kernel has no placed nodes");
    return CoreRangeSet(std::move(ranges));
}

// Names are checked across the whole spec: reusing a prefix is not a runtime update.
std::vector<size_t> validate_targets(
    const m2::ProgramSpec& spec, std::string_view prefix, std::span<const m2::KernelSpecName> targets) {
    TT_FATAL(
        !prefix.empty() && (std::isalpha(static_cast<unsigned char>(prefix.front())) || prefix.front() == '_'),
        "Multicast prefix must be a C++ identifier");
    for (unsigned char ch : prefix) {
        TT_FATAL(std::isalnum(ch) || ch == '_', "Multicast prefix must be a C++ identifier");
    }
    std::vector<std::string> reserved;
#define ADD_RESERVED(P, field) reserved.push_back(spec_name(P, #field));
    TT_MCAST_SPEC_METADATA(ADD_RESERVED, prefix)
#undef ADD_RESERVED
    for (const auto* field :
         {"tag",
          "rt_base",
          "data_ready",
          "consumer_ready",
          "signal_source",
          "data_ready_type",
          "consumer_ready_type",
          "signal_source_type"}) {
        reserved.push_back(spec_name(prefix, field));
    }
    for (const auto& name : reserved) {
        TT_FATAL(name.size() <= m2::MAX_ACCESSOR_NAME_LENGTH, "Multicast generated name is too long");
    }
    std::set<m2::KernelSpecName> kernel_names;
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel_names.insert(kernel.unique_id).second, "Duplicate KernelSpec name");
        for (const auto& name : reserved) {
            TT_FATAL(
                !kernel.compile_time_args.contains(name) && !kernel.compiler_options.defines.contains(name) &&
                    std::find(
                        kernel.runtime_arg_schema.runtime_arg_names.begin(),
                        kernel.runtime_arg_schema.runtime_arg_names.end(),
                        name) == kernel.runtime_arg_schema.runtime_arg_names.end() &&
                    std::find(
                        kernel.runtime_arg_schema.common_runtime_arg_names.begin(),
                        kernel.runtime_arg_schema.common_runtime_arg_names.end(),
                        name) == kernel.runtime_arg_schema.common_runtime_arg_names.end(),
                "Multicast prefix or argument name is already in use: {}",
                name);
            const auto check_bindings = [&](const auto& bindings) {
                for (const auto& binding : bindings) {
                    TT_FATAL(binding.accessor_name != name, "Multicast resource accessor is already in use: {}", name);
                }
            };
            check_bindings(kernel.semaphore_bindings);
            check_bindings(kernel.dfb_bindings);
            check_bindings(kernel.scratchpad_bindings);
            check_bindings(kernel.tensor_bindings);
        }
    }
    TT_FATAL(!targets.empty(), "Multicast attachment requires at least one kernel");
    std::set<m2::KernelSpecName> selected;
    std::vector<size_t> indices;
    for (const auto& target : targets) {
        TT_FATAL(selected.insert(target).second, "Duplicate multicast attachment kernel");
        auto it = std::find_if(
            spec.kernels.begin(), spec.kernels.end(), [&](const auto& kernel) { return kernel.unique_id == target; });
        TT_FATAL(it != spec.kernels.end(), "Unknown multicast attachment kernel");
        TT_FATAL(it->is_data_movement_kernel(), "Multicast attachment requires a data-movement kernel");
        TT_FATAL(it->num_threads == 1, "Multicast topology requires one kernel thread per node");
        indices.push_back(std::distance(spec.kernels.begin(), it));
    }
    return indices;
}

NOC spec_noc(const m2::KernelSpec& kernel) {
    const auto& hw = std::get<m2::DataMovementHardwareConfig>(kernel.hw_config);
    if (const auto* gen1 = std::get_if<m2::DataMovementGen1Config>(&hw)) {
        return gen1->noc;
    }
    return NOC::NOC_0;  // Gen2 has one unified NoC.
}

uint32_t uniform_varargs(const m2::KernelSpec& kernel, const CoreRangeSet& nodes) {
    std::map<CoreCoord, uint32_t> counts;
    for (const auto& core : tt::tt_metal::corerange_to_cores(nodes)) {
        counts.emplace(core, kernel.advanced_options.num_runtime_varargs);
    }
    std::set<CoreCoord> overridden;
    for (const auto& [range, count] : kernel.advanced_options.num_runtime_varargs_per_node) {
        for (const auto& core : tt::tt_metal::corerange_to_cores(node_ranges(range))) {
            TT_FATAL(nodes.contains(core), "Multicast vararg schema targets an unplaced node");
            TT_FATAL(overridden.insert(core).second, "Overlapping multicast per-node vararg schemas");
            counts.at(core) = count;
        }
    }
    const auto count = counts.begin()->second;
    for (const auto& [core, value] : counts) {
        TT_FATAL(value == count, "Multicast attachment requires uniform runtime vararg counts");
    }
    return count;
}

void add_metadata(
    m2::KernelSpec& kernel,
    std::string_view prefix,
    const wire::FamilyMetadata& metadata,
    bool present,
    uint32_t base) {
    kernel.compile_time_args.emplace(spec_name(prefix, "tag"), present ? wire::FAMILY : wire::ABSENT);
    kernel.compile_time_args.emplace(spec_name(prefix, "rt_base"), base);
#define ADD_METADATA(P, field) \
    kernel.compile_time_args.emplace(spec_name(P, #field), static_cast<uint32_t>(metadata.field));
    TT_MCAST_SPEC_METADATA(ADD_METADATA, prefix)
#undef ADD_METADATA
}

constexpr std::array<std::string_view, 3> resource_roles{"data_ready", "consumer_ready", "signal_source"};
}  // namespace

void McastFamily::attach(
    m2::ProgramSpec& spec,
    m2::ProgramRunArgs& run_args,
    std::string_view prefix,
    std::span<const m2::KernelSpecName> targets,
    std::span<const m2::SemaphoreSpecName> adopted) const {
    require_arguments_prepared_();
    require_unbound_();
    TT_FATAL(!cfg_.base_sem_id && !cfg_.sem_ids, "ProgramSpec multicast attachment uses named semaphore resources");
    const auto indices = validate_targets(spec, prefix, targets);
    const auto count = required_semaphores_();
    TT_FATAL(adopted.empty() || adopted.size() == count, "Adopt exactly the required multicast semaphore roles");
    // Native value objects provide the transaction boundary; no retained invocation state.
    auto staged = spec;
    auto staged_args = run_args;
    std::set<m2::SemaphoreSpecName> semaphore_names;
    for (const auto& sem : staged.semaphores) {
        TT_FATAL(semaphore_names.insert(sem.unique_id).second, "Duplicate SemaphoreSpec name");
    }
    std::array<m2::SemaphoreSpecName, 3> names;
    for (uint32_t role = 0; role < count; ++role) {
        names[role] = adopted.empty() ? m2::SemaphoreSpecName(spec_name(prefix, resource_roles[role])) : adopted[role];
        for (uint32_t previous = 0; previous < role; ++previous) {
            TT_FATAL(names[role] != names[previous], "Multicast semaphore roles must use distinct resources");
        }
        if (adopted.empty()) {
            TT_FATAL(semaphore_names.insert(names[role]).second, "Multicast semaphore name is already in use");
            staged.semaphores.push_back({.unique_id = names[role], .target_nodes = participating_});
        } else {
            auto it = std::find_if(staged.semaphores.begin(), staged.semaphores.end(), [&](const auto& sem) {
                return sem.unique_id == names[role];
            });
            TT_FATAL(it != staged.semaphores.end(), "Unknown adopted multicast semaphore");
            TT_FATAL(it->advanced_options.initial_value == 0, "Adopted multicast semaphores must start at zero");
            TT_FATAL(
                participating_.subtract(node_ranges(it->target_nodes)).empty(),
                "Adopted multicast semaphore does not cover participating nodes");
        }
    }
    std::set<m2::KernelSpecName> runtime_kernels;
    for (const auto& args : staged_args.kernel_run_args) {
        TT_FATAL(runtime_kernels.insert(args.kernel).second, "Duplicate multicast kernel run-argument entry");
    }
    const bool chain = wire::transfer_mode(layout_.flags) == dataflow_kernel_lib::TransferMode::ChainUnicast;
    const auto grid = prepared_device_grid_;
    const uint32_t words =
        wire::runtime_words(layout_.rotating_span, layout_.rectangle_capacity, wire::transfer_mode(layout_.flags));
    for (const auto index : indices) {
        auto& kernel = staged.kernels[index];
        const auto nodes = placement(staged, kernel.unique_id);
        const auto base = uniform_varargs(kernel, nodes);
        TT_FATAL(base <= std::numeric_limits<uint32_t>::max() - words, "Multicast runtime vararg count overflows");
        auto args =
            std::find_if(staged_args.kernel_run_args.begin(), staged_args.kernel_run_args.end(), [&](const auto& item) {
                return item.kernel == kernel.unique_id;
            });
        if (args == staged_args.kernel_run_args.end()) {
            TT_FATAL(base == 0, "Missing caller runtime prefix for multicast attachment");
            staged_args.kernel_run_args.push_back({.kernel = kernel.unique_id});
            args = std::prev(staged_args.kernel_run_args.end());
        }
        auto& values = args->advanced_options.runtime_varargs;
        for (const auto& [node, value] : values) {
            TT_FATAL(nodes.contains(node), "Multicast runtime arguments target an unplaced node");
        }
        for (const auto& node : tt::tt_metal::corerange_to_cores(nodes)) {
            TT_FATAL(node.x < grid.x && node.y < grid.y, "Multicast kernel is placed outside the worker grid");
            if (const auto* group = group_for_core_(node)) {
                const bool sender =
                    std::find(group->senders_.begin(), group->senders_.end(), node) != group->senders_.end();
                TT_FATAL(
                    !(sender || chain) || spec_noc(kernel) == cfg_.noc,
                    "Multicast sender/forwarder NoC differs from its family");
            }
            auto entry = values.find(node);
            TT_FATAL(entry != values.end() || base == 0, "Missing caller runtime prefix for multicast attachment");
            auto& args_for_node = values[node];
            TT_FATAL(args_for_node.size() == base, "Multicast runtime varargs must match the declared prefix count");
            const auto payload = runtime_args_(node);
            args_for_node.insert(args_for_node.end(), payload.begin(), payload.end());
        }
        kernel.advanced_options.num_runtime_varargs = base + words;
        kernel.advanced_options.num_runtime_varargs_per_node.clear();
        add_metadata(kernel, prefix, layout_, true, base);
        for (uint32_t role = 0; role < resource_roles.size(); ++role) {
            const auto accessor = spec_name(prefix, resource_roles[role]);
            if (role < count) {
                kernel.semaphore_bindings.push_back({.semaphore_spec_name = names[role], .accessor_name = accessor});
            }
            kernel.compiler_options.defines.emplace(
                accessor + "_type", role < count ? "sem::" + accessor + "_t" : "std::nullptr_t");
        }
    }
    spec = std::move(staged);
    run_args = std::move(staged_args);
}

void attach_absent(m2::ProgramSpec& spec, std::string_view prefix, std::span<const m2::KernelSpecName> targets) {
    const auto indices = validate_targets(spec, prefix, targets);
    auto staged = spec;
    for (const auto index : indices) {
        auto& kernel = staged.kernels[index];
        placement(staged, kernel.unique_id);
        add_metadata(kernel, prefix, {}, false, 0);
        for (const auto role : resource_roles) {
            kernel.compiler_options.defines.emplace(spec_name(prefix, role) + "_type", "std::nullptr_t");
        }
    }
    spec = std::move(staged);
}

void Mcast1D::attach(
    m2::ProgramSpec& spec,
    m2::ProgramRunArgs& args,
    std::string_view prefix,
    std::span<const m2::KernelSpecName> kernels,
    std::span<const m2::SemaphoreSpecName> adopted) const {
    family_->attach(spec, args, prefix, kernels, adopted);
}
void Mcast2D::attach(
    m2::ProgramSpec& spec,
    m2::ProgramRunArgs& args,
    std::string_view prefix,
    std::span<const m2::KernelSpecName> kernels,
    std::span<const m2::SemaphoreSpecName> adopted) const {
    family_->attach(spec, args, prefix, kernels, adopted);
}
}  // namespace ttnn::kernel_lib::host
