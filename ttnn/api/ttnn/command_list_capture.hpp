// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/command_list.hpp>

#include "ttnn/device_operation.hpp"

// Experimental: records ttnn device operations into a Metal 2.0 command list, whose named scalars are patched between
// replays without recompiling or re-recording.
namespace ttnn::experimental {

// A host scalar an op passes to one kernel as a named common runtime argument.
struct CommandListParameter {
    // Shared by every op in a list that reads the same value, such as the sequence length; one patch updates all.
    std::string name;
    tt::tt_metal::experimental::KernelSpecName kernel;
    std::string arg_name;
    // Largest value the op's program and tensors support.
    uint32_t capacity = 0;
};

// An op whose program reads some of its scalars from patchable common runtime arguments. Every program built for one
// cache key must accept any value up to the capacity, so those scalars stay out of the op's program hash.
template <typename Op>
concept HasCommandListParameters =
    requires(const typename Op::operation_attributes_t& attrs, const typename Op::tensor_args_t& tensor_args) {
        { Op::command_list_parameters(attrs, tensor_args) } -> std::same_as<std::vector<CommandListParameter>>;
    };

// A built command list and the capacity of each parameter it can patch.
class CapturedCommandList {
public:
    // Applies to replays enqueued after this call. Throws, writing nothing, for an unknown name or a value over the
    // smallest capacity of the ops that read it.
    void set(const std::string& name, uint32_t value) {
        const auto it = capacities_.find(name);
        TT_FATAL(it != capacities_.end(), "Command list has no parameter '{}'", name);
        TT_FATAL(
            value <= it->second, "Command list parameter '{}' = {} exceeds its capacity {}", name, value, it->second);
        tt::tt_metal::experimental::CmdListArgPatch patch;
        patch.common_runtime_args[tt::tt_metal::experimental::CmdListCommonRuntimeArgName{name}] = value;
        list_.update_args(patch);
    }

    uint32_t capacity(const std::string& name) const { return capacities_.at(name); }

    void replay(bool blocking = false) {
        auto& cq = list_.device().mesh_command_queue(list_.cq_id());
        tt::tt_metal::experimental::EnqueueCommandList(cq, list_, blocking);
    }

private:
    friend class CommandListCapture;
    CapturedCommandList(
        tt::tt_metal::experimental::CommandList list, std::unordered_map<std::string, uint32_t> capacities) :
        list_(std::move(list)), capacities_(std::move(capacities)) {}

    tt::tt_metal::experimental::CommandList list_;
    std::unordered_map<std::string, uint32_t> capacities_;
};

// Records device operations without dispatching them. Each record() call compiles on a program-cache miss and reuses
// the cached program on a hit, exactly like launch(). Only Metal 2.0 (ProgramSpec) programs can be recorded. The
// list replays the raw addresses of the recorded tensors, so the caller keeps inputs and outputs alive.
class CommandListCapture {
public:
    explicit CommandListCapture(ttnn::MeshDevice& device) : builder_(device) {}

    template <ttnn::device_operation::DeviceOperationConcept Op>
    typename Op::tensor_return_value_t record(
        const typename Op::operation_attributes_t& attrs, const typename Op::tensor_args_t& tensor_args) {
        namespace detail = ttnn::device_operation::detail;
        using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Op>;
        namespace m2 = tt::tt_metal::experimental;

        std::vector<std::reference_wrapper<const Tensor>> input_tensors;
        ttsl::reflection::visit_object_of_type<Tensor>(
            [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);
        detail::validate_input_tensors<Op>(attrs, input_tensors);

        auto outputs = Op::create_output_tensors(attrs, tensor_args);
        auto* mesh_device = detail::get_mesh_device<Op>(attrs, tensor_args);
        TT_FATAL(mesh_device == &builder_.device(), "Cannot record an operation on another MeshDevice");
        detail::place_output_tensors<Op>(attrs, tensor_args, input_tensors, mesh_device, outputs);

        std::vector<CommandListParameter> parameters;
        if constexpr (HasCommandListParameters<Op>) {
            parameters = Op::command_list_parameters(attrs, tensor_args);
        }

        const auto lookup = detail::look_up_program_cache<Adapter>(attrs, tensor_args, mesh_device);
        detail::use_mesh_workload<Adapter>(
            attrs, tensor_args, outputs, mesh_device, lookup, [&](tt::tt_metal::distributed::MeshWorkload& workload) {
                m2::CmdListParameters cmd_list_parameters;
                for (const auto& parameter : parameters) {
                    auto& targets =
                        cmd_list_parameters.common_runtime_parameters[m2::CmdListCommonRuntimeArgName{parameter.name}];
                    for (const auto& [range, program] : workload.get_programs()) {
                        targets.push_back(m2::CmdListCommonRuntimeArgInfo{
                            .program = std::cref(program),
                            .kernel_name = parameter.kernel,
                            .arg_name = parameter.arg_name});
                    }
                }
                builder_.add(workload, cmd_list_parameters);
            });

        for (const auto& parameter : parameters) {
            auto [it, inserted] = capacities_.emplace(parameter.name, parameter.capacity);
            if (!inserted) {
                it->second = std::min(it->second, parameter.capacity);
            }
        }
        return outputs;
    }

    CapturedCommandList build(uint8_t cq_id = 0) const {
        return CapturedCommandList(builder_.build(builder_.device().mesh_command_queue(cq_id)), capacities_);
    }

private:
    tt::tt_metal::experimental::CommandListBuilder builder_;
    std::unordered_map<std::string, uint32_t> capacities_;
};

}  // namespace ttnn::experimental
