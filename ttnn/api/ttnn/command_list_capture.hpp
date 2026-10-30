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
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/command_list.hpp>
#include <tt_stl/overloaded.hpp>

#include "ttnn/command_list_parameter.hpp"
#include "ttnn/device_operation.hpp"

// Experimental: records ttnn device operations into a Metal 2.0 command list, whose parameters are patched between
// replays without recompiling or re-recording.
namespace ttnn::experimental {

// An op that lets a command list patch some of its arguments. Every program built for one cache key must accept any
// patched value, so patchable scalars stay out of the op's program hash.
template <typename Op>
concept HasCommandListParameters =
    requires(const typename Op::operation_attributes_t& attrs, const typename Op::tensor_args_t& tensor_args) {
        { Op::command_list_parameters(attrs, tensor_args) } -> std::same_as<std::vector<CommandListParameter>>;
    };

// Maps an op's parameter names to the list parameters that patch them. Ops that bind the same list name read one
// value, such as a sequence length shared by every layer; distinct names keep the values apart.
using CommandListBindings = std::unordered_map<std::string, std::string>;

// New values for some parameters of a command list.
class CommandListPatch {
public:
    CommandListPatch& set(const std::string& name, uint32_t value) {
        scalars_.insert_or_assign(name, value);
        return *this;
    }
    CommandListPatch& set(const std::string& name, const Tensor& tensor) {
        tensors_.insert_or_assign(name, tensor);
        return *this;
    }

private:
    friend class CapturedCommandList;
    std::unordered_map<std::string, uint32_t> scalars_;
    std::unordered_map<std::string, Tensor> tensors_;
};

// A built command list and the parameters it can patch.
class CapturedCommandList {
public:
    // Applies every value to replays enqueued after this call. Throws, writing nothing, for an unknown name, a value
    // over the smallest capacity of the ops that read it, or a tensor whose spec differs from the recorded one.
    void update(const CommandListPatch& patch) {
        namespace m2 = tt::tt_metal::experimental;
        m2::CmdListArgPatch metal_patch;
        for (const auto& [name, value] : patch.scalars_) {
            const auto it = capacities_.find(name);
            TT_FATAL(it != capacities_.end(), "Command list has no scalar parameter '{}'", name);
            TT_FATAL(
                value <= it->second,
                "Command list parameter '{}' = {} exceeds its capacity {}",
                name,
                value,
                it->second);
            metal_patch.common_runtime_args[m2::CmdListCommonRuntimeArgName{name}] = value;
        }
        for (const auto& [name, tensor] : patch.tensors_) {
            TT_FATAL(tensors_.contains(name), "Command list has no tensor parameter '{}'", name);
            metal_patch.tensor_args.emplace(
                m2::CmdListTensorArgName{name}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor.mesh_tensor())});
        }
        list_.update_args(metal_patch);
    }

    void set(const std::string& name, uint32_t value) { update(CommandListPatch{}.set(name, value)); }

    uint32_t capacity(const std::string& name) const { return capacities_.at(name); }

    void replay(bool blocking = false) {
        auto& cq = list_.device().mesh_command_queue(list_.cq_id());
        tt::tt_metal::experimental::EnqueueCommandList(cq, list_, blocking);
    }

private:
    friend class CommandListCapture;
    CapturedCommandList(
        tt::tt_metal::experimental::CommandList list,
        std::unordered_map<std::string, uint32_t> capacities,
        std::unordered_set<std::string> tensors) :
        list_(std::move(list)), capacities_(std::move(capacities)), tensors_(std::move(tensors)) {}

    tt::tt_metal::experimental::CommandList list_;
    std::unordered_map<std::string, uint32_t> capacities_;
    std::unordered_set<std::string> tensors_;
};

// Records device operations without dispatching them. Each record() call compiles on a program-cache miss and reuses
// the cached program on a hit, exactly like launch(). Only Metal 2.0 (ProgramSpec) programs can be recorded. The
// list replays the raw addresses of the recorded tensors, so the caller keeps inputs and outputs alive.
class CommandListCapture {
public:
    explicit CommandListCapture(ttnn::MeshDevice& device) : builder_(device) {}

    // Only the op parameters named in `bindings` become patchable; the rest replay as recorded.
    template <ttnn::device_operation::DeviceOperationConcept Op>
    typename Op::tensor_return_value_t record(
        const typename Op::operation_attributes_t& attrs,
        const typename Op::tensor_args_t& tensor_args,
        const CommandListBindings& bindings = {}) {
        namespace detail = ttnn::device_operation::detail;
        using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Op>;
        namespace m2 = tt::tt_metal::experimental;

        std::vector<CommandListParameter> declared;
        if constexpr (HasCommandListParameters<Op>) {
            declared = Op::command_list_parameters(attrs, tensor_args);
        }
        std::vector<std::pair<std::string, CommandListParameter>> bound;
        for (const auto& [op_name, list_name] : bindings) {
            const auto before = bound.size();
            for (const auto& parameter : declared) {
                if (parameter.name == op_name) {
                    bound.emplace_back(list_name, parameter);
                }
            }
            TT_FATAL(bound.size() > before, "Operation has no command list parameter '{}'", op_name);
        }
        std::unordered_map<std::string, bool> patches_tensor;
        for (const auto& [list_name, parameter] : bound) {
            const bool is_tensor = std::holds_alternative<TensorTarget>(parameter.target);
            const auto [it, inserted] = patches_tensor.emplace(list_name, is_tensor);
            const bool clash =
                it->second != is_tensor || (is_tensor ? capacities_.contains(list_name) : tensors_.contains(list_name));
            TT_FATAL(!clash, "Command list parameter '{}' cannot patch both a tensor and a scalar", list_name);
        }

        std::vector<std::reference_wrapper<const Tensor>> input_tensors;
        ttsl::reflection::visit_object_of_type<Tensor>(
            [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);
        detail::validate_input_tensors<Op>(attrs, input_tensors);

        auto outputs = Op::create_output_tensors(attrs, tensor_args);
        auto* mesh_device = detail::get_mesh_device<Op>(attrs, tensor_args);
        TT_FATAL(mesh_device == &builder_.device(), "Cannot record an operation on another MeshDevice");
        detail::place_output_tensors<Op>(attrs, tensor_args, input_tensors, mesh_device, outputs);

        const auto lookup = detail::look_up_program_cache<Adapter>(attrs, tensor_args, mesh_device);
        detail::use_mesh_workload<Adapter>(
            attrs, tensor_args, outputs, mesh_device, lookup, [&](tt::tt_metal::distributed::MeshWorkload& workload) {
                m2::CmdListParameters cmd_list_parameters;
                for (const auto& [list_name, parameter] : bound) {
                    for (const auto& [range, program] : workload.get_programs()) {
                        std::visit(
                            ttsl::overloaded{
                                [&](const CommonRuntimeArgTarget& target) {
                                    cmd_list_parameters
                                        .common_runtime_parameters[m2::CmdListCommonRuntimeArgName{list_name}]
                                        .push_back(m2::CmdListCommonRuntimeArgInfo{
                                            .program = std::cref(program),
                                            .kernel_name = target.kernel,
                                            .arg_name = target.arg_name});
                                },
                                [&](const TensorTarget& target) {
                                    cmd_list_parameters.tensor_parameters[m2::CmdListTensorArgName{list_name}]
                                        .push_back(m2::CmdListTensorArgInfo{
                                            .program = std::cref(program), .param_name = target.parameter});
                                }},
                            parameter.target);
                    }
                }
                builder_.add(workload, cmd_list_parameters);
            });

        for (const auto& [list_name, parameter] : bound) {
            if (const auto* scalar = std::get_if<CommonRuntimeArgTarget>(&parameter.target)) {
                auto [it, inserted] = capacities_.emplace(list_name, scalar->capacity);
                if (!inserted) {
                    it->second = std::min(it->second, scalar->capacity);
                }
            } else {
                tensors_.insert(list_name);
            }
        }
        return outputs;
    }

    CapturedCommandList build(uint8_t cq_id = 0) const {
        return CapturedCommandList(builder_.build(builder_.device().mesh_command_queue(cq_id)), capacities_, tensors_);
    }

private:
    tt::tt_metal::experimental::CommandListBuilder builder_;
    std::unordered_map<std::string, uint32_t> capacities_;
    std::unordered_set<std::string> tensors_;
};

}  // namespace ttnn::experimental
