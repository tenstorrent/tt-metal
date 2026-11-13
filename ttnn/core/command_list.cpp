// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/command_list.hpp"

#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <string_view>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "ttnn/core.hpp"
#include "ttnn/device_operation_detail.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace {

thread_local ttnn::command_list::CommandListBuilder* t_recording = nullptr;

int64_t integer_in(
    const ttnn::command_list::CmdListScalar& scalar, int64_t min, int64_t max, std::string_view encoder) {
    if (const auto* real = std::get_if<double>(&scalar.value)) {
        TT_FATAL(
            std::trunc(*real) == *real && *real >= static_cast<double>(min) && *real <= static_cast<double>(max),
            "{} needs an integer in [{}, {}], got {}",
            encoder,
            min,
            max,
            *real);
        return static_cast<int64_t>(*real);
    }
    const int64_t integer = std::get<int64_t>(scalar.value);
    TT_FATAL(integer >= min && integer <= max, "{} needs an integer in [{}, {}], got {}", encoder, min, max, integer);
    return integer;
}

// Infinities and NaN pass through; finite values that float32 cannot hold are rejected rather than becoming inf.
float float32_of(const ttnn::command_list::CmdListScalar& scalar, std::string_view encoder) {
    const double real = std::visit([](auto value) { return static_cast<double>(value); }, scalar.value);
    TT_FATAL(
        !std::isfinite(real) || std::abs(real) <= std::numeric_limits<float>::max(),
        "{} needs a value in float32 range, got {}",
        encoder,
        real);
    return static_cast<float>(real);
}

uint32_t bfloat16_of(const ttnn::command_list::CmdListScalar& scalar, std::string_view encoder) {
    return fp32_to_bf16_bits_round_to_nearest_even(float32_of(scalar, encoder));
}

uint32_t uint16_of(const ttnn::command_list::CmdListScalar& scalar, std::string_view encoder) {
    return static_cast<uint32_t>(integer_in(scalar, 0, std::numeric_limits<uint16_t>::max(), encoder));
}

}  // namespace

namespace ttnn::command_list::detail {

namespace metal = tt::tt_metal::experimental;

struct RecordedTensor {
    std::string name;  // TTNN param
    metal::CmdListTensorArgInfo info;
};

struct RecordedScalar {
    std::string name;                       // TTNN param
    std::string metal_cmd_list_param_name;  // "<TTNN param>/<factory type>/<Metal param>"
    CmdListEncodeFn encode;
    std::variant<metal::CmdListRuntimeArgInfo, metal::CmdListCommonRuntimeArgInfo> info;
};

struct RecordedParameters {
    std::vector<RecordedTensor> tensors;
    std::vector<RecordedScalar> scalars;
};

namespace {

bool binds_keyword(const auto& bindings, const std::string& invoke_name) {
    return std::any_of(
        bindings.begin(), bindings.end(), [&](const auto& binding) { return binding.invoke_name == invoke_name; });
}

// Returns whether the program binds the keyword.
bool match_tensor_bind(
    const device_operation::detail::CmdListProgramBindings& program,
    const CmdListBind& bind,
    const Tensor& tensor,
    RecordedParameters& recorded) {
    const tt::tt_metal::MeshTensor& bind_tensor = tensor.mesh_tensor();
    bool matched = false;
    for (const auto& binding : program.bindings->tensors) {
        if (binding.invoke_name != bind.invoke_name) {
            continue;
        }
        const auto bound = std::find_if(program.bound_tensors.begin(), program.bound_tensors.end(), [&](const auto& b) {
            return b.first == binding.program_spec_param_name;
        });
        TT_FATAL(
            bound != program.bound_tensors.end(),
            "{} binds keyword '{}' to TensorParameter '{}', which the program does not bind",
            program.bindings->factory_name,
            bind.invoke_name,
            binding.program_spec_param_name);
        const tt::tt_metal::MeshTensor& program_tensor = *bound->second;
        // A copy, slice, or layout conversion the op made before its device operation is a different allocation.
        TT_FATAL(
            program_tensor.mesh_buffer().address() == bind_tensor.mesh_buffer().address() &&
                program_tensor.mesh_buffer().device() == bind_tensor.mesh_buffer().device(),
            "CmdListParam '{}' passed as '{}' is not the tensor the device operation reads: TensorParameter '{}' of "
            "{} is bound to a different allocation, so the op copied or converted it first",
            bind.name,
            bind.invoke_name,
            binding.program_spec_param_name,
            program.bindings->factory_name);
        // TTNN update_args cannot re-apply a view to the replacement tensor.
        TT_FATAL(
            program_tensor.tensor_spec() == bind_tensor.tensor_spec(),
            "CmdListParam '{}' passed as '{}' reaches TensorParameter '{}' of {} through a view with a different "
            "TensorSpec; pass the tensor the device operation reads",
            bind.name,
            bind.invoke_name,
            binding.program_spec_param_name,
            program.bindings->factory_name);
        recorded.tensors.push_back(
            {.name = bind.name, .info = {.program = *program.program, .param_name = binding.program_spec_param_name}});
        matched = true;
    }
    return matched;
}

// Returns whether the program binds the keyword.
bool match_scalar_bind(
    const device_operation::detail::CmdListProgramBindings& program,
    const CmdListBind& bind,
    RecordedParameters& recorded) {
    bool matched = false;
    for (const auto& binding : program.bindings->scalars) {
        if (binding.invoke_name != bind.invoke_name) {
            continue;
        }
        matched = true;
        const std::string metal_name =
            fmt::format("{}/{}/{}", bind.name, program.bindings->factory_name, binding.metal_cmd_list_param_name);
        for (const auto& arg : binding.runtime_args) {
            recorded.scalars.push_back(
                {.name = bind.name,
                 .metal_cmd_list_param_name = metal_name,
                 .encode = binding.encode,
                 .info = metal::CmdListRuntimeArgInfo{
                     .program = *program.program,
                     .kernel_name = arg.kernel_name,
                     .arg_name = arg.arg_name,
                     .nodes = arg.nodes}});
        }
        for (const auto& arg : binding.common_runtime_args) {
            recorded.scalars.push_back(
                {.name = bind.name,
                 .metal_cmd_list_param_name = metal_name,
                 .encode = binding.encode,
                 .info = metal::CmdListCommonRuntimeArgInfo{
                     .program = *program.program, .kernel_name = arg.kernel_name, .arg_name = arg.arg_name}});
        }
    }
    return matched;
}

// Matches one bind against one program's bindings and appends what matches. Returns whether the program binds the
// bind's keyword.
bool match_cmd_list_bind(
    const device_operation::detail::CmdListProgramBindings& program,
    const CmdListBind& bind,
    RecordedParameters& recorded) {
    const CmdListBindings& bindings = *program.bindings;
    if (const auto* tensor = std::get_if<Tensor>(&bind.value)) {
        TT_FATAL(
            !binds_keyword(bindings.scalars, bind.invoke_name),
            "CmdListParam '{}' is a tensor, but {} binds keyword '{}' to scalars",
            bind.name,
            bindings.factory_name,
            bind.invoke_name);
        return match_tensor_bind(program, bind, *tensor, recorded);
    }
    TT_FATAL(
        !binds_keyword(bindings.tensors, bind.invoke_name),
        "CmdListParam '{}' is a scalar, but {} binds keyword '{}' to tensors",
        bind.name,
        bindings.factory_name,
        bind.invoke_name);
    return match_scalar_bind(program, bind, recorded);
}

}  // namespace

}  // namespace ttnn::command_list::detail

namespace ttnn::device_operation::detail {

bool is_recording_command_list() { return t_recording != nullptr; }

void throw_if_recording_command_list(std::string_view call_name) {
    TT_FATAL(t_recording == nullptr, "{} cannot run while a CommandListBuilder is recording on this thread", call_name);
}

bool has_active_cmd_list_binds() { return t_recording != nullptr && t_recording->has_binds(); }

void record_to_command_list(
    std::string_view operation_name,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::span<const CmdListProgramBindings> programs) {
    auto* builder = t_recording;
    TT_FATAL(builder != nullptr, "{}: no CommandListBuilder is recording on this thread", operation_name);
    TT_FATAL(
        &builder->device() == mesh_device,
        "{} runs on a different MeshDevice than the recording CommandListBuilder",
        operation_name);
    try {
        builder->record(ttnn::get_current_command_queue_id_for_thread(), workload, programs);
    } catch (const std::exception& e) {
        TT_THROW("{} cannot be recorded into a command list: {}", operation_name, e.what());
    }
}

}  // namespace ttnn::device_operation::detail

namespace ttnn::command_list {

namespace metal = tt::tt_metal::experimental;

metal::ArgValue encode_uint32(const CmdListScalar& scalar) {
    return static_cast<uint32_t>(integer_in(scalar, 0, std::numeric_limits<uint32_t>::max(), "encode_uint32"));
}

metal::ArgValue encode_int32_bits(const CmdListScalar& scalar) {
    return static_cast<uint32_t>(integer_in(
        scalar, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), "encode_int32_bits"));
}

metal::ArgValue encode_float32_bits(const CmdListScalar& scalar) {
    return std::bit_cast<uint32_t>(float32_of(scalar, "encode_float32_bits"));
}

metal::ArgValue encode_bfloat16_pair(const CmdListScalar& scalar) {
    const uint32_t bits = bfloat16_of(scalar, "encode_bfloat16_pair");
    return (bits << 16) | bits;
}

metal::ArgValue encode_bfloat16_low(const CmdListScalar& scalar) { return bfloat16_of(scalar, "encode_bfloat16_low"); }

metal::ArgValue encode_bfloat16_high(const CmdListScalar& scalar) {
    return bfloat16_of(scalar, "encode_bfloat16_high") << 16;
}

metal::ArgValue encode_uint16_pair(const CmdListScalar& scalar) {
    const uint32_t bits = uint16_of(scalar, "encode_uint16_pair");
    return (bits << 16) | bits;
}

metal::ArgValue encode_uint16_low(const CmdListScalar& scalar) { return uint16_of(scalar, "encode_uint16_low"); }

metal::ArgValue encode_uint16_high(const CmdListScalar& scalar) {
    return uint16_of(scalar, "encode_uint16_high") << 16;
}

metal::ArgValue encode_float8_e4m3_quad(const CmdListScalar& scalar) {
    const float8_e4m3 value(float32_of(scalar, "encode_float8_e4m3_quad"));
    return pack_four_float8_e4m3_into_uint32(value, value, value, value);
}

void push_cmd_list_binds(std::string op_name, std::vector<CmdListBind> binds) {
    TT_FATAL(
        t_recording != nullptr, "{}: CmdListParams need a TTNN CommandListBuilder recording on this thread", op_name);
    t_recording->push_binds(std::move(op_name), std::move(binds));
}

void pop_cmd_list_binds(bool validate) {
    TT_FATAL(t_recording != nullptr, "pop_cmd_list_binds: no TTNN CommandListBuilder is recording on this thread");
    t_recording->pop_binds(validate);
}

CmdListBindScope::CmdListBindScope(std::string op_name, std::vector<CmdListBind> binds) {
    push_cmd_list_binds(std::move(op_name), std::move(binds));
}

CmdListBindScope::~CmdListBindScope() {
    if (!popped_) {
        pop_cmd_list_binds(false);
    }
}

void CmdListBindScope::finish() {
    TT_FATAL(!popped_, "CmdListBindScope::finish() called twice");
    popped_ = true;
    pop_cmd_list_binds(true);
}

CmdListBinder::CmdListBinder(const metal::ProgramSpec& spec, const metal::ProgramRunArgs& run_params) :
    spec_(spec), run_params_(run_params) {}

void CmdListBinder::tensor(std::string_view invoke_name, const metal::TensorParamName& program_spec_param_name) {
    TT_FATAL(!invoke_name.empty(), "CmdListBinder::tensor: invoke_name must not be empty");
    const bool declared =
        std::any_of(spec_.tensor_parameters.begin(), spec_.tensor_parameters.end(), [&](const auto& parameter) {
            return parameter.unique_id == program_spec_param_name;
        });
    TT_FATAL(declared, "CmdListBinder: the ProgramSpec has no TensorParameter '{}'", program_spec_param_name);
    TT_FATAL(
        bound_tensors_.insert(program_spec_param_name.get()).second,
        "CmdListBinder: TensorParameter '{}' is already bound",
        program_spec_param_name);
    bindings_.tensors.push_back(
        {.invoke_name = std::string(invoke_name), .program_spec_param_name = program_spec_param_name});
}

CmdListBinder::Param CmdListBinder::param(
    std::string_view invoke_name, CmdListEncodeFn encode, std::optional<std::string> metal_cmd_list_param_name) {
    TT_FATAL(!invoke_name.empty(), "CmdListBinder::param: invoke_name must not be empty");
    TT_FATAL(encode != nullptr, "CmdListBinder::param: '{}' has no encoder", invoke_name);
    std::string name = metal_cmd_list_param_name.value_or(std::string(invoke_name));
    TT_FATAL(
        declared_params_.insert(name).second,
        "CmdListBinder: Metal param '{}' is declared twice; a factory pairs one encoding with each Metal param. To "
        "give keyword '{}' a second encoding, pass metal_cmd_list_param_name.",
        name,
        invoke_name);
    bindings_.scalars.push_back(
        {.invoke_name = std::string(invoke_name),
         .metal_cmd_list_param_name = std::move(name),
         .encode = std::move(encode),
         .runtime_args = {},
         .common_runtime_args = {}});
    return Param(*this, bindings_.scalars.size() - 1);
}

CmdListBinder::Param& CmdListBinder::Param::runtime(
    const metal::KernelSpecName& kernel_name,
    std::string arg_name,
    std::optional<std::vector<metal::NodeCoord>> nodes) {
    const auto& names = binder_.kernel_spec(kernel_name).runtime_arg_schema.runtime_arg_names;
    TT_FATAL(
        std::find(names.begin(), names.end(), arg_name) != names.end(),
        "CmdListBinder: kernel '{}' has no runtime argument '{}'",
        kernel_name,
        arg_name);
    binder_.claim_arg(false, kernel_name, arg_name);

    const metal::ProgramRunArgs::KernelRunArgs::RuntimeArgValues::mapped_type* values = nullptr;
    for (const auto& kernel_run_args : binder_.run_params_.kernel_run_args) {
        if (kernel_run_args.kernel == kernel_name) {
            const auto it = kernel_run_args.runtime_arg_values.find(arg_name);
            if (it != kernel_run_args.runtime_arg_values.end()) {
                values = &it->second;
            }
            break;
        }
    }
    std::vector<metal::NodeCoord> resolved;
    if (nodes.has_value()) {
        for (const auto& node : *nodes) {
            TT_FATAL(
                values != nullptr && values->find(node) != values->end(),
                "CmdListBinder: the run args do not set runtime argument '{}' of kernel '{}' on node {}",
                arg_name,
                kernel_name,
                node);
        }
        resolved = std::move(*nodes);
    } else if (values != nullptr) {
        for (const auto& [node, value] : *values) {
            resolved.push_back(node);
        }
    }
    TT_FATAL(
        !resolved.empty(),
        "CmdListBinder: runtime argument '{}' of kernel '{}' is bound on no node",
        arg_name,
        kernel_name);
    binder_.bindings_.scalars[index_].runtime_args.push_back(
        {.kernel_name = kernel_name, .arg_name = std::move(arg_name), .nodes = std::move(resolved)});
    return *this;
}

CmdListBinder::Param& CmdListBinder::Param::common_runtime(
    const metal::KernelSpecName& kernel_name, std::string arg_name) {
    const auto& names = binder_.kernel_spec(kernel_name).runtime_arg_schema.common_runtime_arg_names;
    TT_FATAL(
        std::find(names.begin(), names.end(), arg_name) != names.end(),
        "CmdListBinder: kernel '{}' has no common runtime argument '{}'",
        kernel_name,
        arg_name);
    binder_.claim_arg(true, kernel_name, arg_name);
    binder_.bindings_.scalars[index_].common_runtime_args.push_back(
        {.kernel_name = kernel_name, .arg_name = std::move(arg_name)});
    return *this;
}

CmdListBindings CmdListBinder::finish() { return std::move(bindings_); }

const metal::KernelSpec& CmdListBinder::kernel_spec(const metal::KernelSpecName& kernel_name) const {
    const auto it = std::find_if(spec_.kernels.begin(), spec_.kernels.end(), [&](const auto& kernel) {
        return kernel.unique_id == kernel_name;
    });
    TT_FATAL(it != spec_.kernels.end(), "CmdListBinder: the ProgramSpec has no kernel '{}'", kernel_name);
    return *it;
}

void CmdListBinder::claim_arg(bool common, const metal::KernelSpecName& kernel_name, const std::string& arg) {
    TT_FATAL(
        bound_args_.emplace(common, kernel_name.get(), arg).second,
        "CmdListBinder: {} argument '{}' of kernel '{}' is already bound",
        common ? "common runtime" : "runtime",
        arg,
        kernel_name);
}

CommandListBuilder::CommandListBuilder(tt::tt_metal::distributed::MeshDevice& mesh_device) :
    metal_builder_(mesh_device) {}

CommandListBuilder::~CommandListBuilder() {
    if (t_recording == this) {
        t_recording = nullptr;
    }
}

void CommandListBuilder::begin_recording() {
    TT_FATAL(!recording_, "CommandListBuilder is already recording");
    TT_FATAL(t_recording == nullptr, "Another CommandListBuilder is recording on this thread");
    t_recording = this;
    recording_ = true;
}

void CommandListBuilder::end_recording() {
    TT_FATAL(t_recording == this, "This CommandListBuilder is not recording on this thread");
    TT_FATAL(
        bind_scopes_.empty(),
        "Cannot end recording while {} still has CmdListParam binds pushed",
        bind_scopes_.back().op_name);
    t_recording = nullptr;
    recording_ = false;
}

bool CommandListBuilder::is_recording() const { return recording_; }

ScopeGuard CommandListBuilder::record_scope() {
    begin_recording();
    return make_guard([this]() { end_recording(); });
}

CommandList CommandListBuilder::build(std::optional<QueueId> cq_id) const {
    ZoneScoped;
    const QueueId queue = cq_id.value_or(get_current_command_queue_id_for_thread());
    return CommandList(metal_builder_.build(device().mesh_command_queue(queue.get())), layout_);
}

void CommandListBuilder::clear() {
    TT_FATAL(!recording_, "Cannot clear a recording CommandListBuilder");
    metal_builder_.clear();
    layout_.clear();
}

void CommandListBuilder::deallocate() {
    TT_FATAL(!recording_, "Cannot deallocate a recording CommandListBuilder");
    metal_builder_.deallocate();
    layout_.clear();
}

tt::tt_metal::distributed::MeshDevice& CommandListBuilder::device() const { return metal_builder_.device(); }

void CommandListBuilder::record(
    QueueId binary_upload_cq_id,
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::span<const device_operation::detail::CmdListProgramBindings> programs) {
    detail::RecordedParameters parameters;
    for (const auto& program : programs) {
        for (auto& active : binds_) {
            if (detail::match_cmd_list_bind(program, active.bind, parameters)) {
                active.claimed = true;
            }
        }
    }
    // A Metal name keeps the encoder of its first TTNN record; later TTNN records of it come from the same factory.
    auto layout = layout_;
    metal::CmdListParameters metal_parameters;
    for (const auto& tensor : parameters.tensors) {
        auto& entry = layout[tensor.name];
        TT_FATAL(entry.metal_scalars.empty(), "CmdListParam '{}' is used as both a tensor and a scalar", tensor.name);
        entry.is_tensor = true;
        metal_parameters.tensor_parameters[metal::CmdListTensorArgName(tensor.name)].push_back(tensor.info);
    }
    for (const auto& scalar : parameters.scalars) {
        auto& entry = layout[scalar.name];
        TT_FATAL(!entry.is_tensor, "CmdListParam '{}' is used as both a tensor and a scalar", scalar.name);
        auto metal_scalar =
            std::find_if(entry.metal_scalars.begin(), entry.metal_scalars.end(), [&](const auto& candidate) {
                return candidate.metal_cmd_list_param_name == scalar.metal_cmd_list_param_name;
            });
        if (metal_scalar == entry.metal_scalars.end()) {
            entry.metal_scalars.push_back(
                {.metal_cmd_list_param_name = scalar.metal_cmd_list_param_name, .encode = scalar.encode});
            metal_scalar = std::prev(entry.metal_scalars.end());
        }
        if (const auto* info = std::get_if<metal::CmdListRuntimeArgInfo>(&scalar.info)) {
            metal_scalar->has_runtime = true;
            metal_parameters.runtime_parameters[metal::CmdListRuntimeArgName(scalar.metal_cmd_list_param_name)]
                .push_back(*info);
        } else {
            metal_scalar->has_common_runtime = true;
            metal_parameters
                .common_runtime_parameters[metal::CmdListCommonRuntimeArgName(scalar.metal_cmd_list_param_name)]
                .push_back(std::get<metal::CmdListCommonRuntimeArgInfo>(scalar.info));
        }
    }
    metal_builder_.add(device().mesh_command_queue(binary_upload_cq_id.get()), workload, metal_parameters);
    layout_ = std::move(layout);
}

void CommandListBuilder::push_binds(std::string op_name, std::vector<CmdListBind> binds) {
    TT_FATAL(recording_, "{}: CmdListParams need a recording TTNN CommandListBuilder", op_name);
    for (const auto& bind : binds) {
        TT_FATAL(!bind.name.empty(), "{}: CmdListParam name must not be empty", op_name);
        TT_FATAL(!bind.invoke_name.empty(), "{}: CmdListParam '{}' has no op argument name", op_name, bind.name);
        if (const auto* tensor = std::get_if<Tensor>(&bind.value)) {
            TT_FATAL(
                is_device_tensor(*tensor) && tensor->is_allocated(),
                "{}: CmdListParam '{}' must be an allocated device tensor",
                op_name,
                bind.name);
        }
    }
    bind_scopes_.push_back({.op_name = std::move(op_name), .start = binds_.size()});
    for (auto& bind : binds) {
        binds_.push_back({.bind = std::move(bind)});
    }
}

void CommandListBuilder::pop_binds(bool validate) {
    TT_FATAL(!bind_scopes_.empty(), "pop_cmd_list_binds without push_cmd_list_binds");
    const BindScope scope = std::move(bind_scopes_.back());
    bind_scopes_.pop_back();
    std::string unclaimed;
    for (std::size_t i = scope.start; i < binds_.size(); ++i) {
        if (!binds_[i].claimed) {
            unclaimed += fmt::format("{}'{}'", unclaimed.empty() ? "" : ", ", binds_[i].bind.name);
        }
    }
    binds_.resize(scope.start);
    // Validate after popping, so a throw leaves the stack consistent.
    TT_FATAL(
        !validate || unclaimed.empty(),
        "{}: CmdListParam(s) {} did not reach any recorded program field. Each must be passed by keyword to an op "
        "whose program factory maps that keyword in map_cmd_list_parameters: binder.tensor for tensors, "
        "binder.param for scalars.",
        scope.op_name,
        unclaimed);
}

bool CommandListBuilder::has_binds() const { return !binds_.empty(); }

CommandList::CommandList(metal::CommandList metal_list, detail::CmdListLayout layout) :
    metal_list_(std::move(metal_list)), layout_(std::move(layout)) {}

void CommandList::enqueue(bool blocking, bool check_dram_overlap) {
    ZoneScoped;
    device_operation::detail::throw_if_recording_command_list("CommandList::enqueue");
    metal::EnqueueCommandList(
        device().mesh_command_queue(metal_list_.cq_id()), metal_list_, blocking, check_dram_overlap);
}

void CommandList::update_args(const std::unordered_map<std::string, CmdListValue>& values, bool blocking) {
    ZoneScoped;
    metal::CmdListArgPatch patch;
    for (const auto& [name, value] : values) {
        const auto it = layout_.find(name);
        TT_FATAL(it != layout_.end(), "Unknown TTNN command-list parameter '{}'", name);
        if (it->second.is_tensor) {
            const auto* tensor = std::get_if<Tensor>(&value);
            TT_FATAL(tensor != nullptr, "TTNN command-list parameter '{}' is a tensor", name);
            patch.tensor_args.emplace(metal::CmdListTensorArgName(name), std::cref(tensor->mesh_tensor()));
            continue;
        }
        const auto* scalar = std::get_if<CmdListScalar>(&value);
        TT_FATAL(scalar != nullptr, "TTNN command-list parameter '{}' is a scalar", name);
        for (const auto& metal_scalar : it->second.metal_scalars) {
            const metal::ArgValue word = metal_scalar.encode(*scalar);
            if (metal_scalar.has_runtime) {
                patch.runtime_args.emplace(metal::CmdListRuntimeArgName(metal_scalar.metal_cmd_list_param_name), word);
            }
            if (metal_scalar.has_common_runtime) {
                patch.common_runtime_args.emplace(
                    metal::CmdListCommonRuntimeArgName(metal_scalar.metal_cmd_list_param_name), word);
            }
        }
    }
    metal_list_.update_args(patch, blocking);
}

void CommandList::deallocate() { metal_list_.deallocate(); }

tt::tt_metal::distributed::MeshDevice& CommandList::device() const { return metal_list_.device(); }

QueueId CommandList::cq_id() const { return QueueId(metal_list_.cq_id()); }

}  // namespace ttnn::command_list
