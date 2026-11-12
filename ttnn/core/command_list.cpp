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

namespace ttnn::device_operation::detail {

bool is_recording_command_list() { return t_recording != nullptr; }

void throw_if_recording_command_list(std::string_view call_name) {
    TT_FATAL(t_recording == nullptr, "{} cannot run while a CommandListBuilder is recording on this thread", call_name);
}

void record_to_command_list(
    std::string_view operation_name,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload) {
    auto* builder = t_recording;
    TT_FATAL(builder != nullptr, "{}: no CommandListBuilder is recording on this thread", operation_name);
    TT_FATAL(
        &builder->device() == mesh_device,
        "{} runs on a different MeshDevice than the recording CommandListBuilder",
        operation_name);
    try {
        builder->record(ttnn::get_current_command_queue_id_for_thread(), workload);
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
    return CommandList(metal_builder_.build(device().mesh_command_queue(queue.get())));
}

void CommandListBuilder::clear() {
    TT_FATAL(!recording_, "Cannot clear a recording CommandListBuilder");
    metal_builder_.clear();
}

void CommandListBuilder::deallocate() {
    TT_FATAL(!recording_, "Cannot deallocate a recording CommandListBuilder");
    metal_builder_.deallocate();
}

tt::tt_metal::distributed::MeshDevice& CommandListBuilder::device() const { return metal_builder_.device(); }

void CommandListBuilder::record(QueueId binary_upload_cq_id, tt::tt_metal::distributed::MeshWorkload& workload) {
    metal_builder_.add(device().mesh_command_queue(binary_upload_cq_id.get()), workload);
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

CommandList::CommandList(metal::CommandList metal_list) : metal_list_(std::move(metal_list)) {}

void CommandList::enqueue(bool blocking, bool check_dram_overlap) {
    ZoneScoped;
    device_operation::detail::throw_if_recording_command_list("CommandList::enqueue");
    metal::EnqueueCommandList(
        device().mesh_command_queue(metal_list_.cq_id()), metal_list_, blocking, check_dram_overlap);
}

void CommandList::deallocate() { metal_list_.deallocate(); }

tt::tt_metal::distributed::MeshDevice& CommandList::device() const { return metal_list_.device(); }

QueueId CommandList::cq_id() const { return QueueId(metal_list_.cq_id()); }

}  // namespace ttnn::command_list
