// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/command_list.hpp"

#include <exception>

#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "ttnn/core.hpp"
#include "ttnn/device_operation_detail.hpp"

namespace {

thread_local ttnn::experimental::CommandListBuilder* t_recording = nullptr;

}  // namespace

namespace ttnn::device_operation::detail {

bool is_recording_command_list() { return t_recording != nullptr; }

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
        builder->record(workload);
    } catch (const std::exception& e) {
        TT_THROW("{} cannot be recorded into a command list: {}", operation_name, e.what());
    }
}

}  // namespace ttnn::device_operation::detail

namespace ttnn::experimental {

namespace metal = tt::tt_metal::experimental;

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

void CommandListBuilder::record(tt::tt_metal::distributed::MeshWorkload& workload) { metal_builder_.add(workload); }

CommandList::CommandList(metal::CommandList metal_list) : metal_list_(std::move(metal_list)) {}

void CommandList::enqueue(bool blocking, bool check_dram_overlap) {
    ZoneScoped;
    metal::EnqueueCommandList(
        device().mesh_command_queue(metal_list_.cq_id()), metal_list_, blocking, check_dram_overlap);
}

void CommandList::deallocate() { metal_list_.deallocate(); }

tt::tt_metal::distributed::MeshDevice& CommandList::device() const { return metal_list_.device(); }

QueueId CommandList::cq_id() const { return QueueId(metal_list_.cq_id()); }

}  // namespace ttnn::experimental
