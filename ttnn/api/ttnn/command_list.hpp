// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/command_list.hpp>

#include "ttnn/command_list_params.hpp"
#include "ttnn/common/guard.hpp"
#include "ttnn/common/queue_id.hpp"

// Experimental and subject to change.

namespace ttnn::device_operation::detail {
struct CmdListProgramBindings;
}  // namespace ttnn::device_operation::detail

namespace ttnn::command_list {

class CommandList;

namespace detail {

// One encoding of a scalar TTNN param, written to the kernel arguments recorded under its Metal name.
struct CmdListMetalScalar {
    std::string metal_cmd_list_param_name;  // "<TTNN param>/<factory type>/<Metal param>"
    CmdListEncodeFn encode;                 // from the first TTNN record of the Metal name
    bool has_runtime = false;
    bool has_common_runtime = false;
};

// What TTNN update_args writes for one TTNN param: its tensor, or each of its encodings.
struct CmdListParamLayout {
    bool is_tensor = false;
    std::vector<CmdListMetalScalar> metal_scalars;
};

using CmdListLayout = std::unordered_map<std::string, CmdListParamLayout>;

}  // namespace detail

// Records TTNN device operations into a command list instead of launching them.
// Only Metal 2.0 (ProgramSpec) programs can be recorded. Nothing needs to run before recording: recording compiles
// each workload and uploads its kernel binaries.
// Not movable: the recording thread holds a pointer to it.
class CommandListBuilder {
public:
    // Acquires Metal's builder reservation: one builder per MeshDevice.
    explicit CommandListBuilder(tt::tt_metal::distributed::MeshDevice& mesh_device);
    ~CommandListBuilder();

    CommandListBuilder(const CommandListBuilder&) = delete;
    CommandListBuilder& operator=(const CommandListBuilder&) = delete;
    CommandListBuilder(CommandListBuilder&&) = delete;
    CommandListBuilder& operator=(CommandListBuilder&&) = delete;

    // Device operations launched on the calling thread are recorded instead of enqueued until end_recording().
    // One recording builder per thread; one recording thread per builder.
    void begin_recording();
    void end_recording();
    bool is_recording() const;

    // begin_recording(), with end_recording() when the returned guard goes out of scope.
    [[nodiscard]] ScopeGuard record_scope();

    // Builds an independent command list bound to cq_id. Defaults to the calling thread's current command queue.
    // Repeatable; built lists outlive the builder.
    CommandList build(std::optional<QueueId> cq_id = std::nullopt) const;

    // Drops everything recorded so far. Not allowed while recording.
    void clear();

    // Releases the Metal reservation and invalidates the builder. Not allowed while recording.
    void deallocate();

    tt::tt_metal::distributed::MeshDevice& device() const;

    // Adds one workload to the recording. binary_upload_cq_id is the command queue that writes its kernel binaries to
    // the device. Also records as TTNN command-list parameters the pushed binds that match `programs`: the
    // workload's programs with their stored CmdListBindings. Commits nothing if the workload cannot be recorded.
    void record(
        QueueId binary_upload_cq_id,
        tt::tt_metal::distributed::MeshWorkload& workload,
        std::span<const device_operation::detail::CmdListProgramBindings> programs);

    // The TTNN command-list parameter binds of the op calls in progress while recording, for push_cmd_list_binds and
    // pop_cmd_list_binds. Scopes nest; pop_binds(true) throws if a bind of the popped scope reached no recorded
    // program.
    void push_binds(std::string op_name, std::vector<CmdListBind> binds);
    void pop_binds(bool validate);
    bool has_binds() const;

private:
    struct ActiveBind {
        CmdListBind bind;
        bool claimed = false;
    };

    struct BindScope {
        std::string op_name;
        std::size_t start;  // index into binds_ where the scope starts
    };

    tt::tt_metal::experimental::CommandListBuilder metal_builder_;
    bool recording_ = false;
    std::vector<BindScope> bind_scopes_;
    std::vector<ActiveBind> binds_;  // innermost scope last
    detail::CmdListLayout layout_;
};

// A replayable command list in device DRAM, bound to the command queue it was built for.
// Lists record raw device addresses, not tensors. Replay reads and writes those addresses whether or not the tensors
// are still allocated; what lives there is the caller's responsibility.
class CommandList {
public:
    CommandList(CommandList&&) noexcept = default;
    CommandList& operator=(CommandList&&) noexcept = default;

    // Replays on the bound command queue.
    void enqueue(bool blocking = true, bool check_dram_overlap = false);

    // Sets the values later replays use for the named TTNN command-list parameters. Unknown names, kind mismatches, and
    // values an encoding rejects throw before anything is written. blocking waits for the write to finish.
    void update_args(const std::unordered_map<std::string, CmdListValue>& values, bool blocking = false);

    // Releases device resources and invalidates the list. Repeated calls have no effect.
    void deallocate();

    tt::tt_metal::distributed::MeshDevice& device() const;
    QueueId cq_id() const;

private:
    friend class CommandListBuilder;
    CommandList(tt::tt_metal::experimental::CommandList metal_list, detail::CmdListLayout layout);

    tt::tt_metal::experimental::CommandList metal_list_;
    detail::CmdListLayout layout_;
};

}  // namespace ttnn::command_list
