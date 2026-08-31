// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>
#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/reports.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <any>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace ttnn::graph {

// Node identifiers in the graph
using node_id = int;

// One sub-device of a sub-device manager. Sub-device managers partition a MeshDevice uniformly, so
// this is a mesh-level fact and carries no physical device id.
struct SubDeviceTopology {
    uint8_t sub_device_id = 0;
    tt::tt_metal::CoreRangeSet worker_core_ranges;
};

// Where one program of a MeshWorkload actually ran. `device_id` is the MeshDevice id so it joins
// the report's `devices` table; `physical_device_id` is the chip the program landed on.
struct ProgramExecutionPlacement {
    uint32_t device_id = 0;
    uint32_t physical_device_id = 0;
    uint64_t sub_device_manager_id = 0;
    uint8_t sub_device_id = 0;
    tt::tt_metal::CoreRangeSet worker_core_ranges;
    uint64_t runtime_id = 0;
    uint32_t global_call_count = 0;
    uint64_t program_id = 0;
    uint8_t command_queue_id = 0;
};

// Records where `workload` just ran into every capture active on the calling thread. No-op when no
// capture is active. GraphTracker's processor stack is per-thread, so a workload enqueued on a
// thread that is not capturing is intentionally not observed.
void track_mesh_workload_execution(
    tt::tt_metal::distributed::MeshWorkload& workload,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint64_t runtime_id);

class ProcessorHooks : public tt::tt_metal::IGraphHooks {
private:
    bool do_block = false;

public:
    ProcessorHooks() = default;
    bool hook_allocate(const tt::tt_metal::Buffer* buffer) override;

    bool hook_deallocate(tt::tt_metal::Buffer* buffer) override;

    bool hook_program(tt::tt_metal::Program* program) override;

    bool hook_write_to_device(const tt::tt_metal::Buffer* buffer) override;

    bool hook_read_from_device(tt::tt_metal::Buffer* buffer) override;

    bool hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) override;

    bool hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) override;

    ~ProcessorHooks() override = default;

    void set_block(bool block);

    bool get_block() const;
};
class GraphProcessor : public tt::tt_metal::IGraphProcessor {
public:
    GraphProcessor(tt::tt_metal::IGraphProcessor::RunMode mode);
    ~GraphProcessor() override;

    void track_allocate(const tt::tt_metal::Buffer* buffer) override;

    void track_deallocate(tt::tt_metal::Buffer* buffer) override;

    void track_allocate_cb(
        const tt::tt_metal::CoreRangeSet& core_range,
        uint64_t addr,
        uint64_t size,
        bool is_globally_allocated,
        const tt::tt_metal::IDevice* device) override;

    void track_allocate_dataflow_buffer(
        const tt::tt_metal::CoreRangeSet& core_range,
        uint64_t addr,
        uint64_t size,
        bool borrows_memory,
        const tt::tt_metal::IDevice* device) override;

    void track_allocate_scratchpad(
        const tt::tt_metal::CoreRangeSet& core_range,
        uint64_t addr,
        uint64_t size,
        const tt::tt_metal::IDevice* device) override;

    void track_deallocate_cb(const tt::tt_metal::IDevice* device) override;

    void track_program(tt::tt_metal::Program* program, const tt::tt_metal::IDevice* device) override;

    void track_program_execution(const ProgramExecutionPlacement& placement);

    // True while this capture has not yet recorded the partition of this sub-device manager.
    // Lets the caller skip building the topology on the hot path once it has been captured.
    bool needs_sub_device_manager_snapshot(uint32_t device_id, uint64_t sub_device_manager_id);

    // Records a sub-device manager's full partition, so the report describes every sub-device and
    // not only those that happened to run an operation. Idempotent per (device, manager).
    void track_sub_device_manager(
        uint32_t device_id, uint64_t sub_device_manager_id, const std::vector<SubDeviceTopology>& sub_devices);

    void track_function_start(
        std::string_view function_name, std::span<tt::tt_metal::TrackedArgument> input_parameters) override;

    void track_function_end() override;
    void track_function_end(const std::any& output) override;

    void begin_capture(RunMode mode) override;

    nlohmann::json end_capture() override;

    struct Vertex {
        node_id counter = 0;
        std::string node_type;
        std::unordered_map<std::string, std::string> params;
        std::vector<std::string> arguments;
        std::vector<node_id> connections;
        std::vector<node_id> input_tensors;
        int stacking_level = 0;
        uint64_t duration_ns = 0;  // Duration in nanoseconds (for function_end nodes)
    };

    nlohmann::json get_report() const;

private:
    std::shared_ptr<ProcessorHooks> hook;

    std::mutex mutex;
    RunMode run_mode = RunMode::NORMAL;
    std::stack<node_id> current_op_id;
    std::unordered_map<std::int64_t, node_id> buffer_id_to_counter;
    node_id last_finished_op_id = -1;
    std::vector<Vertex> graph;
    std::vector<node_id> current_input_tensors;

    // Duration tracking - stack of start timestamps for nested operations
    using time_point = std::chrono::steady_clock::time_point;
    std::stack<time_point> function_start_times;

    // Capture timing
    time_point capture_start_time;
    uint64_t capture_start_timestamp_ns = 0;

    // Device info captured at track time (keyed by device_id)
    std::unordered_map<uint32_t, nlohmann::json> captured_device_info;
    // (device_id, sub_device_manager_id) pairs whose partition this capture has already recorded
    std::set<std::pair<uint32_t, uint64_t>> captured_sub_device_managers;
    // Device pointers for buffer pages (only valid during capture)
    std::vector<tt::tt_metal::distributed::MeshDevice*> captured_mesh_devices;
    // Per-operation buffer snapshots (function_start counter -> buffers)
    std::unordered_map<node_id, std::vector<ttnn::reports::BufferInfo>> per_op_buffers_;
    // Buffer pages keyed by address, with versioning for re-allocations.
    // Each address maps to a list of (allocation_counter, pages) pairs so that
    // re-allocations at the same address with different page configs are preserved.
    std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, std::vector<ttnn::reports::BufferPageInfo>>>>
        buffer_pages_by_address_;

    node_id add_tensor(const Tensor& t);
    node_id add_buffer(const tt::tt_metal::Buffer* buffer);

    void begin_function_process(const Tensor& tensor);

    void begin_function_process(const std::reference_wrapper<const Tensor>& tensor_ref);

    template <typename T>
    void begin_function_process(const std::optional<T>& tensor_opt);

    template <typename T>
    void begin_function_process(const std::vector<T>& tensor_vec);

    void end_function_process(const Tensor& tensor);

    template <typename T>
    void end_function_process(const std::optional<T>& tensor_opt);

    template <typename T>
    void end_function_process(const std::vector<T>& tensor_vec);

    void track_function_end_impl();

    void clean_hook();

    void track_device(const tt::tt_metal::IDevice* device);

public:
    static void begin_graph_capture(RunMode mode);
    static nlohmann::json end_graph_capture();

    static nlohmann::json end_graph_capture_to_file(const std::filesystem::path& report_path);

    static bool has_active_instance();
    static void set_pending_program_factory(std::string type, std::size_t index, bool cache_hit);

    // Detailed buffer tracing control
    static void enable_detailed_buffer_tracing();
    static void disable_detailed_buffer_tracing();
    static bool is_detailed_buffer_tracing_enabled();

private:
    struct PendingProgramFactory {
        std::string type;
        std::size_t index = 0;
        bool cache_hit = false;
    };
    static thread_local std::optional<PendingProgramFactory> pending_program_factory_;

    static std::atomic<bool> capture_detailed_buffer_tracing_;
};

/**
 * @class ScopedGraphCapture
 * @brief A RAII wrapper around graph capture that ensures proper resource management.
 *
 * This class automatically calls begin_graph_capture upon construction and
 * end_graph_capture when it goes out of scope. It can be ended regularly
 * by calling ScopedGraphCapture::end_graph_capture().
 *
 * @note Copy and move operations are deleted to prevent multiple instances
 * managing the same resource.
 */
class ScopedGraphCapture {
public:
    explicit ScopedGraphCapture(GraphProcessor::RunMode mode);

    ScopedGraphCapture(GraphProcessor::RunMode mode, std::filesystem::path report_path);

    ~ScopedGraphCapture();

    nlohmann::json end_graph_capture();

    nlohmann::json end_graph_capture_to_file(const std::filesystem::path& report_path);

    ScopedGraphCapture(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture(ScopedGraphCapture&&) = delete;
    ScopedGraphCapture& operator=(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture& operator=(ScopedGraphCapture&&) = delete;

private:
    bool is_active = false;
    std::filesystem::path auto_report_path;
};
}  // namespace ttnn::graph
