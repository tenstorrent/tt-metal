// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>
#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor.hpp"

#include <chrono>
#include <filesystem>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <any>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::graph {

// Node identifiers in the graph
using node_id = int;

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
        const CoreRangeSet& core_range,
        uint64_t addr,
        uint64_t size,
        bool is_globally_allocated,
        const tt::tt_metal::IDevice* device) override;

    void track_deallocate_cb(const tt::tt_metal::IDevice* device) override;

    void track_program(tt::tt_metal::Program* program, const tt::tt_metal::IDevice* device) override;

    void track_function_start(std::string_view function_name, std::span<std::any> input_parameters) override;

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
        uint64_t duration_ns = 0;              // Duration in nanoseconds (for function_end nodes)
        std::vector<std::string> stack_trace;  // Optional stack trace (when enabled)
    };

    void write_report(const std::filesystem::path& report_path) const;

    nlohmann::json get_report() const;

private:
    std::shared_ptr<ProcessorHooks> hook;

    std::mutex mutex;
    RunMode run_mode = RunMode::NORMAL;
    std::stack<node_id> current_op_id;
    std::unordered_map<std::int64_t, node_id> buffer_id_to_counter;
    std::unordered_map<std::int64_t, node_id> tensor_id_to_counter;
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
    // Device pointers for buffer pages (only valid during capture)
    std::vector<tt::tt_metal::distributed::MeshDevice*> captured_mesh_devices;

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

    static nlohmann::json get_current_report();

    // Track an error that occurred during graph capture
    static void track_error(
        const std::string& error_type, const std::string& error_message, const std::string& operation_name = "");

    // Stack trace capture control
    static void enable_stack_traces();
    static void disable_stack_traces();
    static bool is_stack_trace_enabled();

    // Detailed buffer page capture control
    static void enable_buffer_pages();
    static void disable_buffer_pages();
    static bool is_buffer_pages_enabled();

private:
    static bool capture_stack_traces_;
    static bool capture_buffer_pages_;
    static std::vector<std::string> capture_stack_trace();
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

    nlohmann::json get_report() const;

    ScopedGraphCapture(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture(ScopedGraphCapture&&) = delete;
    ScopedGraphCapture& operator=(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture& operator=(ScopedGraphCapture&&) = delete;

private:
    bool is_active = false;
    std::filesystem::path auto_report_path;
};
}  // namespace ttnn::graph
