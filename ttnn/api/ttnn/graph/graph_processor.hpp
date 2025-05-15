// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>
#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor.hpp"

#include <mutex>
#include <stack>
#include <typeindex>
#include <unordered_map>
#include <functional>
#include <any>
namespace ttnn::graph {

class ProcessorHooks : public tt::tt_metal::IGraphHooks {
private:
    bool do_block = false;

public:
    ProcessorHooks() = default;
    bool hook_allocate(const tt::tt_metal::Buffer* buffer) override;

    bool hook_deallocate(tt::tt_metal::Buffer* buffer) override;

    bool hook_program(tt::tt_metal::Program* program) override;

    virtual ~ProcessorHooks() = default;

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

    void track_function_start(std::string_view function_name, std::span<std::any> args) override;

    void track_function_end() override;
    void track_function_end(const std::any& output) override;

    void begin_capture(RunMode mode) override;

    nlohmann::json end_capture() override;

    struct Vertex {
        int counter = 0;
        std::string node_type;
        std::unordered_map<std::string, std::string> params;
        std::vector<std::string> arguments;
        std::vector<int> connections;
    };
    using ProcessFunc = std::function<void(const std::any&)>;

private:
    std::shared_ptr<ProcessorHooks> hook;

    std::mutex mutex;
    RunMode run_mode = RunMode::NORMAL;
    std::stack<int> current_op_id;
    std::unordered_map<std::int64_t, int> buffer_id_to_counter;
    std::unordered_map<std::int64_t, int> tensor_id_to_counter;
    int last_finished_op_id = -1;
    std::vector<Vertex> graph;
    std::unordered_map<std::type_index, ProcessFunc> begin_function_any_map;
    std::unordered_map<std::type_index, ProcessFunc> end_function_any_map;

    int add_tensor(const Tensor& t);
    int add_buffer(const tt::tt_metal::Buffer* buffer);

    void begin_function_process_ref_vector(const std::any& any_val);
    void begin_function_process_ref_vector_optional(const std::any& any_val);
    void begin_function_process_ref_vector_optional_const(const std::any& any_val);
    void begin_function_process_ref_tensor(const std::any& any_val);
    void begin_function_process_ref_const_tensor(const std::any& any_val);
    void begin_function_process_ref_optional_tensor(const std::any& any_val);
    void begin_function_process_ref_optional_tensor_const(const std::any& any_val);
    void begin_function_process_ref_optional_const_tensor(const std::any& any_val);

    void end_function_process_vector(const std::any& any_val);
    void end_function_process_vector_optional(const std::any& any_val);
    void end_function_process_vector_optional_const(const std::any& any_val);
    void end_function_process_tensor(const std::any& any_val);
    void end_function_process_optional_tensor(const std::any& any_val);

    void track_function_end_impl();

    void clean_hook();

public:
    static void begin_graph_capture(RunMode mode);
    static nlohmann::json end_graph_capture();
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
    ScopedGraphCapture(GraphProcessor::RunMode mode);
    ~ScopedGraphCapture();
    nlohmann::json end_graph_capture();

    ScopedGraphCapture(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture(ScopedGraphCapture&&) = delete;
    ScopedGraphCapture& operator=(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture& operator=(ScopedGraphCapture&&) = delete;

private:
    bool is_active = false;
};
}  // namespace ttnn::graph
