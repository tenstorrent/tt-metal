// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_argument_serializer.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/types.hpp"
#include "ttnn/core.hpp"
#include <cxxabi.h>
#include <memory>
#include <string>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/reflection.hpp>
#include <unordered_map>

using namespace tt::tt_metal;

namespace {
std::string tensorMemoryLayoutToString(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return "INTERLEAVED";
        case TensorMemoryLayout::HEIGHT_SHARDED: return "HEIGHT_SHARDED";
        case TensorMemoryLayout::WIDTH_SHARDED: return "WIDTH_SHARDED";
        case TensorMemoryLayout::BLOCK_SHARDED: return "BLOCK_SHARDED";
        default: return "UNKNOWN";  // Handle unexpected values
    }
}

template <class Variant>
const std::type_info& get_type_in_var(const Variant& v) {
    return std::visit([](auto&& x) -> decltype(auto) { return typeid(x); }, v);
}

nlohmann::json to_json(const ttnn::graph::GraphProcessor::Vertex& data) {
    nlohmann::json j;
    j[ttnn::graph::kCounter] = data.counter;
    j[ttnn::graph::kNodeType] = data.node_type;
    j[ttnn::graph::kParams] = data.params;
    j[ttnn::graph::kArguments] = data.arguments;
    j[ttnn::graph::kConnections] = data.connections;
    return j;
}

nlohmann::json to_json(const std::vector<ttnn::graph::GraphProcessor::Vertex>& data) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& item : data) {
        j.push_back(to_json(item));
    }
    return j;
}

}  // namespace

namespace ttnn::graph {

GraphProcessor::GraphProcessor(RunMode mode) : run_mode(mode) { begin_capture(mode); }

void GraphProcessor::track_allocate(const tt::tt_metal::Buffer* buffer) {
    const std::lock_guard<std::mutex> lock(mutex);
    auto buffer_id = add_buffer(buffer);

    auto counter = graph.size();

    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kAddress, std::to_string(buffer->address())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kPageSize, std::to_string(buffer->page_size())},
        {kNumCores, std::to_string(buffer->num_cores().value_or(0))},  // use 0 for interleaved
        {kDeviceId, std::to_string(buffer->device()->id())}};
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeBufferAllocate,
            .params = std::move(params),
            .connections = {buffer_id}});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_deallocate(tt::tt_metal::Buffer* buffer) {
    const std::lock_guard<std::mutex> lock(mutex);
    auto buffer_id = add_buffer(buffer);
    auto counter = graph.size();
    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kPageSize, std::to_string(buffer->page_size())},
        {kNumCores, std::to_string(buffer->num_cores().value_or(0))},  // use 0 for interleaved
        {kDeviceId, std::to_string(buffer->device()->id())}};
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeBufferDeallocate,
            .params = std::move(params),
            .connections = {buffer_id}});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_allocate_cb(
    const CoreRangeSet& core_range_set,
    uint64_t addr,
    uint64_t size,
    bool is_globally_allocated,
    const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);
    const std::lock_guard<std::mutex> lock(mutex);
    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(size)},
        {kAddress, std::to_string(addr)},
        {kCoreRangeSet, core_range_set.str()},
        {kGloballyAllocated, std::to_string(is_globally_allocated)},
        {kDeviceId, std::to_string(device->id())}};
    auto counter = graph.size();
    {
        graph.push_back(
            {.counter = counter, .node_type = kNodeCBAllocate, .params = std::move(params), .connections = {}});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_deallocate_cb(const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);
    const std::lock_guard<std::mutex> lock(mutex);
    auto counter = graph.size();
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeCBDeallocateAll,
            .params = {{kDeviceId, std::to_string(device->id())}},
            .connections = {current_op_id.top()}});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_program(tt::tt_metal::Program* program, const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);

    // All previous CBs are deallocated before a new program run
    track_deallocate_cb(device);

    if (run_mode == RunMode::NORMAL) {
        // we will track real buffer allocations during program run
        return;
    }

    for (auto& cb : program->circular_buffers()) {
        track_allocate_cb(cb->core_ranges(), 0, cb->size(), cb->globally_allocated(), device);
    }
}

template <typename T>
using ProcessFunc = void (GraphProcessor::*)(const T&);

template <typename T, ProcessFunc<T> Process>
static void process(GraphProcessor& self, const std::any& any_val) {
    (self.*Process)(std::any_cast<std::reference_wrapper<T>>(any_val).get());
}

template <typename T, ProcessFunc<T> Process>
consteval std::pair<const std::type_info&, void (*)(GraphProcessor&, const std::any&)> make_process() {
    return {typeid(std::reference_wrapper<T>), &process<T, Process>};
}

void GraphProcessor::track_function_start(std::string_view function_name, std::span<std::any> input_parameters) {
    static constexpr std::array begin_function_any_map = {
        make_process<std::vector<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<std::vector<std::optional<Tensor>>, &GraphProcessor::begin_function_process>(),
        make_process<std::vector<std::optional<const Tensor>>, &GraphProcessor::begin_function_process>(),
        make_process<Tensor, &GraphProcessor::begin_function_process>(),
        make_process<const Tensor, &GraphProcessor::begin_function_process>(),
        make_process<std::optional<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<const std::optional<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<std::optional<const Tensor>, &GraphProcessor::begin_function_process>(),
    };

    const std::lock_guard<std::mutex> lock(mutex);
    log_debug(tt::LogAlways, "Begin op: {}", function_name);
    std::unordered_map<std::string, std::string> params = {
        {kInputs, std::to_string(input_parameters.size())},
        {kName, std::string(function_name)},
    };

    std::vector<std::string> serialized_arguments;
    serialized_arguments = GraphArgumentSerializer::instance().to_list(input_parameters);

    auto counter = graph.size();
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeFunctionStart,
            .params = std::move(params),
            .arguments = serialized_arguments,
            .connections = {/*current_op_id.top()*/}});
        if (last_finished_op_id != -1) {
            graph[last_finished_op_id].connections.push_back(counter);
            last_finished_op_id = -1;
        }
        graph[current_op_id.top()].connections.push_back(counter);
        current_op_id.push(counter);
    }

    for (auto& any : input_parameters) {
        const auto it = std::ranges::find(
            begin_function_any_map, any.type(), [](const auto& pair) -> const auto& { return pair.first; });

        if (it != begin_function_any_map.end()) {
            it->second(*this, any);
        } else {
            log_debug(tt::LogAlways, "input any type name ignored: {}", graph_demangle(any.type().name()));
        }
    }
}

void GraphProcessor::track_function_end_impl() {
    auto name = graph[current_op_id.top()].params[kName];
    log_debug(tt::LogAlways, "End op: {}", name);

    auto counter = graph.size();
    {
        graph.push_back(
            Vertex{.counter = counter, .node_type = kNodeFunctionEnd, .params = {{kName, name}}, .connections = {}});
        graph[current_op_id.top()].connections.push_back(counter);
    }
    last_finished_op_id = counter;
}

void GraphProcessor::track_function_end() {
    const std::lock_guard<std::mutex> lock(mutex);
    this->track_function_end_impl();
    TT_ASSERT(current_op_id.size() > 0);  // we should always have capture_start on top
    current_op_id.pop();
}

void GraphProcessor::track_function_end(const std::any& output_tensors) {
    static constexpr std::array end_function_any_map{
        make_process<std::vector<Tensor>, &GraphProcessor::end_function_process>(),
        make_process<std::vector<std::optional<Tensor>>, &GraphProcessor::end_function_process>(),
        make_process<std::vector<std::optional<const Tensor>>, &GraphProcessor::end_function_process>(),
        make_process<Tensor, &GraphProcessor::end_function_process>(),
    };

    const std::lock_guard<std::mutex> lock(mutex);
    this->track_function_end_impl();

    const auto it = std::ranges::find(
        end_function_any_map, output_tensors.type(), [](const auto& pair) -> const auto& { return pair.first; });

    if (it != end_function_any_map.end()) {
        it->second(*this, output_tensors);
    } else {
        log_debug(tt::LogAlways, "output any type name ignored: {}", graph_demangle(output_tensors.type().name()));
    }
    TT_ASSERT(current_op_id.size() > 0);  // we should always have capture_start on top
    current_op_id.pop();
}

int GraphProcessor::add_tensor(const Tensor& t) {
    auto& storage = t.storage();
    tt::tt_metal::Buffer* buffer = std::visit(
        [&t]<typename T>(const T& storage) -> tt::tt_metal::Buffer* {
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (storage.mesh_buffer) {
                    // `t.buffers()` returns a reference buffer allocated on first device in a mesh.
                    // It has an ID different from the "backing" buffer that was used to perform the allocation.
                    // To deduplicate an entry for this buffer, captured during its allocation, use the "backing"
                    // buffer.
                    return storage.mesh_buffer->get_backing_buffer();
                } else {
                    return t.buffer();
                }
            }
            return nullptr;
        },
        storage);
    std::int64_t tensor_id;
    if (not t.tensor_id.has_value()) {
        log_warning(
            tt::LogAlways,
            "Tensor doesn't have tensor_id, generating new one. Ideally this should not happen. Please set tensor_id "
            "for this tensor ahead of time.");
        tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
    } else {
        tensor_id = t.tensor_id.value();
    }
    auto tensor_counter = tensor_id_to_counter.count(tensor_id) > 0 ? tensor_id_to_counter[tensor_id] : graph.size();
    auto shape = t.logical_shape();

    std::unordered_map<std::string, std::string> params = {
        {kShape, fmt::format("{}", shape)},
        {kTensorId, fmt::format("{}", tensor_id)},
    };

    if (tensor_id_to_counter.count(tensor_id) == 0) {
        graph.push_back(Vertex{
            .counter = tensor_counter, .node_type = kNodeTensor, .params = std::move(params), .connections = {}});
        tensor_id_to_counter[tensor_id] = tensor_counter;
    }

    if (buffer == nullptr) {
        log_debug(
            tt::LogAlways,
            "Tensor doesn't have buffer, but storage is {}",
            graph_demangle(get_type_in_var(t.storage()).name()));
    } else {
        auto buffer_id = add_buffer(buffer);
        graph[buffer_id].connections.push_back(tensor_counter);
    }

    return tensor_counter;
}

int GraphProcessor::add_buffer(const tt::tt_metal::Buffer* buffer) {
    const auto buffer_id = buffer->unique_id();

    if (const auto it = buffer_id_to_counter.find(buffer_id); it != buffer_id_to_counter.end()) {
        return it->second;
    }

    const auto counter = graph.size();
    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kDeviceId, std::to_string(buffer->device()->id())}};

    graph.push_back(
        Vertex{.counter = counter, .node_type = kNodeBuffer, .params = std::move(params), .connections = {}});
    graph[current_op_id.top()].connections.push_back(counter);
    buffer_id_to_counter.emplace(buffer_id, counter);
    return counter;
}

void GraphProcessor::begin_function_process(const Tensor& tensor) {
    int tensor_id = add_tensor(tensor);
    graph[tensor_id].connections.push_back(current_op_id.top());
}

template <typename T>
void GraphProcessor::begin_function_process(const std::optional<T>& tensor_opt) {
    if (tensor_opt.has_value()) {
        begin_function_process(*tensor_opt);
    }
}

template <typename T>
void GraphProcessor::begin_function_process(const std::vector<T>& tensor_vec) {
    for (auto& it : tensor_vec) {
        begin_function_process(it);
    }
}

void GraphProcessor::end_function_process(const Tensor& tensor) {
    int tensor_id = add_tensor(tensor);
    graph[last_finished_op_id].connections.push_back(tensor_id);
}

template <typename T>
void GraphProcessor::end_function_process(const std::optional<T>& tensor_opt) {
    if (tensor_opt.has_value()) {
        end_function_process(*tensor_opt);
    }
}

template <typename T>
void GraphProcessor::end_function_process(const std::vector<T>& tensor_vec) {
    for (auto& it : tensor_vec) {
        end_function_process(it);
    }
}

void GraphProcessor::begin_capture(RunMode mode) {
    const std::lock_guard<std::mutex> lock(mutex);
    graph.clear();
    buffer_id_to_counter.clear();
    tensor_id_to_counter.clear();
    graph.push_back(Vertex{.counter = 0, .node_type = kNodeCaptureStart, .params = {}, .connections = {}});

    if (!tt::tt_metal::GraphTracker::instance().get_hook()) {
        hook = std::make_shared<ProcessorHooks>();
        tt::tt_metal::GraphTracker::instance().add_hook(hook);
        hook->set_block(mode == RunMode::NO_DISPATCH);
    }
    current_op_id.push(0);
}
nlohmann::json GraphProcessor::end_capture() {
    const std::lock_guard<std::mutex> lock(mutex);
    int counter = graph.size();
    graph.push_back(Vertex{.counter = counter, .node_type = kNodeCaptureEnd, .params = {}, .connections = {}});
    if (last_finished_op_id != -1) {
        graph[last_finished_op_id].connections.push_back(counter);
    } else {
        // lets connect capture_start with capture_end
        // it means we didn't capture any functions
        TT_ASSERT(
            current_op_id.size(),
            "Graph size cannot be 0. This means that track_function_end was called more than begin.");
        graph[0].connections.push_back(counter);
    }
    clean_hook();
    return to_json(graph);
}

void GraphProcessor::clean_hook() {
    if (hook) {
        // If we installed hooks then we must clean
        hook = nullptr;
        tt::tt_metal::GraphTracker::instance().clear_hook();
    }
}

GraphProcessor::~GraphProcessor() { clean_hook(); }

void GraphProcessor::begin_graph_capture(RunMode mode = RunMode::NORMAL) {
    tt::tt_metal::GraphTracker::instance().push_processor(std::make_shared<GraphProcessor>(mode));
}
nlohmann::json GraphProcessor::end_graph_capture() {
    auto res = tt::tt_metal::GraphTracker::instance().get_processors().back()->end_capture();
    tt::tt_metal::GraphTracker::instance().pop_processor();
    return res;
}

bool ProcessorHooks::hook_allocate(const tt::tt_metal::Buffer* buffer) { return do_block; }

bool ProcessorHooks::hook_deallocate(tt::tt_metal::Buffer* buffer) { return do_block; }

bool ProcessorHooks::hook_write_to_device(const tt::tt_metal::Buffer* buffer) { return do_block; }

bool ProcessorHooks::hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) { return do_block; }

bool ProcessorHooks::hook_read_from_device(tt::tt_metal::Buffer* buffer) { return do_block; }

bool ProcessorHooks::hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) {
    return do_block;
}

bool ProcessorHooks::hook_program(tt::tt_metal::Program*) { return do_block; }

void ProcessorHooks::set_block(bool block) { do_block = block; }
bool ProcessorHooks::get_block() const { return do_block; }

ScopedGraphCapture::ScopedGraphCapture(GraphProcessor::RunMode mode) : is_active(true) {
    GraphProcessor::begin_graph_capture(mode);
}
ScopedGraphCapture::~ScopedGraphCapture() {
    if (is_active) {
        GraphProcessor::end_graph_capture();
    }
}
nlohmann::json ScopedGraphCapture::end_graph_capture() {
    is_active = false;
    return GraphProcessor::end_graph_capture();
}

}  // namespace ttnn::graph
