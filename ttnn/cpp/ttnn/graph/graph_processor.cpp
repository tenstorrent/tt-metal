// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_processor.hpp"
#include "graph_consts.hpp"
#include <tt-metalium/reflection.hpp>
#include "ttnn/types.hpp"
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program_impl.hpp>
#include "ttnn/graph/graph_consts.hpp"
#include <cxxabi.h>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include "ttnn/core.hpp"

using namespace tt::tt_metal;

namespace {
std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Layout& layout) {
    switch (layout) {
        case Layout::ROW_MAJOR:     return os << "Row Major";
        case Layout::TILE:          return os << "Tile";
        case Layout::INVALID:       return os << "Invalid";
        default:                    return os << "Unknown layout";
    }
}

std::ostream& operator<<(std::ostream& os, const Tile& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tt::stl::reflection::operator<<(os, tensor);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>& h) {
    return os << *h;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional_value) {
    if (optional_value.has_value()) {
        os << optional_value.value();
    } else {
        os << "nullopt";
    }
    return os;
}

std::string demangle(const char* name) {
    int status = -4;

    char* res = abi::__cxa_demangle(name, NULL, NULL, &status);

    const char* const demangled_name = (status == 0) ? res : name;

    std::string ret_val(demangled_name);

    free(res);

    return ret_val;
}

struct AnyToString {
    using ConvertionFunction = std::function<std::string(const std::any&)>;

    static std::unordered_map<std::type_index, ConvertionFunction>& registry() {
        static std::unordered_map<std::type_index, ConvertionFunction> map;
        return map;
    }

    template <typename T>
    static void register_type() {
        registry()[typeid(T)] = [](const std::any& value) -> std::string {
            std::ostringstream oss;
            auto reference_value = std::any_cast<T>(value);
            oss << reference_value.get();
            std::string result = oss.str();
            return result;
        };
    }

    static std::string to_string(const std::span<std::any>& span) {
        std::ostringstream oss;
        for (const auto& element : span) {
            if (!element.has_value()) {
                oss << "[any, empty],";
                continue;
            }

            auto it = registry().find(element.type());
            oss << "[ ";
            if (it != registry().end()) {
                 oss << it->second(element) << ", ";
            } else {
                oss << "unsupported type" << " , ";
            }

            oss << demangle(element.type().name());
            oss << "],";
        }

        std::string result = oss.str();
        if (!result.empty() && result.back() == ',')
        {
            result.pop_back(); // Remove last comma
        }

        return result;
    }
};

std::string tensorMemoryLayoutToString(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return "INTERLEAVED";
        case TensorMemoryLayout::SINGLE_BANK: return "SINGLE_BANK";
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

GraphProcessor::GraphProcessor(RunMode mode) : run_mode(mode) {
    begin_capture(mode);
    begin_function_any_map[typeid(std::reference_wrapper<std::vector<Tensor>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_vector(val); };
    begin_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_vector_optional(val); };
    begin_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_vector_optional_const(val); };
    begin_function_any_map[typeid(std::reference_wrapper<Tensor>)] = [ptr = this](const std::any& val) mutable {
        ptr->begin_function_process_ref_tensor(val);
    };
    begin_function_any_map[typeid(std::reference_wrapper<const Tensor>)] = [ptr = this](const std::any& val) mutable {
        ptr->begin_function_process_ref_const_tensor(val);
    };
    begin_function_any_map[typeid(std::reference_wrapper<std::optional<Tensor>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_optional_tensor(val); };
    begin_function_any_map[typeid(std::reference_wrapper<const std::optional<Tensor>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_optional_tensor_const(val); };
    begin_function_any_map[typeid(std::reference_wrapper<std::optional<const Tensor>>)] =
        [ptr = this](const std::any& val) mutable { ptr->begin_function_process_ref_optional_const_tensor(val); };

    end_function_any_map[typeid(std::reference_wrapper<std::vector<Tensor>>)] =
        [ptr = this](const std::any& val) mutable { ptr->end_function_process_vector(val); };
    end_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)] =
        [ptr = this](const std::any& val) mutable { ptr->end_function_process_vector_optional(val); };
    end_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)] =
        [ptr = this](const std::any& val) mutable { ptr->end_function_process_vector_optional_const(val); };
    end_function_any_map[typeid(std::reference_wrapper<Tensor>)] = [ptr = this](const std::any& val) mutable {
        ptr->end_function_process_tensor(val);
    };

    AnyToString::register_type<std::reference_wrapper<bool>>();
    AnyToString::register_type<std::reference_wrapper<bool const>>();
    AnyToString::register_type<std::reference_wrapper<int>>();
    AnyToString::register_type<std::reference_wrapper<int const>>();
    AnyToString::register_type<std::reference_wrapper<long>>();
    AnyToString::register_type<std::reference_wrapper<long const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<float> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::DataType> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::DataType>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::DataType const>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Layout> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Layout>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Layout const>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::MemoryConfig> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::MemoryConfig>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Shape>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Shape const>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Tensor> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Tile> const>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Tile>>>();
    AnyToString::register_type<std::reference_wrapper<std::optional<tt::tt_metal::Tile const>>>();
    AnyToString::register_type<std::reference_wrapper<tt::stl::SmallVector<long, 8ul>>>();
    AnyToString::register_type<std::reference_wrapper<tt::stl::SmallVector<long, 8ul> const>>();
    AnyToString::register_type<std::reference_wrapper<tt::stl::SmallVector<unsigned int, 8ul>>>();
    AnyToString::register_type<std::reference_wrapper<tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::DataType>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::DataType const>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Layout>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Layout const>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::MemoryConfig>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::MemoryConfig const>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Shape>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Shape const>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Tensor const>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Tile>>();
    AnyToString::register_type<std::reference_wrapper<tt::tt_metal::Tile const>>();
}

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
        graph.push_back(
            Vertex{.counter = counter, .node_type = kNodeBufferAllocate, .params = params, .arguments = {}, .connections = {buffer_id}});
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
            .counter = counter, .node_type = kNodeBufferDeallocate, .params = params, .arguments = {}, .connections = {buffer_id}});
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
        graph.push_back({.counter = counter, .node_type = kNodeCBAllocate, .params = params, .connections = {}});
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
            .arguments = {},
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

void GraphProcessor::track_function_start(std::string_view function_name, std::span<std::any> input_parameters) {
    const std::lock_guard<std::mutex> lock(mutex);
    tt::log_info("Begin op: {}", function_name);
    std::unordered_map<std::string, std::string> params = {
        {kInputs, std::to_string(input_parameters.size())},
        {kName, std::string(function_name)},
    };

    auto serialized_arguments = AnyToString::to_string(input_parameters);
    TT_ASSERT(serialized_arguments.size());

    auto counter = graph.size();
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeFunctionStart,
            .params = params,
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
        std::type_index any_type = any.type();
        auto it = begin_function_any_map.find(any_type);

        if (it != begin_function_any_map.end()) {
            it->second(any);
        } else {
            tt::log_info("input any type name ignored: {}", demangle(any.type().name()));
        }
    }
}

void GraphProcessor::track_function_end_impl() {
    auto name = graph[current_op_id.top()].params[kName];
    tt::log_info("End op: {}", name);

    auto counter = graph.size();
    {
        graph.push_back(
            Vertex{.counter = counter, .node_type = kNodeFunctionEnd, .params = {{kName, name}}, .arguments = {}, .connections = {}});
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
    const std::lock_guard<std::mutex> lock(mutex);
    this->track_function_end_impl();

    std::type_index any_type = output_tensors.type();
    auto it = end_function_any_map.find(any_type);

    if (it != end_function_any_map.end()) {
        it->second(output_tensors);
    } else {
        tt::log_info("output any type name ignored: {}", demangle(output_tensors.type().name()));
    }
    TT_ASSERT(current_op_id.size() > 0);  // we should always have capture_start on top
    current_op_id.pop();
}

int GraphProcessor::add_tensor(const Tensor& t) {
    auto& storage = t.get_storage();
    std::vector<tt::tt_metal::Buffer*> buffers = std::visit(
        [&t](auto&& storage) -> std::vector<tt::tt_metal::Buffer*> {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage> || std::is_same_v<T, MultiDeviceStorage>) {
                return t.buffers();
            }
            return {};
        },
        storage);
    std::int64_t tensor_id;
    if (not t.tensor_id.has_value()) {
        tt::log_warning(
            "Tensor doesn't have tensor_id, generating new one. Ideally this should not happen. Please set tensor_id "
            "for this tensor ahead of time.");
        tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
    } else {
        tensor_id = t.tensor_id.value();
    }
    auto tensor_counter = tensor_id_to_counter.count(tensor_id) > 0 ? tensor_id_to_counter[tensor_id] : graph.size();
    auto shape = t.get_logical_shape();

    std::unordered_map<std::string, std::string> params = {
        {kShape, fmt::format("{}", shape)},
        {kTensorId, fmt::format("{}", tensor_id)},
    };

    if (tensor_id_to_counter.count(tensor_id) == 0) {
        graph.push_back(
            Vertex{.counter = tensor_counter, .node_type = kNodeTensor, .params = params, .arguments = {}, .connections = {}});
        tensor_id_to_counter[tensor_id] = tensor_counter;
    }

    if (buffers.empty()) {
        tt::log_info(
            "Tensor doesn't have buffer, but storage is {}", demangle(get_type_in_var(t.get_storage()).name()));
    }

    for (auto& buffer : buffers) {
        auto buffer_id = add_buffer(buffer);
        graph[buffer_id].connections.push_back(tensor_counter);
    }

    return tensor_counter;
}

int GraphProcessor::add_buffer(const tt::tt_metal::Buffer* buffer) {
    auto buffer_id = buffer->unique_id();
    auto counter = buffer_id_to_counter.count(buffer_id) > 0 ? buffer_id_to_counter[buffer_id] : graph.size();
    if (buffer_id_to_counter.count(buffer_id) == 0) {
        std::unordered_map<std::string, std::string> params = {
            {kSize, std::to_string(buffer->size())},
            {kType, buffer->is_dram() ? "DRAM" : "L1"},
            {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
            {kDeviceId, std::to_string(buffer->device()->id())}};

        graph.push_back(Vertex{.counter = counter, .node_type = kNodeBuffer, .params = params, .arguments = {}, .connections = {}});
        graph[current_op_id.top()].connections.push_back(counter);
        buffer_id_to_counter[buffer_id] = counter;
        return counter;
    }
    return buffer_id_to_counter[buffer_id];
}

void GraphProcessor::begin_function_process_ref_vector(const std::any& any_val) {
    const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<Tensor>>>(any_val).get();
    for (auto& it : tensor_vec) {
        int tensor_id = add_tensor(it);
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
}
void GraphProcessor::begin_function_process_ref_vector_optional(const std::any& any_val) {
    const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<Tensor>>>>(any_val).get();
    for (auto& it : tensor_vec) {
        if (it.has_value()) {
            int tensor_id = add_tensor(it.value());
            graph[tensor_id].connections.push_back(current_op_id.top());
        }
    }
}
void GraphProcessor::begin_function_process_ref_vector_optional_const(const std::any& any_val) {
    const auto& tensor_vec =
        std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any_val).get();
    for (auto& it : tensor_vec) {
        if (it.has_value()) {
            int tensor_id = add_tensor(it.value());
            graph[tensor_id].connections.push_back(current_op_id.top());
        }
    }
}
void GraphProcessor::begin_function_process_ref_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<Tensor>>(any_val).get();
    int tensor_id = add_tensor(tensor);
    graph[tensor_id].connections.push_back(current_op_id.top());
}
void GraphProcessor::begin_function_process_ref_const_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<const Tensor>>(any_val).get();
    int tensor_id = add_tensor(tensor);
    graph[tensor_id].connections.push_back(current_op_id.top());
}
void GraphProcessor::begin_function_process_ref_optional_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor>>>(any_val).get();
    if (tensor.has_value()) {
        int tensor_id = add_tensor(tensor.value());
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
}
void GraphProcessor::begin_function_process_ref_optional_tensor_const(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<const std::optional<Tensor>>>(any_val).get();
    if (tensor.has_value()) {
        int tensor_id = add_tensor(tensor.value());
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
}
void GraphProcessor::begin_function_process_ref_optional_const_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<const Tensor>>>(any_val).get();
    if (tensor.has_value()) {
        int tensor_id = add_tensor(tensor.value());
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
}
void GraphProcessor::end_function_process_vector(const std::any& any_val) {
    const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<Tensor>>>(any_val).get();
    for (auto& it : tensor_vec) {
        int tensor_id = add_tensor(it);
        graph[last_finished_op_id].connections.push_back(tensor_id);
    }
}
void GraphProcessor::end_function_process_vector_optional(const std::any& any_val) {
    const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<Tensor>>>>(any_val).get();
    for (auto& it : tensor_vec) {
        if (it.has_value()) {
            int tensor_id = add_tensor(it.value());
            graph[last_finished_op_id].connections.push_back(tensor_id);
        }
    }
}
void GraphProcessor::end_function_process_vector_optional_const(const std::any& any_val) {
    const auto& tensor_vec =
        std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any_val).get();
    for (auto& it : tensor_vec) {
        if (it.has_value()) {
            int tensor_id = add_tensor(it.value());
            graph[last_finished_op_id].connections.push_back(tensor_id);
        }
    }
}
void GraphProcessor::end_function_process_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<Tensor>>(any_val).get();
    int tensor_id = add_tensor(tensor);
    graph[last_finished_op_id].connections.push_back(tensor_id);
}
void GraphProcessor::end_function_process_optional_tensor(const std::any& any_val) {
    const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor>>>(any_val).get();
    if (tensor.has_value()) {
        int tensor_id = add_tensor(tensor.value());
        graph[last_finished_op_id].connections.push_back(tensor_id);
    }
}

void GraphProcessor::begin_capture(RunMode mode) {
    const std::lock_guard<std::mutex> lock(mutex);
    graph.clear();
    buffer_id_to_counter.clear();
    tensor_id_to_counter.clear();
    graph.push_back(Vertex{.counter = 0, .node_type = kNodeCaptureStart, .params = {}, .arguments = {}, .connections = {}});

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
    graph.push_back(Vertex{.counter = counter, .node_type = kNodeCaptureEnd, .params = {}, .arguments = {}, .connections = {}});
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

bool ProcessorHooks::hook_program(tt::tt_metal::Program*) { return do_block; }

void ProcessorHooks::set_block(bool block) { do_block = block; }
bool ProcessorHooks::get_block() const { return do_block; }

ScopedGraphCapture::ScopedGraphCapture(GraphProcessor::RunMode mode) {
    GraphProcessor::begin_graph_capture(mode);
    is_active = true;
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
