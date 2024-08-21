// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_processor.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "types.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/program/program.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cxxabi.h>
#include <memory>
#include <typeindex>
#include <unordered_map>

namespace {
std::string demangle(const char* name) {

    int status = -4;

    char* res = abi::__cxa_demangle(name, NULL, NULL, &status);

    const char* const demangled_name = (status==0)?res:name;

    std::string ret_val(demangled_name);

    free(res);

    return ret_val;
}

std::string tensorMemoryLayoutToString(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED:
            return "INTERLEAVED";
        case TensorMemoryLayout::SINGLE_BANK:
            return "SINGLE_BANK";
        case TensorMemoryLayout::HEIGHT_SHARDED:
            return "HEIGHT_SHARDED";
        case TensorMemoryLayout::WIDTH_SHARDED:
            return "WIDTH_SHARDED";
        case TensorMemoryLayout::BLOCK_SHARDED:
            return "BLOCK_SHARDED";
        default:
            return "UNKNOWN"; // Handle unexpected values
    }
}

template<class Variant>
std::type_info const& get_type_in_var(const Variant& v){
    return std::visit( [](auto&&x)->decltype(auto){ return typeid(x); }, v );
}

nlohmann::json to_json(const ttnn::GraphProcessor::Vertex& data) {
    nlohmann::json j;
    j["counter"] = data.counter;
    j["name"] = data.name;
    j["params"] = data.params;
    j["connections"] = data.connections;
    return j;
}

nlohmann::json to_json(const std::vector<ttnn::GraphProcessor::Vertex>& data) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& item : data) {
        j.push_back(to_json(item));
    }
    return j;
}

}

namespace ttnn {
    GraphProcessor::GraphProcessor(RunMode mode) : run_mode(mode) {
        begin_capture(mode);
        begin_function_any_map[typeid(std::reference_wrapper<std::vector<Tensor>>)] = [ptr = this]  (const std::any& val) mutable {ptr->begin_function_process_ref_vector(val);};
        begin_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_vector_optional(val);};
        begin_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_vector_optional_const(val);};
        begin_function_any_map[typeid(std::reference_wrapper<Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_tensor(val);};
        begin_function_any_map[typeid(std::reference_wrapper<const Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_const_tensor(val);};
        begin_function_any_map[typeid(std::reference_wrapper<std::optional<Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_optional_tensor(val);};
        begin_function_any_map[typeid(std::reference_wrapper<std::optional<Tensor> const>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_optional_tensor_const(val);};
        begin_function_any_map[typeid(std::reference_wrapper<std::optional<const Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_function_process_ref_optional_const_tensor(val);};

        end_function_any_map[typeid(std::reference_wrapper<std::vector<Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->end_function_process_vector(val);};
        end_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->end_function_process_vector_optional(val);};
        end_function_any_map[typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->end_function_process_vector_optional_const(val);};
        end_function_any_map[typeid(std::reference_wrapper<Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->end_function_process_tensor(val);};

    }
    void GraphProcessor::track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto buf_id = add_buffer(buffer);

        auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = graph.size();

        std::unordered_map<std::string, std::string> params = {
                {"size", std::to_string(buffer->size())},
                {"address", std::to_string(buffer->address())},
                {"type", buffer->is_dram() ? "DRAM" : "L1"},
                {"layout", tensorMemoryLayoutToString(buffer->buffer_layout())}
        };
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer_allocate",
                .params = params,
                .connections = {buf_id}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
    }

    void GraphProcessor::track_deallocate(tt::tt_metal::Buffer* buffer) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto counter = graph.size();
        auto buffer_idx = add_buffer(buffer);
        std::unordered_map<std::string, std::string> params = {
                {"size", std::to_string(buffer->size())},
                {"type", buffer->is_dram() ? "DRAM" : "L1"},
                {"layout", tensorMemoryLayoutToString(buffer->buffer_layout())}
        };
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer_deallocate",
                .params = params,
                .connections = {buffer_idx}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }

    }

    void GraphProcessor::track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size) {
        const std::lock_guard<std::mutex> lock(mutex);
        std::unordered_map<std::string, std::string> params = {
            {"size", std::to_string(size)},
            {"address", std::to_string(addr)},
            {"core_range_set", core_range_set.str()}
        };
        auto counter = graph.size();
        {
            graph.push_back({
                .counter = counter,
                .name = "circular_buffer_allocate",
                .params = params,
                .connections = {}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }

    }

    void GraphProcessor::track_deallocate_cb() {
        const std::lock_guard<std::mutex> lock(mutex);
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "circular_buffer_deallocate_all",
                .params = {},
                .connections = {current_op_id.top()}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
    }

    void GraphProcessor::track_program(tt::tt_metal::Program* program) {
        if (run_mode == RunMode::NORMAL) {
            // we will track real buffer allocations during program run
            return;
        }
        for (auto& cb : program->circular_buffers()) {
            track_allocate_cb(cb->core_ranges(), 0, cb->size());
        }
    }

    void GraphProcessor::track_begin_function(std::string_view function_name, std::span<std::any> input_parameters) {
        const std::lock_guard<std::mutex> lock(mutex);
        tt::log_info("Begin op: {}", function_name);
        std::unordered_map<std::string, std::string> params = {
            {"inputs", std::to_string(input_parameters.size())},
            {"name", std::string(function_name)},
        };
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "begin_function",
                .params = params,
                .connections = {/*current_op_id.top()*/}
            });
            if ( last_finished_op_id != -1 ) {
                graph[last_finished_op_id].connections.push_back(counter);
                last_finished_op_id = -1;
            }
            graph[current_op_id.top()].connections.push_back(counter);
            current_op_id.push(counter);

        }

        for (int i = 0; auto& any : input_parameters) {
            std::type_index any_type = any.type();
            auto it = begin_function_any_map.find(any_type);

            if (it != begin_function_any_map.end()) {
                it->second(any);
            } else {
                tt::log_info("input any type name ignored: {}", demangle(any.type().name()));
            }
            i++;
        }

    }

    void GraphProcessor::track_end_function_impl() {
        auto name = graph[current_op_id.top()].params["name"];
        tt::log_info("End op: {}", name);

        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = fmt::format("end_function"),
                .params = {{"name", name}},
                .connections = {}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
        last_finished_op_id = counter;
    }

    void GraphProcessor::track_end_function() {
        const std::lock_guard<std::mutex> lock(mutex);
        this->track_end_function_impl();
        current_op_id.pop();
        TT_ASSERT(current_op_id.size() > 0); // we should always have capture_start on top
    }

    void GraphProcessor::track_end_function(const std::any& output_tensors) {
        const std::lock_guard<std::mutex> lock(mutex);
        this->track_end_function_impl();

        std::type_index any_type = output_tensors.type();
        auto it = end_function_any_map.find(any_type);

        if (it != end_function_any_map.end()) {
            it->second(output_tensors);
        } else {
            tt::log_info("output any type name ignored: {}", demangle(output_tensors.type().name()));
        }
        current_op_id.pop();
    }

    int GraphProcessor::add_tensor(const Tensor& t) {
        const uint64_t pointer_shift = (uint64_t)1 << 32; // to avoid reusing the same alloc_id
        auto& storage = t.get_storage();
        auto buffer = std::visit(
        [&t](auto&& storage) -> tt::tt_metal::Buffer* {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                return t.buffer();
            } else {
                return nullptr;
            }
        },
        storage);
        auto alloc_id = buffer ? reinterpret_cast<std::uintptr_t>(buffer) + pointer_shift : reinterpret_cast<std::uintptr_t>(t.tensor_attributes.get());
        tt::log_info("Tensor ID: {}, used: {}", alloc_id, tensors_used);
        auto tensor_counter = id_to_counter.count(alloc_id) > 0 ? id_to_counter[alloc_id] : graph.size();
        auto shape = t.get_shape();
        std::ostringstream oss;
        oss << shape;
        std::string shape_str = oss.str();
        std::unordered_map<std::string, std::string> params = {
            {"shape", shape_str},
        };
        if (id_to_counter.count(alloc_id) == 0) {
            graph.push_back(Vertex{
                .counter = tensor_counter,
                .name = fmt::format("tensor[{}]", tensors_used),
                .params = params,
                .connections = {}
            });
            tensors_used++;
            id_to_counter[alloc_id] = tensor_counter;
        }

        if (buffer) {
            auto buffer_idx = add_buffer(buffer);
            graph[buffer_idx].connections.push_back(tensor_counter);
        } else {
            tt::log_info("Tensor doesn't have buffer, but storage is {}", demangle(get_type_in_var(t.get_storage()).name()));
        }
        return tensor_counter;
    }

    int GraphProcessor::add_buffer(tt::tt_metal::Buffer* buffer) {
        auto buffer_alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = id_to_counter.count(buffer_alloc_id) > 0 ? id_to_counter[buffer_alloc_id] : graph.size();
        if (id_to_counter.count(buffer_alloc_id) == 0) {
            std::unordered_map<std::string, std::string> params = {
                {"size", std::to_string(buffer->size())},
                {"type", buffer->is_dram() ? "DRAM" : "L1"},
                {"layout", tensorMemoryLayoutToString(buffer->buffer_layout())}
            };

            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer",
                .params = params,
                .connections = {}
            });
            graph[current_op_id.top()].connections.push_back(counter);
            id_to_counter[buffer_alloc_id] = counter;
            return counter;
        }
        return id_to_counter[buffer_alloc_id];
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
        const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any_val).get();
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
        const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor> const>>(any_val).get();
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
        const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any_val).get();
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
        id_to_counter.clear();
        graph.push_back(Vertex{
            .counter = 0,
            .name = "capture_start",
            .params = {},
            .connections = {}
        });

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
        graph.push_back(Vertex{
            .counter = counter,
            .name = "capture_end",
            .params = {},
            .connections = {}
        });
        if ( last_finished_op_id != -1 ) {
            graph[last_finished_op_id].connections.push_back(counter);
        } else {
            // lets connect capture_start with capture_end
            // it means we didn't capture any functions
            TT_ASSERT(current_op_id.size(), "Graph size cannot be 0. This means that track_end_function was called more than begin.");
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

    GraphProcessor::~GraphProcessor() {
        clean_hook();
    }

    void GraphProcessor::begin_graph_capture(RunMode mode = RunMode::NORMAL) {
        tt::tt_metal::GraphTracker::instance().push_processor(std::make_shared<GraphProcessor>(mode));

    }
    nlohmann::json GraphProcessor::end_graph_capture() {
         auto res = tt::tt_metal::GraphTracker::instance().get_processors().back()->end_capture();
         tt::tt_metal::GraphTracker::instance().pop_processor();
         return res;
    }

    bool ProcessorHooks::hook_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {
        return do_block;
    }

    bool ProcessorHooks::hook_deallocate(tt::tt_metal::Buffer* buffer) {
        return do_block;
    }

    bool ProcessorHooks::hook_program(tt::tt_metal::Program*) {
        return do_block;
    }

    void ProcessorHooks::set_block(bool block) {
        do_block = block;
    }
    bool ProcessorHooks::get_block() const {
        return do_block;
    }
}
