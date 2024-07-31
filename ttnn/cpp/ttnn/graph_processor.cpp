#include "graph_processor.hpp"

#include "third_party/json/json.hpp"
#include "types.hpp"


#include <cxxabi.h>
#include <typeindex>

const string demangle(const char* name) {

    int status = -4;

    char* res = abi::__cxa_demangle(name, NULL, NULL, &status);

    const char* const demangled_name = (status==0)?res:name;

    string ret_val(demangled_name);

    free(res);

    return ret_val;
}

namespace tt::stl::json {

/*
            int counter = 0;
            std::string name;
            uint64_t param = 0;
            std::vector<int> connections;
*/

template <>
struct to_json_t<ttnn::GraphProcessor::Vertex> {
    nlohmann::json operator()(const ttnn::GraphProcessor::Vertex& v) noexcept {
        return {
            {"counter", to_json(v.counter)},
            {"name", to_json(v.name)},
            {"param", to_json(v.param)},
            {"connections", to_json(v.connections)},

        };
    }
};

}
namespace ttnn {
    GraphProcessor::GraphProcessor() {
        graph.push_back(Vertex{
            .counter = 0,
            .name = "application_start",
            .param = 0,
            .connections = {}
        });
        current_op_id.push(0);
        begin_op_any_map[typeid(std::reference_wrapper<std::vector<Tensor>>)] = [ptr = this]  (const std::any& val) mutable {ptr->begin_op_process_ref_vector(val);};
        begin_op_any_map[typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_vector_optional(val);};
        begin_op_any_map[typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_vector_optional_const(val);};
        begin_op_any_map[typeid(std::reference_wrapper<Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_tensor(val);};
        begin_op_any_map[typeid(std::reference_wrapper<const Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_const_tensor(val);};
        begin_op_any_map[typeid(std::reference_wrapper<std::optional<Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_optional_tensor(val);};
        begin_op_any_map[typeid(std::reference_wrapper<std::optional<Tensor> const>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_optional_tensor_const(val);};
        begin_op_any_map[typeid(std::reference_wrapper<std::optional<const Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->begin_op_process_ref_optional_const_tensor(val);};

        end_op_any_map[typeid(std::vector<Tensor>)] = [ptr = this] (const std::any& val) mutable {ptr->end_op_process_vector(val);};
        end_op_any_map[typeid(std::vector<std::optional<Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->end_op_process_vector_optional(val);};
        end_op_any_map[typeid(std::vector<std::optional<const Tensor>>)] = [ptr = this] (const std::any& val) mutable {ptr->end_op_process_vector_optional_const(val);};
        end_op_any_map[typeid(Tensor)] = [ptr = this] (const std::any& val) mutable {ptr->end_op_process_tensor(val);};

    }
    void GraphProcessor::track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto buf_id = add_buffer(buffer);

        auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = graph.size();

        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer_allocate",
                .param = buffer->size(),
                .connections = {buf_id}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
    }

    void GraphProcessor::track_deallocate(tt::tt_metal::Buffer* buffer) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = graph.size();
        TT_ASSERT(id_to_counter.count(alloc_id));
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer_deallocate",
                .param = buffer->size(),
                .connections = {id_to_counter[alloc_id]}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }

    }

    void GraphProcessor::track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto counter = graph.size();
        {
            graph.push_back({
                .counter = counter,
                .name = "circular_buffer_allocate",
                .param = size,
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
                .param = 0,
                .connections = {current_op_id.top()}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
    }

    void GraphProcessor::track_begin_op(std::string_view function_name, std::span<std::any> input_parameters) {
        const std::lock_guard<std::mutex> lock(mutex);
        tt::log_info("Begin op: {}", function_name);
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = std::format("begin: {}", function_name),
                .param = input_parameters.size(),
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
            auto it = begin_op_any_map.find(any_type);

            if (it != begin_op_any_map.end()) {
                it->second(any);
            } else {
                tt::log_info("input any type name ignored: {}", demangle(any.type().name()));
            }
            i++;
        }

    }

    void GraphProcessor::track_end_op(const std::any& output_tensors) {
        const std::lock_guard<std::mutex> lock(mutex);
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = std::format("end op"),
                .param = 0,
                .connections = {}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
        last_finished_op_id = counter;

        std::type_index any_type = output_tensors.type();
        auto it = end_op_any_map.find(any_type);

        if (it != end_op_any_map.end()) {
            it->second(output_tensors);
        } else {
            tt::log_info("output any type name ignored: {}", demangle(output_tensors.type().name()));
        }
        current_op_id.pop();
    }

    int GraphProcessor::add_tensor(const Tensor& t) {
        auto alloc_id = reinterpret_cast<std::uintptr_t>(t.tensor_attributes.get());
        tt::log_info("Tensor ID: {}, used: {}", alloc_id, tensors_used);
        auto tensor_counter = id_to_counter.count(alloc_id) > 0 ? id_to_counter[alloc_id] : graph.size();
        if (id_to_counter.count(alloc_id) == 0) {
            graph.push_back(Vertex{
                .counter = tensor_counter,
                .name = std::format("tensor[{}]", tensors_used),
                .param = 0,
                .connections = {}
            });
            tensors_used++;
            id_to_counter[alloc_id] = tensor_counter;
        }
        auto buffer = std::visit(
        [&t](auto&& storage) -> tt::tt_metal::Buffer* {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                return t.buffer();
            } else {
                return nullptr;
            }
        },
        t.get_storage());

        if (buffer) {
            auto buffer_idx = add_buffer(buffer);
            graph[buffer_idx].connections.push_back(tensor_counter);
        }
        return tensor_counter;
    }

    int GraphProcessor::add_buffer(tt::tt_metal::Buffer* buffer) {
        auto buffer_alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = id_to_counter.count(buffer_alloc_id) > 0 ? id_to_counter[buffer_alloc_id] : graph.size();
        if (id_to_counter.count(buffer_alloc_id) == 0) {
            graph.push_back(Vertex{
                .counter = counter,
                .name = "buffer",
                .param = 0,
                .connections = {}
            });
            graph[current_op_id.top()].connections.push_back(counter);
            id_to_counter[buffer_alloc_id] = counter;
            return counter;
        }
        return id_to_counter[buffer_alloc_id];
    }


    void GraphProcessor::begin_op_process_ref_vector(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<Tensor>>>(any_val).get();
        for (int j = 0; auto& it : tensor_vec) {
            int tensor_id = add_tensor(it);
            graph[tensor_id].connections.push_back(current_op_id.top());
            j++;
        }
    }
    void GraphProcessor::begin_op_process_ref_vector_optional(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<Tensor>>>>(any_val).get();
        for (int j = 0; auto& it : tensor_vec) {
            if (it.has_value()) {
                int tensor_id = add_tensor(it.value());
                graph[tensor_id].connections.push_back(current_op_id.top());
            }
            j++;
        }
    }
    void GraphProcessor::begin_op_process_ref_vector_optional_const(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any_val).get();
        for (int j = 0; auto& it : tensor_vec) {
            if (it.has_value()) {
                int tensor_id = add_tensor(it.value());
                graph[tensor_id].connections.push_back(current_op_id.top());
            }
            j++;
        }
    }
    void GraphProcessor::begin_op_process_ref_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::reference_wrapper<Tensor>>(any_val).get();
        int tensor_id = add_tensor(tensor);
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
    void GraphProcessor::begin_op_process_ref_const_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::reference_wrapper<const Tensor>>(any_val).get();
        int tensor_id = add_tensor(tensor);
        graph[tensor_id].connections.push_back(current_op_id.top());
    }
    void GraphProcessor::begin_op_process_ref_optional_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor>>>(any_val).get();
        if (tensor.has_value()) {
            int tensor_id = add_tensor(tensor.value());
            graph[tensor_id].connections.push_back(current_op_id.top());
        }
    }
    void GraphProcessor::begin_op_process_ref_optional_tensor_const(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor> const>>(any_val).get();
        if (tensor.has_value()) {
            int tensor_id = add_tensor(tensor.value());
            graph[tensor_id].connections.push_back(current_op_id.top());
        }
    }
    void GraphProcessor::begin_op_process_ref_optional_const_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<const Tensor>>>(any_val).get();
        if (tensor.has_value()) {
            int tensor_id = add_tensor(tensor.value());
            graph[tensor_id].connections.push_back(current_op_id.top());
        }
    }
    void GraphProcessor::end_op_process_vector(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::vector<Tensor>>(any_val);
        for (int j = 0; auto& it : tensor_vec) {
            int tensor_id = add_tensor(it);
            graph[last_finished_op_id].connections.push_back(tensor_id);
            j++;
        }
    }
    void GraphProcessor::end_op_process_vector_optional(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::vector<std::optional<Tensor>>>(any_val);
        for (int j = 0; auto& it : tensor_vec) {
            if (it.has_value()) {
                int tensor_id = add_tensor(it.value());
                graph[last_finished_op_id].connections.push_back(tensor_id);
                j++;
            }
        }
    }
    void GraphProcessor::end_op_process_vector_optional_const(const std::any& any_val) {
        const auto& tensor_vec = std::any_cast<std::vector<std::optional<const Tensor>>>(any_val);
        for (int j = 0; auto& it : tensor_vec) {
            if (it.has_value()) {
                int tensor_id = add_tensor(it.value());
                graph[last_finished_op_id].connections.push_back(tensor_id);
            }
        }
    }
    void GraphProcessor::end_op_process_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<Tensor>(any_val);
        int tensor_id = add_tensor(tensor);
        graph[last_finished_op_id].connections.push_back(tensor_id);
    }
    void GraphProcessor::end_op_process_optional_tensor(const std::any& any_val) {
        const auto& tensor = std::any_cast<std::optional<Tensor>>(any_val);
        if (tensor.has_value()) {
            int tensor_id = add_tensor(tensor.value());
            graph[last_finished_op_id].connections.push_back(tensor_id);
        }
    }

    GraphProcessor::~GraphProcessor() {
        auto json_object = tt::stl::json::to_json(graph);
        std::ofstream output_file_stream("test_graph.json");
        output_file_stream << json_object << std::endl;
    }
}
