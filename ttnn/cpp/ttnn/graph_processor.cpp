#include "graph_processor.hpp"

#include "third_party/json/json.hpp"
#include "types.hpp"


#include <cxxabi.h>

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
                .connections = {current_op_id.top(), buf_id}
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
                .connections = {current_op_id.top(), id_to_counter[alloc_id]}
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
                .connections = {current_op_id.top()}
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
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = std::format("begin: {}", function_name),
                .param = input_parameters.size(),
                .connections = {/*current_op_id.top()*/}
            });
            graph[current_op_id.top()].connections.push_back(counter);
            current_op_id.push(counter);
        }

        for (int i = 0; auto& any : input_parameters) {
            //tt::log_info("any type name param[{}]: {}", i, demangle(any.type().name()));
            if (any.type() == typeid(std::vector<Tensor>)) {
                const auto& tensor_vec = std::any_cast<std::vector<Tensor>>(any);
                for (int j = 0; auto& it : tensor_vec) {
                    add_tensor(it, std::format("input[{}]/tensor[{}]", i, j));
                    j++;
                }
            }

            if (any.type() == typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)) {
                const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<Tensor>>>>(any).get();
                for (int j = 0; auto& it : tensor_vec) {
                    if (it.has_value()) {
                        add_tensor(it.value(), std::format("input[{}]/optional_tensor[{}]", i, j));
                    }
                    j++;
                }
            }

            if (any.type() == typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)) {
                const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(any).get();
                for (int j = 0; auto& it : tensor_vec) {
                    if (it.has_value()) {
                        add_tensor(it.value(), std::format("input[{}]/optional_const_tensor[{}]", i, j));
                    }
                    j++;
                }
            }

            if (any.type() == typeid(std::reference_wrapper<Tensor>)) {
                const auto& tensor = std::any_cast<std::reference_wrapper<Tensor>>(any).get();
                add_tensor(tensor, std::format("input[{}]/tensor", i));
            }

            if (any.type() == typeid(std::reference_wrapper<std::optional<Tensor>>)) {
                const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor>>>(any).get();
                if (tensor.has_value()) {
                    add_tensor(tensor.value(), std::format("input[{}]/tensor", i));
                }

            }

            i++;
        }

    }

    void GraphProcessor::track_end_op(const std::any& output_tensors) {
        const std::lock_guard<std::mutex> lock(mutex);
        tt::log_info("any type name: {}", demangle(output_tensors.type().name()));
        if (output_tensors.type() == typeid(std::vector<Tensor>)) {
            tt::log_info("Casting types: {} == {}", output_tensors.type().name(), typeid(std::vector<Tensor>).name());
            const auto& tensor_vec = std::any_cast<std::vector<Tensor>>(output_tensors);
            for (int j = 0; auto& it : tensor_vec) {
                add_tensor(it, std::format("output/tensor[{}]", j));
                j++;
            }
        }

        if (output_tensors.type() == typeid(std::reference_wrapper<std::vector<std::optional<Tensor>>>)) {
            const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<Tensor>>>>(output_tensors).get();
            for (int j = 0; auto& it : tensor_vec) {
                if (it.has_value()) {
                    add_tensor(it.value(), std::format("output/optional_tensor[{}]", j));
                }
                j++;
            }
        }

        if (output_tensors.type() == typeid(std::reference_wrapper<std::vector<std::optional<const Tensor>>>)) {
            const auto& tensor_vec = std::any_cast<std::reference_wrapper<std::vector<std::optional<const Tensor>>>>(output_tensors).get();
            for (int j = 0; auto& it : tensor_vec) {
                if (it.has_value()) {
                    add_tensor(it.value(), std::format("output/optional_const_tensor[{}]", j));
                }
                j++;
            }
        }

        if (output_tensors.type() == typeid(std::reference_wrapper<Tensor>)) {
            const auto& tensor = std::any_cast<std::reference_wrapper<Tensor>>(output_tensors).get();
            add_tensor(tensor, std::format("output/tensor"));
        }

        if (output_tensors.type() == typeid(std::reference_wrapper<std::optional<Tensor>>)) {
            const auto& tensor = std::any_cast<std::reference_wrapper<std::optional<Tensor>>>(output_tensors).get();
            if (tensor.has_value()) {
                add_tensor(tensor.value(), std::format("output/tensor"));
            }
        }
        auto counter = graph.size();
        {
            graph.push_back(Vertex{
                .counter = counter,
                .name = std::format("end op"),
                .param = 0,
                .connections = {current_op_id.top()}
            });
            graph[current_op_id.top()].connections.push_back(counter);
        }
        current_op_id.pop();
    }

    int GraphProcessor::add_tensor(const Tensor& t, string_view name) {
        auto alloc_id = reinterpret_cast<std::uintptr_t>(t.tensor_attributes.get());
        auto tensor_counter = id_to_counter.count(alloc_id) > 0 ? id_to_counter[alloc_id] : graph.size();
        if (id_to_counter.count(alloc_id) == 0) {
            graph.push_back(Vertex{
                .counter = tensor_counter,
                .name = std::string(name),
                .param = 0,
                .connections = {current_op_id.top()}
            });
            graph[current_op_id.top()].connections.push_back(tensor_counter);
            id_to_counter[alloc_id] = tensor_counter;
        }
        auto buffer = t.buffer();
        if (buffer) {
            auto buffer_idx = add_buffer(buffer);
            graph[tensor_counter].connections.push_back(buffer_idx);
            graph[buffer_idx].connections.push_back(tensor_counter);
        }
        return id_to_counter[alloc_id];
    }

    int GraphProcessor::add_buffer(tt::tt_metal::Buffer* buffer) {
        auto buffer_alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
        auto counter = graph.size();
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

GraphProcessor::~GraphProcessor() {
    auto json_object = tt::stl::json::to_json(graph);
    std::ofstream output_file_stream("test_graph.json");
    output_file_stream << json_object << std::endl;
}
}
