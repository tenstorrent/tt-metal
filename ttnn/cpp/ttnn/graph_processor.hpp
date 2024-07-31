#pragma once

#include "tt_metal/graph_tracking.hpp"

#include <mutex>
#include <stack>
#include <typeindex>
#include <unordered_map>
#include <functional>
#include <any>
namespace ttnn {
    class GraphProcessor : public tt::tt_metal::IGraphProcessor{

    public:
        GraphProcessor();

        virtual void track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up);

        virtual void track_deallocate(tt::tt_metal::Buffer* buffer);

        virtual void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size);

        virtual void track_deallocate_cb();

        virtual void track_begin_op(std::string_view function_name, std::span<std::any> input_parameters);

        virtual void track_end_op(const std::any& output_tensors);

        virtual ~GraphProcessor();

        struct Vertex {
            int counter = 0;
            std::string name;
            uint64_t param = 0;
            std::vector<int> connections;
        };
        using ProcessFunc = std::function<void(const std::any&)>;
    private:
        std::mutex mutex;
        std::stack<int> current_op_id;
        std::unordered_map<uint64_t, int> id_to_counter;
        int last_finished_op_id = -1;
        int tensors_used = 0;
        std::vector<Vertex> graph;
        std::unordered_map<std::type_index, ProcessFunc> begin_op_any_map;
        std::unordered_map<std::type_index, ProcessFunc> end_op_any_map;
        int add_tensor(const Tensor& t);
        int add_buffer(tt::tt_metal::Buffer* buffer);

        void begin_op_process_ref_vector(const std::any& any_val);
        void begin_op_process_ref_vector_optional(const std::any& any_val);
        void begin_op_process_ref_vector_optional_const(const std::any& any_val);
        void begin_op_process_ref_tensor(const std::any& any_val);
        void begin_op_process_ref_const_tensor(const std::any& any_val);
        void begin_op_process_ref_optional_tensor(const std::any& any_val);
        void begin_op_process_ref_optional_tensor_const(const std::any& any_val);
        void begin_op_process_ref_optional_const_tensor(const std::any& any_val);

        void end_op_process_vector(const std::any& any_val);
        void end_op_process_vector_optional(const std::any& any_val);
        void end_op_process_vector_optional_const(const std::any& any_val);
        void end_op_process_tensor(const std::any& any_val);
        void end_op_process_optional_tensor(const std::any& any_val);

    };

    inline auto var = tt::tt_metal::GraphTracker::instance().add_processor(std::make_shared<GraphProcessor>());
}
