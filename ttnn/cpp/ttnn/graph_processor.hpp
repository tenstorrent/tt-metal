#pragma once

#include "tt_metal/graph_tracking.hpp"

#include <mutex>
#include <stack>
#include <typeindex>
#include <unordered_map>
#include <functional>
#include <any>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ttnn {

    class ProcessorHooks : public tt::tt_metal::IGraphHooks {
    private:
        bool do_block = false;
    public:
        ProcessorHooks() = default;
        virtual bool hook_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) override;

        virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) override;

        virtual bool block_run_program() override;

        virtual ~ProcessorHooks() = default;

        void set_block(bool block) {
            do_block = block;
        }
        bool get_block() const {
            return do_block;
        }
    };
    class GraphProcessor : public tt::tt_metal::IGraphProcessor{

    public:
        GraphProcessor();

        void track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) override;

        void track_deallocate(tt::tt_metal::Buffer* buffer) override;

        void track_allocate_cb(const CoreRangeSet &core_range, uint64_t addr, uint64_t size) override;

        void track_deallocate_cb() override;

        void track_begin_op(std::string_view function_name, std::span<std::any> input_parameters) override;

        void track_end_op(const std::any& output_tensors) override;

        void begin_capture() override;
        std::string end_capture() override;

        ~GraphProcessor() override;

        struct Vertex {
            int counter = 0;
            std::string name;
            uint64_t param = 0;
            std::vector<int> connections;
        };
        using ProcessFunc = std::function<void(const std::any&)>;
    private:
        std::shared_ptr<ProcessorHooks> hooks;

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

    public:
        static void begin_graph_capture();
        static std::string end_graph_capture();
    };

    namespace py = pybind11;
    inline void py_graph_module(py::module& m) {
        auto doc_begin =
            R"doc(begin_graph_capture()
        )doc";
        auto doc_end =
            R"doc(end_graph_capture() -> string
            returns json string.
        )doc";

        m.def("begin_graph_capture", &GraphProcessor::begin_graph_capture, doc_begin);
        m.def("end_graph_capture", &GraphProcessor::end_graph_capture, doc_end);
    }

}
