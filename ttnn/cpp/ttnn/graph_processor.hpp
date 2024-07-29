#pragma once

#include "tt_metal/graph_tracking.hpp"

#include <mutex>
#include <unordered_map>

namespace ttnn {
    class GraphProcessor : public tt::tt_metal::IGraphProcessor{

    public:
        GraphProcessor() = default;

        virtual void track_allocate(Buffer* buffer, bool bottom_up);

        virtual void track_deallocate(Buffer* buffer);

        virtual void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size);

        virtual void track_deallocate_cb();

        virtual void track_begin_op(std::string_view function_name, std::span<std::any> input_parameters);

        virtual void track_end_op(const std::any& output_tensors);

        virtual ~GraphProcessor() = default;

    private:
        std::mutex mutex;
        int counter = 0;
        int current_op_counter = -1;
        std::unordered_map<uint64_t, int> id_to_counter;
        std::unordered_map<uint64_t, int> counter_to_id;

        struct Vertex {
            int idx = 0;
            std::string name;
            uint32_t param = 0;
            std::vector<int> connections;
        };

        std::vector<Vertex> graph;

    };

    inline auto var = tt::tt_metal::GraphTracker::instance().add_processor(std::make_shared<GraphProcessor>());
}
