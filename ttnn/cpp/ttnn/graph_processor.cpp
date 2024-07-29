#include "graph_processor.hpp"

#include "types.hpp"


namespace ttnn {

    void GraphProcessor::track_allocate(Buffer* buffer, bool bottom_up) {
        const std::lock_guard<std::mutex> lock(mutex);
    }

    void GraphProcessor::track_deallocate(Buffer* buffer) {
        const std::lock_guard<std::mutex> lock(mutex);


    }

    void GraphProcessor::track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {
        const std::lock_guard<std::mutex> lock(mutex);

    }

    void GraphProcessor::track_deallocate_cb() {
        const std::lock_guard<std::mutex> lock(mutex);

    }

    void GraphProcessor::track_begin_op(std::string_view function_name, std::span<std::any> input_parameters) {
        const std::lock_guard<std::mutex> lock(mutex);

    }

    void GraphProcessor::track_end_op(const std::any& output_tensors) {
        const std::lock_guard<std::mutex> lock(mutex);

    };
}
