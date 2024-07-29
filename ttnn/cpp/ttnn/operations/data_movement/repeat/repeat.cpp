#include "repeat.hpp"
#include "device/repeat_op.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor Repeat::operator()(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape, std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [shape, output_mem_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            uint32_t input_rank = input_tensor.get_legacy_shape().rank();
            TT_FATAL(shape.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
            Tensor output = input_tensor;
            for (uint32_t dim = 0; dim < shape.rank(); ++dim) {
                if (shape[dim] == 1) {
                    continue;
                }
                TT_FATAL(shape[dim] > 0, "Number of repetitions along a dim must be greater than 0");
                if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
                    TT_FATAL(
                        (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() == 0,
                        "Current repeat implementation requires aligned last dim when repeating on last dim");
                }
                output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output}).at(0);
            }
            return {output};
        }, {input_tensor}, output_tensors);
    return output_tensors.at(0);
}

}
