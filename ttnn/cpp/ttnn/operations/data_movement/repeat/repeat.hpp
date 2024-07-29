#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct Repeat {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value, mem_config);
        return output_tensor;
    }
};


}  // namespace operations::data_movement

constexpr auto repeat = ttnn::register_operation_with_auto_launch_op<"ttnn::repeat", ttnn::operations::data_movement::Repeat>();

} // namespace ttnn
