#include "ttnn/operations/normalization/softmax/softmax_device_operation.hpp"
#include "ttnn/operations/normalization/softmax/softmax_program_factory.hpp"
#include "ttnn/operations/normalization/softmax/softmax_program_factory_attention_optimized.hpp"
#include "ttnn/operations/normalization/softmax/softmax_program_factory_attention_optimized_sharded.hpp"
#include "ttnn/operations/normalization/softmax/softmax_program_factory.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::normalization {

Tensor scale_mask_softmax_in_place(
    const Tensor& input,
    float scale,
    const Tensor& mask,
    const MemoryConfig& memory_config,
    const std::optional<SoftmaxProgramConfig>& program_config) {
    auto mask_tensor = mask;
    // The non-in-place path tilizes ROW_MAJOR masks before the kernel.
    // The in-place path must do the same to keep the interleaved
    // attention-optimized factory on the TILE mask path.
    if (mask_tensor.get_layout() == Layout::ROW_MAJOR) {
        // Documented unsharded ROW_MAJOR form is [B,1,W/32,32]
        // Tilize to the TILE layout expected by the attention-optimized kernels.
        mask_tensor = ttnn::tilize(mask_tensor, /*...*/);
    }

    auto output = ttnn::operations::normalization::softmax::scale_mask_softmax_device_operation(
        input, scale, mask_tensor, memory_config, program_config);

    return output;
}

Tensor scale_mask_softmax(
    const Tensor& input,
    float scale,
    const Tensor& mask,
    const MemoryConfig& memory_config,
    const std::optional<SoftmaxProgramConfig>& program_config) {
    auto mask_tensor = mask;
    if (mask_tensor.get_layout() == Layout::ROW_MAJOR) {
        mask_tensor = ttnn::tilize(mask_tensor, /*...*/);
    }
    return ttnn::operations::normalization::softmax::scale_mask_softmax_device_operation(
        input, scale, mask_tensor, memory_config, program_config);
}

} // namespace ttnn::operations::normalization
