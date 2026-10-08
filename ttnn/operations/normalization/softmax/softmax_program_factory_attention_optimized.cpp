#include "ttnn/operations/normalization/softmax/softmax_program_factory_attention_optimized.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::normalization::softmax {

Program create_program_attention_optimized(
    const Tensor& input_tensor,
    const Tensor& mask_tensor,
    float scale,
    const SoftmaxProgramConfig& program_config) {

    auto input_shape = input_tensor.get_shape();
    auto mask_shape = mask_tensor.get_shape();
    auto mask_layout = mask_tensor.get_layout();

    int mask_Ht;
    int mask_Wt;
    std::string mask_reader_kernel = "softmax_mask_reader";

    if (mask_layout == Layout::ROW_MAJOR) {
        // Documented unsharded ROW_MAJOR mask: [B,1,W/32,32]
        // Use the row-major reader that the sharded factory already uses.
        mask_Ht = 1;
        mask_Wt = static_cast<int>(mask_shape[2]);
        mask_reader_kernel = "softmax_mask_reader_row_major";
    } else {
        mask_Ht = static_cast<int>(mask_shape[2] / 32);
        mask_Wt = static_cast<int>(mask_shape[3] / 32);
        // Guard against W < 1024 where mask_Ht would be zero and cause
        // integer divide-by-zero in curr_ht % mask_Ht.
        if (mask_Ht == 0) {
            mask_Ht = 1;
        }
    }

    // ... rest of program creation uses mask_Ht, mask_Wt and mask_reader_kernel ...
    // curr_ht % mask_Ht is now safe and the correct reader is selected for ROW_MAJOR masks.

    Program program;
    // kernel creation using mask_reader_kernel
    return program;
}

} // namespace ttnn::operations::normalization::softmax
