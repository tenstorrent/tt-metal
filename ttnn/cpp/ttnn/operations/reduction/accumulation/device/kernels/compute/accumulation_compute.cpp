#include "ttnn/operations/experimental/cumsum/device/kernels/compute/accumulation_compute.hpp"

#include "ttnn/operations/experimental/cumsum/device/kernels/compute/accumulation_compute.hpp"

namespace ttnn::operations::experimental::cumsum {

void accumulation_compute(
    const Tensor& input,
    Tensor& output,
    const std::vector<uint32_t>& scan_dimensions,
    const bool disable_compensated_sum
) {
    // ... existing code ...

    // Compensated accumulation for FP32
    if (input.get_dtype() == DataType::FLOAT32 && !disable_compensated_sum) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < num_elements; ++i) {
            float y = input_data[i];
            float t = acc + y;
            float c = 0.0f;
            if (std::isfinite(t)) {
                c = (t - acc) - y;
            }
            acc = t + c;
            output_data[i] = acc;
        }
    } else {
        // ... existing code ...
    }

    // ... rest of the function ...
}

}  // namespace ttnn::operations::experimental::cumsum
