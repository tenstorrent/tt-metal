// gelu_backward.cpp
#include "device/gelu_backward/gelu_backward_device_operation.hpp"
#include "unary_backward.hpp"

namespace ttnn::operations::experimental {

Tensor GeluBackwardOperation::invoke(
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const string& approximate,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> input_grad_tensor) {
    // TODO: verify those attributes are calculated properly
    DataType output_dtype = input_tensor.get_dtype();
    auto arch = input_tensor.device()->arch();
    bool preserve_fp32_precision = (arch != tt::ARCH::GRAYSKULL) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or input_tensor.get_dtype() == DataType::INT32;
    bool bfp8_pack_precise = false;

    auto output_memory_config = input_grad_tensor.has_value() ? input_grad_tensor.value().memory_config()
                                                              : memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::gelu_bw(
        grad_output_tensor,
        input_tensor,
        approximate,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        input_grad_tensor);
}

}  // namespace ttnn::operations::experimental
