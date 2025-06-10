// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/copy/typecast/device/typecast_device_op.hpp"

namespace ttnn {
namespace operations {
namespace copy {

namespace detail {

inline Tensor typecast_impl(
    QueueId queue_id,
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    DataType input_dtype = input_tensor.dtype();
    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
    bool bfp8_pack_precise = (output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return ttnn::prim::typecast(
        queue_id,
        input_tensor,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}
}  // namespace detail

Tensor Typecast::invoke(
    const QueueId queue_id,
    const Tensor& input,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    DataType input_dtype = input.dtype();
    return detail::typecast_impl(
        queue_id, input, output_dtype, memory_config_arg, optional_output_tensor, sub_core_grids);
}

// eltwise_typecast is not currently supported on Grayskull
Tensor Typecast::invoke(
    const QueueId queue_id,
    const Tensor& input_tensor,
    const DataType& tt_input_dtype,
    const DataType& tt_output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(tt_input_dtype == input_tensor.dtype(), "input dtype and input tensor's dtype provided should match");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            tt_output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    return detail::typecast_impl(
        queue_id, input_tensor, tt_output_dtype, memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace copy
}  // namespace operations
}  // namespace ttnn
