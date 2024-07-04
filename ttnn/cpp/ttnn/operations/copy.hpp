// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

namespace ttnn {
namespace operations {
namespace copy {

struct Typecast {
    static Tensor execute_on_worker_thread(
        const uint8_t& queue_id,
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {

        if(optional_output_tensor.has_value()){
            TT_FATAL(output_dtype == optional_output_tensor.value().get_dtype(), "If both output dtype and output tensor provided dtype should match");
        }

        DataType input_dtype = input.get_dtype();
        auto memory_config = memory_config_arg.value_or(input.memory_config());
        bool preserve_fp32_precision = input_dtype == DataType::FLOAT32;
        bool fp32_dest_acc_en = preserve_fp32_precision or
                                output_dtype == DataType::UINT32 or
                                output_dtype == DataType::INT32 or
                                output_dtype == DataType::FLOAT32 or
                                input_dtype == DataType::UINT32 or
                                input_dtype == DataType::INT32;
        auto unary_op = UnaryWithParam{UnaryOpType::TYPECAST, {static_cast<float>(input_dtype), static_cast<float>(output_dtype)}};
        auto eltwise_op = EltwiseUnary{{unary_op}, memory_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype};
        return operation::run(eltwise_op, {input}, {}, {optional_output_tensor}, queue_id).at(0);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {

        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(DefaultQueueId, input, output_dtype, memory_config_arg, optional_output_tensor);
    }
};
}  // namespace copy
}  // namespace operations

constexpr auto typecast =
    ttnn::register_operation<ttnn::operations::copy::Typecast>("ttnn::typecast");

}  // namespace ttnn
