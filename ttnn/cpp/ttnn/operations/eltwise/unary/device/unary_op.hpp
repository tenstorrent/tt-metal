// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"


namespace ttnn::operations::unary {

enum class UnaryOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct Unary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;
    bool fp32_dest_acc_en;
    bool preserve_fp32_precision;
    DataType output_dtype;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &optional_output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UnaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;

    static Tensor operator()(
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& op_chain,
        const MemoryConfig& output_mem_config,
        bool fp32_dest_acc_en,
        bool preserve_fp32_precision,
        DataType output_dtype);
};

}  // namespace ttnn::operations::unary

namespace tt::stl::json {

template <>
struct from_json_t<ttnn::operations::unary::UnaryWithParam> {
    auto operator()(const nlohmann::json& json_object) const {
        return ttnn::operations::unary::UnaryWithParam{
            from_json<ttnn::operations::unary::UnaryOpType>(json_object["op_type"]),
            from_json<std::vector<float>>(json_object["params"])};
    }
};
};  // namespace tt::stl::json
