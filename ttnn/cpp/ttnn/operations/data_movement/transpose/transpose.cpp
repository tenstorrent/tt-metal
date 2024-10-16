// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/decorators.hpp"
#include "device/transpose_op.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
inline Tensor transpose_(const Tensor& a, TransposeOpDim transpose_dim, const MemoryConfig& output_mem_config) {
    bool pad_c = false;
    bool pad_n = false;

    switch (transpose_dim) {
        case TransposeOpDim::HC: pad_c = true; break;
        case TransposeOpDim::NH:
            return ttnn::permute((const ttnn::Tensor)a, std::vector<int64_t>({2, 1, 0, 3}), output_mem_config);
        case TransposeOpDim::NW:
            return ttnn::permute((const ttnn::Tensor)a, std::vector<int64_t>({3, 1, 2, 0}), output_mem_config);
        case TransposeOpDim::CW:
            return ttnn::permute((const ttnn::Tensor)a, std::vector<int64_t>({0, 3, 2, 1}), output_mem_config);
        default: break;
    }

    if (a.get_layout() == Layout::ROW_MAJOR) {
        return operation::run(Transpose{transpose_dim, output_mem_config}, {a}).at(0);
    } else {
        // TODO: Add pad_n to run_with_autoformat when needed
        return operation::run_with_autoformat(
                   Transpose{transpose_dim, output_mem_config}, {a}, {}, {}, 0, pad_c /*, pad_n */)
            .at(0);
    }
}

}  // namespace detail

ttnn::Tensor ExecuteTranspose::invoke(uint8_t queue_id,
                                      const ttnn::Tensor& input_tensor,
                                      const int64_t& dim1,
                                      const int64_t& dim2,
                                      const std::optional<MemoryConfig>& memory_config_arg) {
    uint32_t normalized_dim1 = input_tensor.get_legacy_shape().get_normalized_index(dim1);
    uint32_t normalized_dim2 = input_tensor.get_legacy_shape().get_normalized_index(dim2);
    bool wh = normalized_dim2 == 2 && normalized_dim1 == 0;
    bool typecast = input_tensor.get_dtype() == DataType::BFLOAT8_B and input_tensor.get_layout() == Layout::TILE and
                    !wh and !input_tensor.is_sharded();
    Tensor b = typecast ? ttnn::typecast(input_tensor, DataType::BFLOAT16) : input_tensor;

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({b}))};

    operation::launch_with_autoformat(
        [dim1, dim2, memory_config_arg](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& a = input_tensors.at(0);
            auto memory_config = memory_config_arg.value_or(a.memory_config());
            uint32_t normalized_dim1 = a.get_legacy_shape().get_normalized_index(dim1);
            uint32_t normalized_dim2 = a.get_legacy_shape().get_normalized_index(dim2);

            TT_FATAL(normalized_dim1 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
            TT_FATAL(normalized_dim2 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");

            if ((normalized_dim1 == normalized_dim2) ||
                (a.get_legacy_shape()[normalized_dim1] == 1 && a.get_legacy_shape()[normalized_dim2] == 1)) {
                return {ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(
                    a, memory_config)};
            }

            if (normalized_dim1 > normalized_dim2) {
                std::swap(normalized_dim1, normalized_dim2);
            }

            TransposeOpDim transpose_dim = TransposeOpDim::NW;

            if (normalized_dim2 == 3 && normalized_dim1 == 0) {
                transpose_dim = TransposeOpDim::NW;
            } else if (normalized_dim2 == 3 && normalized_dim1 == 1) {
                transpose_dim = TransposeOpDim::CW;
            } else if (normalized_dim2 == 3 && normalized_dim1 == 2) {
                transpose_dim = TransposeOpDim::WH;
            } else if (normalized_dim2 == 2 && normalized_dim1 == 0) {
                transpose_dim = TransposeOpDim::NH;
            } else if (normalized_dim2 == 2 && normalized_dim1 == 1) {
                transpose_dim = TransposeOpDim::HC;
            } else if (normalized_dim2 == 1 && normalized_dim1 == 0) {
                transpose_dim = TransposeOpDim::CN;
            } else {
                TT_ASSERT(false, "Unsupported transpose dims");
            }
            return {detail::transpose_(a, transpose_dim, memory_config)};
        },
        {b},
        output_tensors);

    return typecast ? ttnn::typecast(output_tensors.at(0), DataType::BFLOAT8_B) : output_tensors.at(0);
}

ttnn::Tensor ExecuteTranspose::invoke(const ttnn::Tensor& input_tensor,
                                      const int64_t& dim1,
                                      const int64_t& dim2,
                                      const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, dim1, dim2, memory_config);
}

ttnn::Tensor ExecuteTranspose::invoke(const ttnn::Tensor& input_tensor, const int64_t& dim1, const int64_t& dim2) {
    return invoke(DefaultQueueId, input_tensor, dim1, dim2, std::nullopt);
}

}  // namespace ttnn::operations::data_movement
