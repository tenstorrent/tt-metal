// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks multi_core_nlp_kv_cache_load_slice(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);

struct NlpKVCacheLoadSlice {
    const Shape output_tensor_start;
    const Shape output_tensor_end;
    const Shape output_shape;
    const Shape input_shape;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

inline Tensor nlp_kv_cache_load_slice(const Tensor &input_tensor_a, const uint32_t seq_len_start, const uint32_t seq_len_end){
    // No-op (Will do a tensor copy)
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a}))};

    operation::launch_op(
        [seq_len_start, seq_len_end] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor_a = input_tensors.at(0);
            auto input_tensor_shape = input_tensor_a.get_legacy_shape();
            auto dim0 = input_tensor_shape[0];
            auto dim1 = input_tensor_shape[1];
            auto head_dim = input_tensor_shape[3];

            const Shape output_tensor_start = {
                0,
                0,
                seq_len_start,
                0,
            };

            const Shape output_tensor_end = {
                dim0-1,
                dim1-1,
                seq_len_end-1,
                head_dim-1,
            };

            const Shape output_tensor_shape = {
                output_tensor_end[0] - output_tensor_start[0] + 1,
                output_tensor_end[1] - output_tensor_start[1] + 1,
                output_tensor_end[2] - output_tensor_start[2] + 1,
                output_tensor_end[3] - output_tensor_start[3] + 1,
            };
            return operation::run(NlpKVCacheLoadSlice{output_tensor_start, output_tensor_end, output_tensor_shape, input_tensor_shape}, {input_tensor_a});
        }, {input_tensor_a}, output_tensors);
    return output_tensors.at(0);
}

}  // namespace tt_metal

}  // namespace tt
