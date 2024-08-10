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

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_falcon7b(const Tensor &input_tensor_a, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(const Tensor &input_tensor, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_kv_cache_load_slice(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);

struct NlpCreateHeadsFalcon7B {
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

struct NlpCreateHeadsDecode {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

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

inline std::vector<Tensor> nlp_create_qkv_heads_falcon7b(const Tensor& input_tensor_a, const MemoryConfig& mem_config) {
    // TODO: hard-coded for falcon-7b; can delete if we switch to the more generic one (but perf may be worse)
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_a})),
        Tensor(operation::get_workers_for_op_output({input_tensor_a})),
        Tensor(operation::get_workers_for_op_output({input_tensor_a}))};
    operation::launch_op(
        [mem_config](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(NlpCreateHeadsFalcon7B{mem_config}, input_tensors);
        },
        {input_tensor_a},
        output_tensors);
    return output_tensors;
}
inline std::vector<Tensor> nlp_create_qkv_heads_decode(
    const Tensor &input_tensor,
    const uint32_t num_heads, std::optional<const uint32_t> num_kv_heads,
    const MemoryConfig& mem_config
) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                        Tensor(operation::get_workers_for_op_output({input_tensor})),
                                        Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [num_heads, num_kv_heads, mem_config] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);

            // Infer head_dim
            const auto& input_tensor = input_tensors.at(0);
            TT_FATAL(input_tensor.get_legacy_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
            uint32_t head_dim = input_tensor.get_legacy_shape()[3] / (num_heads + 2 * num_kv_heads_val);
            return operation::run(NlpCreateHeadsDecode{num_heads, num_kv_heads_val, head_dim, mem_config}, input_tensors);

        }, {input_tensor}, output_tensors);
    return output_tensors;
}

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
