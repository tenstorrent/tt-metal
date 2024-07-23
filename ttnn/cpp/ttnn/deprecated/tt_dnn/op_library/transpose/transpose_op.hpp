// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class TransposeOpDim {
    WH, HC, CN, NH, NW, CW
};

enum class TransposeOpParallelizationStrategy {
    MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN
};

struct Transpose {
    const TransposeOpDim dim;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    TransposeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim transpose_dim, const MemoryConfig& output_mem_config);

// transpose with tensor and dimensions
Tensor transpose(const Tensor &a, std::int64_t dim1, std::int64_t dim2, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor &a, Tensor &output);

}  // namespace tt_metal

}  // namespace tt
