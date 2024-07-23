// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/split/split_tiled.hpp"
#include "tt_metal/host_api.hpp"
namespace tt {

namespace tt_metal {

struct SplitLastDimTwoChunksTiled : public SplitTiled {
    // setting dim = 3 (last dim)
    // num_chunks = 2
    SplitLastDimTwoChunksTiled(const MemoryConfig &mem_config) : SplitTiled{3, 2, mem_config} { ; }
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("dim", "num_chunks", "output_mem_config");
    const auto attribute_values() const { return std::forward_as_tuple(dim, num_chunks, output_mem_config); }
};

std::vector<Tensor> split_last_dim_two_chunks_tiled(
    const Tensor &a, const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> split_dim_two_chunks_tiled(
    const Tensor &a, uint dim = 3, const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
