// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"
namespace ttnn::operations::data_movement {

struct InterleavedToShardedPartialDeviceOperation {
    const CoreCoord grid_size;
    const tt::tt_metal::ShardSpec shard_spec;
    const uint32_t num_slices;
    const uint32_t slice_index;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("grid_size", "shard_spec", "output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->grid_size),
            std::cref(this->shard_spec),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype));
    }
};
}
