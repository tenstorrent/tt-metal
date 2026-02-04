// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"

#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::operations::core {

struct ToMemoryConfig {
    // TODO: Move to cpp once we merge with tt_eager
    static Tensor invoke(
        const ttnn::Tensor& tensor,
        const ttnn::MemoryConfig& memory_config,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt) {
        using namespace tt::tt_metal;
        // Temporary until we see why buffer data not being populated
        const auto original_memory_config = ttnn::get_memory_config(tensor);
        if (original_memory_config.has_value() && original_memory_config.value() == memory_config &&
            !output_tensor.has_value()) {
            return tensor;
        }
        std::vector<std::optional<Tensor>> optional_output_tensors;
        if (output_tensor.has_value()) {
            optional_output_tensors.push_back(output_tensor);
        }

        if (memory_config.is_sharded()) {
            // to_sharded path
            if (tensor.is_sharded()) {
                // reshard
                const auto input_memory_config = ttnn::get_memory_config(tensor);
                const auto input_shard_spec = input_memory_config.value().shard_spec().value();
                const auto output_shard_spec = memory_config.shard_spec().value();
                // Check if we need to use the s2i->i2s workaround
                bool use_reshard_workaround =
                    (input_shard_spec.shape[1] != output_shard_spec.shape[1]) &&
                    (input_memory_config.value().memory_layout() != memory_config.memory_layout() &&
                     tensor.layout() == Layout::ROW_MAJOR);
                if (!use_reshard_workaround) {
                    if (dtype.has_value()) {
                        throw std::runtime_error(
                            "dtype cannot be specified when converting sharded tensor to sharded tensor");
                    }
                    return ttnn::reshard(tensor, memory_config, output_tensor);
                }  // for row-major tensors where shard-spec[1] is different for input shard and output shard

                TT_FATAL(memory_config.is_sharded(), "Memory config must be sharded for this operation");
                Tensor temp = ttnn::prim::sharded_to_interleaved(
                    tensor, ttnn::DRAM_MEMORY_CONFIG, dtype.value_or(tensor.dtype()));
                const bool keep_l1_aligned = false;
                return ttnn::interleaved_to_sharded(
                    temp,
                    memory_config,
                    dtype.value_or(temp.dtype()),
                    keep_l1_aligned,
                    optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
            }
            auto bbox = memory_config.shard_spec().value().grid.bounding_box();
            CoreCoord grid_size(bbox.end_coord.x + 1, bbox.end_coord.y + 1);
            const bool keep_l1_aligned = false;
            return ttnn::interleaved_to_sharded(
                tensor,
                memory_config,
                dtype.value_or(tensor.dtype()),
                keep_l1_aligned,
                optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
        }
        // to_interleaved path
        if (tensor.is_sharded()) {
            return ttnn::prim::sharded_to_interleaved(
                tensor, memory_config, dtype.value_or(tensor.dtype()), output_tensor);
        }  // L1 to DRAM or DRAM to L1
        return ttnn::prim::copy(
            tensor,
            memory_config,
            dtype.value_or(tensor.dtype()),
            optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
    }
};

}  // namespace ttnn::operations::core
