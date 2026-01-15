// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "split_query_key_value_and_split_heads_device_operation_types.hpp"
#include "split_query_key_value_and_split_heads_program_factory.hpp"
#include "split_query_key_value_and_split_heads_sharded_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads {

struct SplitFusedQKVAndSplitHeadsDeviceOperation {
    using operation_attributes_t = SplitQueryKeyValueAndSplitHeadsParams;
    using tensor_args_t = SplitQueryKeyValueAndSplitHeadsInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<
        program::SplitFusedQKVAndSplitHeadsProgramFactory,
        program::SplitFusedQKVAndSplitHeadsShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads

namespace ttnn::prim {
std::vector<Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    uint32_t num_heads,
    const std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim
