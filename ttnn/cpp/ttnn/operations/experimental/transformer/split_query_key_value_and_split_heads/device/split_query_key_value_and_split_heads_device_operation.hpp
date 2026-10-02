// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "split_query_key_value_and_split_heads_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Interleaved input: the core grid and every per-core tile id derive from the input's padded shape
// (hashed), so the input/q/k/v runtime-arg bindings are the whole cache-hit refresh.
struct SplitFusedQKVAndSplitHeadsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
        const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

// Sharded input: no runtime args at all; the input and q/k/v shard buffers back globally-allocated
// CBs, and those CB bindings are the whole cache-hit refresh.
struct SplitFusedQKVAndSplitHeadsShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
        const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

struct SplitFusedQKVAndSplitHeadsDeviceOperation {
    using operation_attributes_t = SplitQueryKeyValueAndSplitHeadsParams;
    using tensor_args_t = SplitQueryKeyValueAndSplitHeadsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t =
        std::variant<SplitFusedQKVAndSplitHeadsProgramFactory, SplitFusedQKVAndSplitHeadsShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::vector<Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    uint32_t num_heads,
    const std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim
