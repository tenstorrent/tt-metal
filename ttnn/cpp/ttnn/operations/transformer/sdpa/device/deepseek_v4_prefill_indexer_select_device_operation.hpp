// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_indexer_select_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <array>
#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct DeepseekV4PrefillIndexerSelectOperation {
    using operation_attributes_t = DeepseekV4PrefillIndexerSelectParams;
    using tensor_args_t = DeepseekV4PrefillIndexerSelectInputs;
    // [indices [1, 1, T, 512] uint32, survivor scratch for split query tiles, candidate blocks [1, 1, T, 2048]
    //  uint32, candidate survivor scratch]; candidates set fills the candidate pair, otherwise the first pair, and
    //  the other pair are placeholders.
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 4>;
    using tensor_return_value_t = std::array<Tensor, 4>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& outputs);

        // Cache hit: refresh the buffer addresses and the entry count.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    // Pool mode (candidates given): one query per unit over its candidate blocks' entries.
    struct PoolProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& outputs);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory, PoolProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Cores one query tile's blocks are split across: spare cores go to splitting once every query tile has a core.
uint32_t deepseek_v4_prefill_indexer_select_num_splits(uint32_t num_q_tiles, uint32_t num_cores);

// The indices (restricted to pool's candidate blocks when given), or with candidates set the candidate block ids.
Tensor deepseek_v4_prefill_indexer_select(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& idx_keys,
    const Tensor& start,
    const std::optional<Tensor>& pool,
    uint32_t compress_ratio,
    bool candidates,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim
