// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

struct NeighborhoodSDPAOperation {
    using operation_attributes_t = NeighborhoodSDPAParams;
    using tensor_args_t = NeighborhoodSDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct NeighborhoodSDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<NeighborhoodSDPAProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Explicit rather than reflected: the attributes carry a NeighborhoodConfig of std::arrays,
    // and the eight stage-5 blocks must share one compiled program.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor neighborhood_sdpa(
    const Tensor& query_tensor,
    const Tensor& key_tensor,
    const Tensor& value_tensor,
    const Tensor& gather_origin_table,
    const std::optional<Tensor>& interior_mask,
    const transformer::neighborhood::NeighborhoodConfig& config,
    uint32_t head_count,
    float scale,
    uint32_t tiles_per_kv_chunk,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    std::optional<float> k_norm_bound = std::nullopt,
    std::optional<uint32_t> probe = std::nullopt,
    uint32_t path_mode = 0);

}  // namespace ttnn::prim
