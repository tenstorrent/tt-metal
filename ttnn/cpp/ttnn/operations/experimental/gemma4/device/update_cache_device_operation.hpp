// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::gemma4 {

struct UpdateCacheParams {
    uint32_t request_index;
    uint32_t request_local_tokens;
    uint32_t layer_idx;
    uint32_t num_layers;
    uint32_t cluster_axis;
};

struct UpdateCacheInputs {
    Tensor cache;
    Tensor input;
    Tensor descriptors;
};

struct UpdateCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UpdateCacheParams&, const UpdateCacheInputs&, Tensor&, const std::optional<MeshCoordinate>&);
};

struct UpdateCacheDeviceOperation {
    using operation_attributes_t = UpdateCacheParams;
    using tensor_args_t = UpdateCacheInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<UpdateCacheProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::gemma4
