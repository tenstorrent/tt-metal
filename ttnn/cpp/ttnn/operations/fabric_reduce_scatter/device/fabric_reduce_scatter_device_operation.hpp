// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "fabric_reduce_scatter_device_operation_types.hpp"
#include "fabric_reduce_scatter_program_factory.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

// Every attribute is structural; the key (compute_program_hash) is the attributes and the input, scratch and
// output specs, not whether the output was preallocated. The addresses of input, scratch and output are written
// by the mesh workload factory's override on a hit.
struct FabricReduceScatterDeviceOperation {
    using operation_attributes_t = FabricReduceScatterParams;
    using tensor_args_t = FabricReduceScatterInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<FabricReduceScatterMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes every spec the miss checked; a hit checks only where the scratch and output sit.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // [1, 1, (G + 1) Sb, W], the input's dtype, TILE, DRAM interleaved.
    static tt::tt_metal::TensorSpec scratch_spec(const operation_attributes_t&, const Tensor& input);
};

}  // namespace ttnn::operations::fabric_reduce_scatter

namespace ttnn::prim {

Tensor fabric_reduce_scatter(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::fabric_reduce_scatter::FabricReduceScatterParams& params);

}  // namespace ttnn::prim
