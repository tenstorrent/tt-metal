// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include "moe_head_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// One program for the mesh (head_program.hpp build_head_workload): every kernel's args in per-chip DRAM tables. A cache
// hit (override_runtime_arguments) writes every source's address into the kernels' common args and re-points the
// arena's CBs (common/args_table.hpp write_program_addresses): no descriptor is rebuilt, nothing per chip or per core
// is written.
struct MoEHeadFactory {
    struct shared_variables_t {
        ProgramAddresses addresses;  // what a cache hit writes (and the DRAM tables, kept with the program)
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoEHeadParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MoEHeadInputs& tensor_args,
        Tensor& output);
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached, const MoEHeadParams& attrs, const MoEHeadInputs& tensor_args, Tensor& output);
};

struct MoEHeadDeviceOperation {
    using operation_attributes_t = MoEHeadParams;
    using tensor_args_t = MoEHeadInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MoEHeadFactory>;

    // the config, plan_key and the input specs: never an address.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::moe_block::moe_head

namespace ttnn::prim {

Tensor moe_block_head(
    const ttnn::operations::experimental::moe_block::moe_head::MoEHeadParams& params,
    const ttnn::operations::experimental::moe_block::moe_head::MoEHeadInputs& inputs);

}  // namespace ttnn::prim
