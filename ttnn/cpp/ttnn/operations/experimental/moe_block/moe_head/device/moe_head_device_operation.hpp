// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include "moe_head_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// One program for the mesh (head_program.hpp build_head_workload): every kernel's args in per-chip DRAM tables. A cache
// hit (override_runtime_arguments, once for the mesh's program) compares each source's address with the one the
// program holds and rewrites only the kernels' common args - and, for the arena, the CBs - of a source that moved: no
// descriptor is rebuilt, nothing per chip or per core is written.
struct MoEHeadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MoEHeadParams& attrs,
        const MoEHeadInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MoEHeadParams& attrs,
        const MoEHeadInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& coord);
};

// The HeadBuffers of a call: its tensors, the arena and the persistent words.
HeadBuffers head_buffers(const MoEHeadParams& attrs, const MoEHeadInputs& args);

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
    ttnn::operations::experimental::moe_block::moe_head::MoEHeadInputs inputs);

}  // namespace ttnn::prim
