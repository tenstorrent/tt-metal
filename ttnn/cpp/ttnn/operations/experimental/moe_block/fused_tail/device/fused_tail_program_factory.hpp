// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/moe_block/common/address_patches.hpp"
#include "ttnn/operations/experimental/moe_block/common/chip_program.hpp"

#include "fused_tail_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

// One program a chip. Every address a kernel takes is a tensor's base; a cache hit rewrites a tensor's arguments only
// when its address moved (address_patches.hpp). The landing / ring / exchange slots other cores and chips write are CBs
// at the same address on every chip, read by the kernels with get_write_ptr.
struct TailEngineFactory {
    struct shared_variables_t {
        AddressPatches patches;
        // one program for the mesh: the chips' arg patch blocks (common/chip_program.hpp), kept with the program
        std::shared_ptr<const void> chip_table;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FusedTailParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FusedTailInputs& tensor_args,
        Tensor& output);
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const FusedTailParams& attrs,
        const FusedTailInputs& tensor_args,
        Tensor& output);
};

struct TailA2AFactory {
    using shared_variables_t = TailEngineFactory::shared_variables_t;
    using cached_mesh_workload_t = TailEngineFactory::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const FusedTailParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FusedTailInputs& tensor_args,
        Tensor& output);
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const FusedTailParams& attrs,
        const FusedTailInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
