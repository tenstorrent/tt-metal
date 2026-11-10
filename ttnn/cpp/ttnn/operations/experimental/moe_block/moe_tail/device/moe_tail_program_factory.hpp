// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/moe_block/common/address_patches.hpp"
#include "ttnn/operations/experimental/moe_block/common/args_table.hpp"
#include "ttnn/operations/experimental/moe_block/common/chip_program.hpp"

#include "moe_tail_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

// One program for the mesh: the data-movement kernels' per-core args sit in per-chip DRAM tables (args_table.hpp), so
// the chips' programs are identical. Every address a kernel takes is a tensor's base; a cache hit rewrites a tensor's
// arguments only when its address moved (address_patches.hpp). The landing / ring / exchange slots other cores and
// chips write are CBs at the same address on every chip, read by the kernels with get_write_ptr.
struct TailEngineFactory {
    struct shared_variables_t {
        AddressPatches patches;
        // the per-chip DRAM tables of the kernels' args (common/args_table.hpp), kept with the program
        std::shared_ptr<const void> args_tables;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoETailParams& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coordinates,
        const MoETailInputs& tensor_args,
        Tensor& output);
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const MoETailParams& attributes,
        const MoETailInputs& tensor_args,
        Tensor& output);
};

struct TailAllToAllFactory {
    using shared_variables_t = TailEngineFactory::shared_variables_t;
    using cached_mesh_workload_t = TailEngineFactory::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const MoETailParams& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coordinates,
        const MoETailInputs& tensor_args,
        Tensor& output);
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const MoETailParams& attributes,
        const MoETailInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
