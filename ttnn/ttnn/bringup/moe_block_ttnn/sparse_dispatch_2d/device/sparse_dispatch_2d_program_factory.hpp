// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sparse_dispatch_2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "../../common/args_table.hpp"
#include "../../common/chip_program.hpp"

namespace ttnn::operations::bringup::moe_block::sparse_dispatch_2d {

// One program for the whole mesh: per port core a reader (RISCV_1, NoC0) and a sender (RISCV_0, NoC1, the port's
// fabric link). Each core's values (its wiring, neighbours, fabric connection) sit in per-chip DRAM tables, written
// once with the program (common/args_table.hpp); the launch carries only the tensors' addresses (xbuf, route, sems,
// arena), as common args, which a cache hit rewrites with the arena-backed CBs (write_program_addresses). Nothing
// else is hash-excluded.
struct SparseDispatch2dProgramFactory {
    struct shared_variables_t {
        ProgramAddresses addresses;  // what a cache hit writes (and the DRAM tables, kept with the program)
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseDispatch2dParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseDispatch2dInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const SparseDispatch2dParams& args,
        const SparseDispatch2dInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::bringup::moe_block::sparse_dispatch_2d
