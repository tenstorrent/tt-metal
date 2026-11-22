// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "../../common/args_table.hpp"
#include "../../common/chip_program.hpp"

#include "moe_tail_device_operation_types.hpp"

namespace ttnn::operations::bringup::moe_block::moe_tail {

// One program for the mesh: the data-movement kernels' per-core args sit in per-chip DRAM tables (args_table.hpp), so
// the chips' programs are identical. Every address a kernel takes is a tensor's base, added on the device from the
// kernel's common args; a cache hit rewrites those common args (write_program_addresses). The landing / ring /
// exchange slots other cores and chips write are CBs at the same address on every chip, read by the kernels with
// get_write_ptr.
struct TailEngineFactory {
    struct shared_variables_t {
        ProgramAddresses addresses;  // what a cache hit writes (and the DRAM tables, kept with the program)
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

}  // namespace ttnn::operations::bringup::moe_block::moe_tail
