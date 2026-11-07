// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "small_m_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct SmallMMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;  // logical worker coords, index i = bank*preaders + slice
        std::vector<uint32_t> core_noc;              // per-core NoC group (0 => A/g0, 1 => B/g1)
        // Split-NOC kernel handles. readerA/writerA run on the noc==0 group, readerB/writerB on noc==1.
        tt::tt_metal::KernelHandle readerA{};
        tt::tt_metal::KernelHandle readerB{};
        tt::tt_metal::KernelHandle writerA{};
        tt::tt_metal::KernelHandle writerB{};
        tt::tt_metal::KernelHandle compute{};
        // Fused-epilogue / output-split layout (so override_runtime_arguments can locate the appended writer
        // args on a program-cache replay with fresh buffers). Writer fused args begin at index 17.
        bool has_bias{false};
        bool has_ternary{false};
        uint32_t n_chunks{1};
    };
    // One program per mesh coordinate: placement depends on each chip's DRAM-bank-adjacent workers and NoC
    // hop distances, which differ with per-chip harvesting.
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SmallMMatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SmallMMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SmallMMatmulParams& operation_attributes,
        const SmallMMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

private:
    // Kept private: a public cached_program_t would also make this type match ProgramFactoryConcept.
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const SmallMMatmulParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const SmallMMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_program_runtime_arguments(
        tt::tt_metal::Program& program,
        const shared_variables_t& shared_variables,
        const SmallMMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<SmallMMatmulProgramFactory>);

}  // namespace ttnn::experimental::prim
