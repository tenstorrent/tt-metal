// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <cstdint>
#include <vector>

namespace ttnn::prim {
using namespace tt::tt_metal;

// Single row - single core
struct SortProgramFactorySingleRowSingleCore {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
struct SortProgramFactoryCrossCoreDataExchange {
    // The physical-core lookup table is a device tensor the factory allocates for itself, beyond the
    // op's declared io. It is built once on cache miss and returned in
    // ProgramArtifacts::op_owned_tensors, where the framework keeps it alive at a stable address for
    // the cached Program's lifetime and re-binds it on every dispatch alongside the io tensors.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);

    /**
     * @brief Strategies for slicing work across cores in cross-core data exchange sort.
     */
    enum class CrossCoreDataExchangeSortSlicingStrategy : uint8_t {
        USE_AS_MANY_CORES,  ///< Use all available cores to process the same line, optimizing for latency.
        FILL_CORES_FIRST,   ///< Fill cores sequentially before assigning additional work.
    };

    static uint32_t get_number_of_tiles_per_core(
        uint32_t total_number_of_cores,
        uint32_t Wt,
        const DataType& input_dtype,
        const DataType& index_dtype,
        CrossCoreDataExchangeSortSlicingStrategy slicing_strategy =
            CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);

    static uint32_t rounddown_pow2(uint32_t n);
};

// Single row - multi core
//
// Splits its nodes into two roles that run different kernels and share no dataflow buffer: a
// single-node coordinator work unit and a worker work unit over the rest of the grid.
struct SortProgramFactorySingleRowMultiCore {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim
