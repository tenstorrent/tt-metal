// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"

#include <cstdint>

namespace ttnn::prim {
using namespace tt::tt_metal;

// Single row - single core
struct SortProgramFactorySingleRowSingleCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
struct SortProgramFactoryCrossCoreDataExchange {
    // Workload-scoped helper tensor (physical-core lookup table) is allocated once
    // on cache miss inside create_workload_descriptor() and parked on the returned
    // WorkloadDescriptor::buffers so it outlives the cached workload via the
    // program cache.  emplace_runtime_args() with Buffer* lets the framework patch
    // the buffer address on cache hits without re-running this factory.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SortParams& attributes,
        const SortInputs& tensor_args,
        std::vector<Tensor>& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

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
struct SortProgramFactorySingleRowMultiCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim
