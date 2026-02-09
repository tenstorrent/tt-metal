// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

#include <cstdint>

namespace ttnn::operations::data_movement::sort::program {
using namespace tt::tt_metal;
// Single row - single core
struct SortProgramFactorySingleRowSingleCore {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle compute_kernel_id{};
        KernelHandle writer_kernel_id{};
        CoreCoord storage_grid_size;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SortParams&, const SortInputs&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const SortParams&, const SortInputs&, tensor_return_value_t&);
};

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
struct SortProgramFactoryCrossCoreDataExchange {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle compute_kernel_id{};
        KernelHandle writer_kernel_id{};
        CoreRangeSet core_range_set;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const SortParams&, const SortInputs&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const SortParams&, const SortInputs&, tensor_return_value_t&);

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
    struct shared_variables_t {
        KernelHandle coordinator_kernel_id{};
        KernelHandle reader_kernel_id{};
        KernelHandle compute_kernel_id{};
        KernelHandle writer_kernel_id{};
        CoreCoord coordinator_core;
        CoreRangeSet worker_core_range;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SortParams&, const SortInputs&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const SortParams&, const SortInputs&, tensor_return_value_t&);
};

}  // namespace ttnn::operations::data_movement::sort::program
