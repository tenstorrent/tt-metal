// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gather_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::gather::program {
using namespace tt::tt_metal;
// Single row - single core
struct GatherProgramFactorySingleRowSingleCore {
    struct shared_variables_t {
        KernelHandle gather_reader_kernel_id{};
        KernelHandle gather_writer_kernel_id{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GatherParams&, const GatherInputs&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const GatherParams&, const GatherInputs&, tensor_return_value_t&);
};

// Single row - multi core
struct GatherProgramFactorySingleRowMultiCore {
    struct shared_variables_t {
        KernelHandle gather_reader_kernel_id{};
        KernelHandle gather_writer_kernel_id{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GatherParams&, const GatherInputs&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const GatherParams&, const GatherInputs&, tensor_return_value_t&);
};
}  // namespace ttnn::operations::data_movement::gather::program
