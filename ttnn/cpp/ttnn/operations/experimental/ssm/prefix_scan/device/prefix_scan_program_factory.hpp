// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prefix_scan_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::ssm::prefix_scan::program {

struct PrefixScanSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::CBHandle cb_a_in{};
    tt::tt_metal::CBHandle cb_bx_in{};
    tt::tt_metal::CBHandle cb_h_in{};
    tt::tt_metal::CBHandle cb_out{};
    uint32_t total_tiles = 0;
    uint32_t total_tiles_per_row = 0;
    uint32_t total_tiles_per_col = 0;
    uint32_t num_chunks_per_row = 0;
    uint32_t sharded_hidden_state_length = 0;
};

struct PrefixScanProgramFactory {
    using shared_variables_t = PrefixScanSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ssm::prefix_scan::program
