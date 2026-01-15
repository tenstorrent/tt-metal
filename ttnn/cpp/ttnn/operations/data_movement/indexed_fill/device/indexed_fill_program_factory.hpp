// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation_types.hpp"

namespace ttnn::operations::data_movement::indexed_fill::program {

struct IndexedFillSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t page_size = 0;
};

struct IndexedFillProgramFactory {
    using shared_variables_t = IndexedFillSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const IndexedFillParams& operation_attributes, const IndexedFillInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const IndexedFillParams& operation_attributes,
        const IndexedFillInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::data_movement::indexed_fill::program
