// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation_types.hpp"

namespace ttnn::prim {

struct IndexedFillSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t page_size = 0;
    // When true, override_runtime_arguments must call UpdateDynamicCircularBufferAddress on
    // cb_data_handle to keep the cached program's CB base in sync with the output buffer.
    bool is_native = false;
    // Like is_native the data CB is aliased to the output buffer, but the kernel iterates
    // over all batches per core (WIDTH_SHARDED / BLOCK_SHARDED path).
    bool is_shard_local = false;
    bool is_tile = false;
    tt::tt_metal::CBHandle cb_data_handle{};
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

}  // namespace ttnn::prim
