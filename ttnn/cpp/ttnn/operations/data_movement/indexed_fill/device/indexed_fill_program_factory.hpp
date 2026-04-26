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
    // True iff this program was built with the native CB-aliased fast path
    // (output CB globally-allocated to output buffer; reader copies input_a -> output L1
    // locally instead of via NoC). When true, override_runtime_arguments must call
    // UpdateDynamicCircularBufferAddress on `cb_data_handle` so the cached program's CB
    // base address tracks the current output buffer.
    bool is_native = false;
    // True iff this program was built with the shard-local path (WIDTH_SHARDED / BLOCK_SHARDED
    // input_a with matching output sharding). Like the native path, the data CB is globally
    // allocated to the output buffer, but the kernel iterates over ALL B batches per core.
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
