// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "matmul_wo_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::mla::program {

struct MatmulWOProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  cb_s2c_in and cb_s2c_out are
    // sharded onto the input / output tensor buffers respectively and are bound
    // via .buffer for dynamic CB address re-application on cache-hit.  The
    // collector-core reduction semaphore and per-core DRAM-bank / VChannel
    // runtime args are recomputed every call (apply_descriptor_runtime_args).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::mla::program
