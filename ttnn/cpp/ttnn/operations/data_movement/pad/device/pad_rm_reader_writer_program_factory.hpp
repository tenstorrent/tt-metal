// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmReaderWriterSharedVariables {
    int ncores_h{};
    int ncores_w{};
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    // Owns the on-device pad-value constant. Without this, the local Tensor in create() drops
    // its DRAM buffer at the end of the function; on a cache hit override_runtime_arguments
    // does not refresh runtime_args[13], so the kernel reads the pad value from a freed slot
    // that the allocator may have handed out for other data (see #44565).
    Tensor pad_value_const_tensor;
};

struct PadRmReaderWriterProgramFactory {
    using shared_variables_t = PadRmReaderWriterSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PadParams& operation_attributes,
        const PadInputs& tensor_args,
        Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
