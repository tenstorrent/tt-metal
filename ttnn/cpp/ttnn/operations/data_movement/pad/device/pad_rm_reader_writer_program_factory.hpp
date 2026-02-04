// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
