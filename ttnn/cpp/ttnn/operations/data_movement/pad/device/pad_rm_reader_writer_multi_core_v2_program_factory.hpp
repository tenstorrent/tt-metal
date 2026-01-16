// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::operations::data_movement::pad::program {

struct PadRmReaderWriterMultiCoreV2SharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreCoord compute_with_storage_grid_size;
    ttnn::Shape input_tensor_start{};
};

struct PadRmReaderWriterMultiCoreV2ProgramFactory {
    using shared_variables_t = PadRmReaderWriterMultiCoreV2SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PadParams& operation_attributes,
        const PadInputs& tensor_args,
        Tensor& output);
};
}  // namespace ttnn::operations::data_movement::pad::program
