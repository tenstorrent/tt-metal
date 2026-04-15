// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unary_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct UnaryProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);
};

struct UnarySubCoreGridProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        std::vector<CoreCoord> cores_with_rtargs;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UnaryParams& operation_attributes,
        const UnaryInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
