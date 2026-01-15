// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/pool/upsample/device/upsample_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct UpsampleMultiCoreInterleavedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        std::size_t num_cores = 0;
        std::size_t num_cores_y = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UpsampleParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
