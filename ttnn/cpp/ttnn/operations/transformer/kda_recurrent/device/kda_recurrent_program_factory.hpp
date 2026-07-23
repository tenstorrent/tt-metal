// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_device_operation_types.hpp"

namespace ttnn::prim {

struct KDARecurrentProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        uint32_t grid_y;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const KDARecurrentParams& attributes,
        const KDARecurrentInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const KDARecurrentParams& attributes,
        const KDARecurrentInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim
