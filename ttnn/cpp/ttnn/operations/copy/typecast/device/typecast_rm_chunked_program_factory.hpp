// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct TypecastRowMajorChunkedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle typecast_reader_kernel_id{};
        tt::tt_metal::KernelHandle typecast_writer_kernel_id{};
        uint32_t num_cores{};
        uint32_t chunks_per_row{};
        uint32_t input_chunk_size_bytes{};
        uint32_t output_chunk_size_bytes{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TypecastParams& operation_attributes,
        const TypecastInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
