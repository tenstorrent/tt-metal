// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

struct TransposeWHShardedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::CBHandle cb_src0{};
    tt::tt_metal::CBHandle cb_output{};
    uint32_t src0_single_tile_size{};
    uint32_t dst_single_tile_size{};
    uint32_t num_cores_x{};
    uint32_t num_cores_y{};
};

struct TransposeWHShardedProgramFactory {
    using shared_variables_t = TransposeWHShardedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TransposeParams& operation_attributes,
        const TransposeInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
