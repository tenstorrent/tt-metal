// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    tt::tt_metal::CBHandle cb_src0{};
    std::vector<CoreCoord> cores;
    uint32_t num_slices{};
    uint32_t num_cores_unpadded{};
};

struct ShardedToInterleavedProgramFactory {
    using shared_variables_t = ShardedToInterleavedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
