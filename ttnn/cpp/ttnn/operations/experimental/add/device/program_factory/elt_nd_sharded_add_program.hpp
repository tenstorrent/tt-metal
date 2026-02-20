// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/add/device/add_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct EltNDShardedAddProgram {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle eltwise_kernel_id{};
        tt::tt_metal::CBHandle a_tensor_cb{};
        tt::tt_metal::CBHandle b_tensor_cb{};
        tt::tt_metal::CBHandle output_cb{};
        CoreRangeSet all_device_cores;
        uint32_t a_tensor_tile_size{};
        uint32_t b_tensor_tile_size{};
        uint32_t dst_tile_size{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const AddParams& operation_attributes, const AddInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const AddParams& operation_attributes,
        const AddInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
