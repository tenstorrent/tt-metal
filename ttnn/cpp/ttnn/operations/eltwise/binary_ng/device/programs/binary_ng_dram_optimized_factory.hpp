// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation_types.hpp"

namespace ttnn::operations::binary_ng::program {

struct BinaryNgDramOptimizedProgram {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle eltwise_kernel_id{};
        tt::tt_metal::CBHandle a_tensor_cb{};
        tt::tt_metal::CBHandle b_tensor_cb{};
        tt::tt_metal::CBHandle output_cb{};
        CoreRangeSet dram_device_cores;
        // TODO: Add check for tile/page size for input/output tensro and reduce num of variables in the struct
        uint32_t a_tensor_tile_size{};
        uint32_t b_tensor_tile_size{};
        uint32_t dst_tile_size{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static std::optional<std::string> validate_program(
        const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args);

    static cached_program_t create(
        const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const BinaryNgParams& operation_attributes,
        const BinaryNgInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::binary_ng::program
