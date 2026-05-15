// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        uint32_t tile_size{};
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
