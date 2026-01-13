// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::conv::conv2d::program {

struct Conv2dDepthwiseProgramFactory {
    struct shared_variables_t {
        std::vector<CoreCoord> cores_vec;
        tt::tt_metal::KernelHandle reader_id{};
        tt::tt_metal::KernelHandle writer_id{};
        tt::tt_metal::KernelHandle compute_id{};
        tt::tt_metal::CBHandle cb_input{};
        tt::tt_metal::CBHandle cb_output{};
        tt::tt_metal::CBHandle cb_weight{};
        tt::tt_metal::CBHandle cb_mul{};
        tt::tt_metal::CBHandle cb_bias{};  // Bias CB handle for depthwise conv with bias
        tt::tt_metal::CBHandle cb_raw_in{};          // Raw input CB handle
        tt::tt_metal::CBHandle cb_reader_indices{};  // Reader indices CB handle
        bool has_bias = false;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::conv::conv2d::program
