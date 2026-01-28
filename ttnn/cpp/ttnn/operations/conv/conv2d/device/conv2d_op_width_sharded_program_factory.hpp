// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct Conv2dWidthShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_sharded_act{};
        tt::tt_metal::CBHandle cb_output{};
        tt::tt_metal::CBHandle cb_partials{};
        bool partials_cb_uses_output = false;
        bool has_bias = false;
        CoreCoord full_core_grid;
        tt::tt_metal::KernelHandle weights_kernel_id{};
        uint32_t total_num_active_cores = 0;
        tt::tt_metal::DeviceStorage conv_reader_indices_storage;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
