// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/pool/upsample/device/upsample_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct UpsampleMultiCoreShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle writer_kernel{};
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle out_cb{};
        tt::tt_metal::CBHandle config_cb{};
        tt::tt_metal::DeviceStorage config_storage;
        tt::tt_metal::Buffer* config_buffer = nullptr;
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
