// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/circular_buffer.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/operations/madd/device/madd_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct MAddProgramFactorySharded {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_srcA{};
        tt::tt_metal::CBHandle cb_srcB{};
        tt::tt_metal::CBHandle cb_srcC{};
        tt::tt_metal::CBHandle cb_output{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MAddParams& operation_attributes, const MAddArgs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MAddParams& operation_attributes,
        const MAddArgs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
