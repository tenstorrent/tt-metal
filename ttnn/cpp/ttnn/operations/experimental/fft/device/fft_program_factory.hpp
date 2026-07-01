// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <vector>

#include "fft_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct FFTSharedVariables {
    // Kernel handles for the four-pass Stockham pipeline. Phase 1 stores
    // placeholders until the program factory is wired to the actual
    // pass1 / pass2 / pass3 / batch_fft kernels (see fft_program_factory.cpp).
    std::vector<tt::tt_metal::KernelHandle> kernel_ids;
    std::vector<CoreCoord>                  cores;
    uint32_t                                N = 0;
};

struct FFTProgramFactory {
    using shared_variables_t = FFTSharedVariables;
    using cached_program_t   = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FFTParams&            operation_attributes,
        const FFTTensorArgs&        tensor_args,
        std::tuple<Tensor, Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t&           cached_program,
        const FFTParams&            operation_attributes,
        const FFTTensorArgs&        tensor_args,
        std::tuple<Tensor, Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
