// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <tt-metalium/kernel_types.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::prim {
//
// General-purpose softmax with arbitrary dimension support
//
struct SoftmaxProgramFactoryGeneral {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        std::size_t num_cores{};
        std::size_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static void override_runtime_arguments(cached_program_t&, const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
};

}  // namespace ttnn::prim
