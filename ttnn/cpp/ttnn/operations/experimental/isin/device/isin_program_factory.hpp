// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::experimental::prim {
struct IsInProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const IsinParams&, const IsinInputs&, Tensor&);
    static void override_runtime_arguments(cached_program_t&, const IsinParams&, const IsinInputs&, Tensor&);
};

}  // namespace ttnn::experimental::prim
