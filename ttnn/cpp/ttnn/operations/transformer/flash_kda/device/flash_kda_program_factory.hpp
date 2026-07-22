// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/flash_kda/device/flash_kda_device_operation_types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct FlashKdaSharedVars {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    std::uint32_t grid_y = 1;  // rows per column — used for column-major item->core mapping
    std::uint32_t num_cores = 0;
};

struct FlashKdaProgramFactory {
    using shared_variables_t = FlashKdaSharedVars;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const FlashKdaParams& attrs, const FlashKdaInputs& in, std::vector<Tensor>& outputs);

    static void override_runtime_arguments(
        cached_program_t& cached, const FlashKdaParams& attrs, const FlashKdaInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
