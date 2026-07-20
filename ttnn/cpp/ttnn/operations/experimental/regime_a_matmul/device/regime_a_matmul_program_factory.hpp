// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "regime_a_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RegimeAMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;  // logical worker coords, index i = bank*preaders + slice
        std::vector<uint32_t> core_noc;              // per-core NoC group (0 => A/g0, 1 => B/g1)
        // Split-NOC kernel handles. readerA/writerA run on the noc==0 group, readerB/writerB on noc==1.
        tt::tt_metal::KernelHandle readerA{};
        tt::tt_metal::KernelHandle readerB{};
        tt::tt_metal::KernelHandle writerA{};
        tt::tt_metal::KernelHandle writerB{};
        tt::tt_metal::KernelHandle compute{};
        // Fused-epilogue / output-split layout (so override_runtime_arguments can locate the appended writer
        // args on a program-cache replay with fresh buffers). Writer fused args begin at index 17.
        bool has_bias{false};
        bool has_ternary{false};
        uint32_t n_chunks{1};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RegimeAMatmulParams& operation_attributes,
        const RegimeAMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RegimeAMatmulParams& operation_attributes,
        const RegimeAMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
