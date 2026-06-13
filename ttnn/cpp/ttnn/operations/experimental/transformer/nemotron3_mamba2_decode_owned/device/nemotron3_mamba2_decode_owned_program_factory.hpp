// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <vector>

#include "nemotron3_mamba2_decode_owned_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Nemotron3Mamba2DecodeOwnedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
    uint32_t g1_numcores{};
    uint32_t g2_numcores{};
    uint32_t num_blocks_per_core_group_1{};
    uint32_t num_blocks_per_core_group_2{};
    uint32_t head_dim_tiles{};
    uint32_t ssm_state_tiles{};
};

struct Nemotron3Mamba2DecodeOwnedProgramFactory {
    using shared_variables_t = Nemotron3Mamba2DecodeOwnedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Nemotron3Mamba2DecodeOwnedParams& operation_attributes,
        const Nemotron3Mamba2DecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Nemotron3Mamba2DecodeOwnedParams& operation_attributes,
        const Nemotron3Mamba2DecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
