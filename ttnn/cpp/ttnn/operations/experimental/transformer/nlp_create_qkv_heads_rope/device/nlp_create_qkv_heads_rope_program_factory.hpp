// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/nlp_create_qkv_heads_rope_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsRopeSharedVariables {
    tt::tt_metal::KernelHandle qk_reader_kernel_id = 0;  // rope reader (q+k cores)
    tt::tt_metal::KernelHandle writer_kernel_id = 0;     // rope writer (ALL cores, drains c_16)
    tt::tt_metal::KernelHandle v_reader_kernel_id = 0;   // copy reader (v cores)
    std::vector<CoreCoord> cores;
    uint32_t Wt = 0;
    uint32_t num_q_rows = 0;  // head counts (num_q_heads / num_kv_heads); NOT scaled by Sqt
    uint32_t num_k_rows = 0;
    uint32_t num_v_rows = 0;
    uint32_t Sqt = 1;  // number of seq tile-rows; cores = (nq + 2*nkv) * Sqt (one core per head-row x seq-tile)
};

struct NlpCreateQkvHeadsRopeProgramFactory {
    using shared_variables_t = NlpCreateQkvHeadsRopeSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

    static cached_program_t create(
        const NlpCreateQkvHeadsRopeParams& operation_attributes,
        const NlpCreateQkvHeadsRopeInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpCreateQkvHeadsRopeParams& operation_attributes,
        const NlpCreateQkvHeadsRopeInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
