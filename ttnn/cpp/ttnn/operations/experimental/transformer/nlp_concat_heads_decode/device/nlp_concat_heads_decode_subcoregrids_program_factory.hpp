// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_concat_heads_decode_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsDecodeSubcoregridsSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t element_size{};
    uint32_t sub_tile_line_bytes{};
    uint32_t num_cores{};
    tt::tt_metal::CBHandle cb_q_output{};
    uint32_t face_h{};
    uint32_t tile_w{};
};

struct NLPConcatHeadsDecodeSubcoregridsProgramFactory {
    using shared_variables_t = NLPConcatHeadsDecodeSubcoregridsSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NlpConcatHeadsDecodeParams& operation_attributes,
        const NlpConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpConcatHeadsDecodeParams& operation_attributes,
        const NlpConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
