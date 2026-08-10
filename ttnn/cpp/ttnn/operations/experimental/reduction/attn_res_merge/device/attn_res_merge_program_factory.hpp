// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_merge_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Where `site` lands once resolved against each operand's shape, in pages. The
// descriptor build and the cache-hit patch both need these and must agree, so
// one formula serves both.
struct AttnResMergeSiteOffsets {
    uint32_t shift;
    uint32_t mass;
    uint32_t live_scores;
    uint32_t partial;
};

AttnResMergeSiteOffsets attn_res_merge_site_offsets(
    const AttnResMergeParams& operation_attributes, const AttnResMergeInputs& tensor_args);

// Reader common-runtime-arg layout: the four offsets `site` decides occupy the
// leading slots in AttnResMergeSiteOffsets order, so the cache-hit patch writes
// one contiguous run. The kernel index is into ProgramDescriptor::kernels.
inline constexpr uint32_t kAttnResMergeReaderKernelIdx = 0;
inline constexpr uint32_t kAttnResMergeSiteArgIdx = 0;

struct AttnResMergeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResMergeParams& operation_attributes,
        const AttnResMergeInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
