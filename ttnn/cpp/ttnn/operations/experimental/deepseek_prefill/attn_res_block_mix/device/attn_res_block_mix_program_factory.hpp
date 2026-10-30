// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_block_mix_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockMixProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 3>;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResBlockMixParams& operation_attributes,
        const AttnResBlockMixInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
