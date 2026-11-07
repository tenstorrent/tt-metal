// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// use_multicore and sub_core_grids are not part of the key: the codegen path always spreads work over the full
// compute grid, and the routing gate declines any call that asks for anything else.
struct PadCodegenParams {
    // 4D logical output shape. Padding is trailing only, so the input sits at the origin of the output.
    ttnn::Shape output_logical_shape;
    uint32_t packed_pad_value{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct PadCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

struct PadCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
