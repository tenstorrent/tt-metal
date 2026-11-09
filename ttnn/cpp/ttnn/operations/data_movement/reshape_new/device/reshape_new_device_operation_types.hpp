// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

// The program-cache hash and the profiler's op metadata come from attribute_names + attribute_values(): every field
// must be in both. Auto-reflection cannot be used, since TensorSpec has no default constructor.
struct ReshapeNewParams {
    // The complete output spec, decided by the front end; the device op only realises it.
    tt::tt_metal::TensorSpec output_spec;
    // PageCopy or Gather; NoOp and View never reach the device.
    ttnn::operations::data_movement::reshape_new::ReshapeKind kind;
    // Gather only: the 32-bit pattern written into output padding (the pad value's bits, repeated to fill 32 bits),
    // and whether padding is written at all.
    uint32_t pad_bits = 0;
    bool fill_padding = false;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grid;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_spec", "kind", "pad_bits", "fill_padding", "sub_core_grid");
    auto attribute_values() const {
        return std::forward_as_tuple(output_spec, kind, pad_bits, fill_padding, sub_core_grid);
    }
};

struct ReshapeNewInputs {
    Tensor input;
};

}  // namespace ttnn::prim
