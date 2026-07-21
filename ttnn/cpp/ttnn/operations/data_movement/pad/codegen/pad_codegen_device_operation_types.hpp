// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

// Mirrors manifest.cache_key_fields for pad exactly; units (tile-page vs element) depend on
// layout and are resolved by the program factory, not stored here.
struct PadCodegenParams {
    uint32_t N_out{};
    uint32_t C_out{};
    uint32_t H_out{};
    uint32_t W_out{};
    uint32_t front_n{};
    uint32_t front_c{};
    uint32_t front_h{};
    uint32_t front_w{};
    uint32_t packed_pad_value{};
    uint32_t read_batch{};
    uint32_t write_batch{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct PadCodegenInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
