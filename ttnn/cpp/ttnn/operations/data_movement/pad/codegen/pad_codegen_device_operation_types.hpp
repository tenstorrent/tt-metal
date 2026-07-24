// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

// Mirrors manifest.cache_key_fields for pad. The manifest leaves the tile-page-vs-element unit
// choice to the translate stage ("a translate-stage decision driven by layout_split"); this port
// stores H_out/W_out/front_h/front_w in ELEMENT units for both layouts (not tile-pages for TILE),
// because validate_on_program_cache_miss only has this struct + the input tensor to re-check
// supported_by_codegen(), and a tile-rounded H_out cannot distinguish an exact multiple-of-32
// back-pad from a sub-tile one
// (both round to the same Ht_out) -- exactly the distinction the TILE gate must make. The program
// factory derives tile-page counts (Ht_in/Wt_in/front_ht/front_wt/Ht_out/Wt_out) from these element
// values via ceil-division, matching ops/pad/pad.py's own Ht_in/Ht_out computation.
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
