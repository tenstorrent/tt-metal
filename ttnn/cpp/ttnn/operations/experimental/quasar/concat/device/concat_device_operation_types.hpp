// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim::qsr {

// Most inputs one concat program takes; the host op concatenates longer lists in batches. The
// original allows 47 (its runtime-arg budget). On Quasar the bound is the DM stack: a DM thread's
// 8 KiB local block holds its TLS (6,736 B with the current firmware) at one end and the stack at the
// other (crt0.S), so the reader gets ~1.4 KiB, and it keeps a TensorAccessor plus two counters per
// input on the stack (~45 B each: a 1,088 B frame at 24 inputs, 1,472 B at 32). 32 inputs overflowed
// it and hung on craq-sim; 16 (a 768 B frame) leaves ~700 B of margin.
inline constexpr uint32_t max_inputs_per_concat_program = 16;

// Quasar copy of ttnn::prim::ConcatParams. `groups` is gone: only the L1 height-sharded
// zero-copy factory implements grouped concat, and that factory is not part of the Quasar port
// (the host op rejects groups != 1).
struct ConcatParams {
    // Already normalized to [0, rank).
    uint32_t dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::CoreRangeSet> sub_core_grids;
};

struct ConcatInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim::qsr
