// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_supported.hpp"

#include <array>

#include <tt-metalium/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::gather {

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::kNative;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::kCodegen;
    }
    TT_FATAL(implementation == "auto", "Unknown gather implementation selector: {}", implementation);
    return ImplementationSelector::kAuto;
}

bool supported_by_codegen(const Tensor& input_tensor, int8_t /*dim*/, const Tensor& input_index_tensor) {
    // manifest cases: ROW_MAJOR input/index is scope: out ("real-kernel-limit") -- this port's
    // kernels address purely in tile-page units (Ht/Wt_input/Wt_index) with no stick/row-major mode.
    if (input_tensor.layout() != Layout::TILE || input_index_tensor.layout() != Layout::TILE) {
        return false;
    }
    // port_scope.dtypes == [bfloat16]; nothing in the sweep exercises int32/uint32/float32/bfp8_b/
    // bfp4_b input through this factory directly (those go through GatherCodegen's own
    // decode->gather->encode host composition, out of this port's kernel scope).
    if (input_tensor.dtype() != DataType::BFLOAT16) {
        return false;
    }
    // Neither the sweep nor any manifest case exercises a sharded input/index through this factory:
    // GatherCodegen's own orchestrator (ops/gather/gather.py Step 5) converts sharded input/index to
    // DRAM-interleaved BEFORE ever reaching spec.py's factories, so sharded is not a real in-scope
    // case for this kernel; fall back to native, which already handles the full memory-config
    // spectrum through its own TensorAccessor-based kernels.
    if (input_tensor.memory_config().is_sharded() || input_index_tensor.memory_config().is_sharded()) {
        return false;
    }
    return true;
}

bool is_demoted(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor) {
    if (input_tensor.dtype() != DataType::BFLOAT16 || input_tensor.layout() != Layout::TILE) {
        return false;
    }

    // Ungeneralized perf demotions (design-prompt "Perf-demoted ledger entries" block): no mechanism
    // was identified relating these five case_ids by a general predicate, so each is an exact-match
    // branch on the ORIGINAL (pre pre_gather_transform_tensor) shape/dim, matching the case_id's
    // "shape|dim=X&index=shape|dtype|layout" encoding. None matches the ROW_MAJOR scope:out
    // condition, so all five stay demoted (not rejected by supported_by_codegen).
    struct DemotedCase {
        std::array<uint32_t, 4> input_shape;
        uint8_t input_rank;
        int8_t dim;
        std::array<uint32_t, 4> index_shape;
        uint8_t index_rank;
    };
    static constexpr std::array<DemotedCase, 5> kUngeneralizedDemotions = {{
        {{1, 1, 32, 64}, 4, -1, {1, 1, 32, 32}, 4},
        {{1, 1, 64, 128}, 4, -2, {1, 1, 32, 128}, 4},
        {{1, 1, 64, 64}, 4, -1, {1, 1, 64, 32}, 4},
        {{0, 1, 32, 64}, 3, -1, {0, 1, 32, 32}, 3},
        {{0, 0, 32, 64}, 2, -1, {0, 0, 32, 32}, 2},
    }};

    auto shape_matches = [](const ttnn::Shape& shape, const std::array<uint32_t, 4>& expected, uint8_t rank) {
        if (shape.rank() != rank) {
            return false;
        }
        // expected is right-aligned (unused leading slots are 0 and skipped via `rank`).
        const uint8_t offset = 4 - rank;
        for (uint8_t i = 0; i < rank; ++i) {
            if (shape[i] != expected[offset + i]) {
                return false;
            }
        }
        return true;
    };

    const auto& input_shape = input_tensor.logical_shape();
    const auto& index_shape = input_index_tensor.logical_shape();
    for (const auto& c : kUngeneralizedDemotions) {
        if (c.dim == dim && shape_matches(input_shape, c.input_shape, c.input_rank) &&
            shape_matches(index_shape, c.index_shape, c.index_rank)) {
            return true;
        }
    }
    return false;
}

}  // namespace ttnn::operations::data_movement::gather
