// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_supported.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_program_factory.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::untilize_codegen {

ImplementationSelector parse_implementation(const std::string& implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("Unknown implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool has_execution_control_override(
    bool use_multicore, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return !use_multicore || sub_core_grids.has_value();
}

namespace {

// port_scope.tile_aligned: [bfloat8_b] -- only bfloat8_b is required to be
// tile-aligned; bfloat16 is scope=in even non-aligned (routes through the
// with-unpadding builder). Manifest case [1,1,33,64] bfloat8_b (scope=out,
// reason=real-kernel-limit) confirms bf8_b non-aligned must route to native:
// the generic serves it via a typecast->bf16 step outside any ported builder
// entry point.
bool tile_alignment_ok(const Tensor& input) {
    const auto& shape = input.logical_shape();
    const bool aligned = (shape[-1] % tt::constants::TILE_WIDTH == 0) && (shape[-2] % tt::constants::TILE_HEIGHT == 0);
    if (aligned) {
        return true;
    }
    // Non-aligned: only bfloat16 is in scope (build_untilize_with_unpadding).
    return input.dtype() == DataType::BFLOAT16;
}

}  // namespace

bool supported_by_codegen(const Tensor& input) {
    // Scope: input must be TILE layout, interleaved (not sharded), dtype
    // bfloat16 or bfloat8_b (port_scope.layouts / port_scope.dtypes).
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    if (input.memory_config().is_sharded()) {
        return false;
    }
    if (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::BFLOAT8_B) {
        return false;
    }
    if (!tile_alignment_ok(input)) {
        return false;
    }
    // Rank must be at least 2 (row/col dims) for the tile-row decomposition to
    // make sense; every builder assumes shape[-2]/shape[-1] exist.
    if (input.logical_shape().rank() < 2) {
        return false;
    }

    // CB-fit bound at every dispatch-tree leaf. Only checkable once the input
    // is on a real device (plan_untilize_dispatch queries
    // compute_with_storage_grid_size()); a host or not-yet-allocated tensor
    // answers the scope questions above just fine and defers the structural
    // device/buffer checks to validate_on_program_cache_miss, per the porting
    // guide.
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return true;
    }
    const bool with_unpadding = input.logical_shape() != input.padded_shape();
    auto plan = ttnn::prim::plan_untilize_dispatch(input, with_unpadding);
    return ttnn::prim::untilize_cb_plan_fits(plan, ttnn::prim::kUntilizeUsableL1);
}

bool is_demoted(const Tensor& /*input*/) {
    // Start conservative: nothing is demoted until verify's performance
    // verdict identifies measured losers (routing.demotion_candidates), at
    // which point a general condition (not a shape list) should be added
    // here, per the porting guide.
    return false;
}

}  // namespace ttnn::operations::data_movement::untilize_codegen
