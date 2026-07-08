// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::data_movement::move_codegen {

using namespace tt::tt_metal;

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
    TT_THROW("ttnn.move: unknown implementation \"{}\" (expected \"auto\", \"native\", or \"codegen\")", implementation);
}

bool supported_by_codegen(const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config) {
    // manifests/move.yaml scopes only the non-sharded interleaved TILE/ROW_MAJOR paths for this
    // port's first pass; ops/identity/spec.py's build_identity_sharded_factory covers sharded
    // in tt-dm-codegen, but that path isn't ported here, so sharded stays on native's
    // MULTI_CORE_SHARDED strategy.
    if (input_tensor.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }

    const Layout layout = input_tensor.layout();
    if (layout == Layout::TILE) {
        // common/sweeps/codegen_move.py invalidate_vector: TILE needs rank >= 2 and the last two
        // dims tile-aligned.
        const auto& shape = input_tensor.padded_shape();
        if (shape.rank() < 2) {
            return false;
        }
        return shape[-1] % tt::constants::TILE_WIDTH == 0 && shape[-2] % tt::constants::TILE_HEIGHT == 0;
    }
    if (layout == Layout::ROW_MAJOR) {
        // Block-float formats are tile-packed (shared exponent per tile) and have no row-major
        // representation (common/sweeps/codegen_move.py invalidate_vector).
        return !is_block_float(input_tensor.dtype());
    }
    // ops/move/move.py raises NotImplementedError for any other layout; TILE and ROW_MAJOR are
    // exhaustive today, so this is a defensive fallback.
    return false;
}

}  // namespace ttnn::operations::data_movement::move_codegen
