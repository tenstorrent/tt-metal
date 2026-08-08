// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

#include <tt_stl/assert.hpp>

#include "permute_codegen_device_operation.hpp"

namespace ttnn::operations::data_movement {

bool supported_by_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    if (rank != dims.size() || rank < 2 || rank > PermuteCodegenDeviceOperation::kMaxDims) {
        return false;
    }
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    // Both readers and both writers bind interleaved TensorAccessorArgs (a two-element compile-time
    // ABI); a sharded buffer on either side widens that and the program factory rejects it. The
    // output side is gated here rather than at validate because native supports interleaved-to-
    // sharded, so "auto" has somewhere to fall back to.
    if (input_tensor.memory_config().is_sharded()) {
        return false;
    }
    if (output_mem_config.has_value() && output_mem_config->is_sharded()) {
        return false;
    }
    // Every kernel builder here assumes positive per-core work (split_work_to_cores rejects a
    // zero total); a nil-volume permute is logically well-defined (see ttnn's own zero-volume
    // shortcut) but has no bytes to move, so it is left to the native path rather than the
    // codegen kernels below.
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }

    // Manifest coverage restricts the RM port to these three dtypes. bfloat8_b + ROW_MAJOR is also
    // independently invalid (codegen_permute.py's invalidate_vector: no row-major representation
    // for the shared-exponent block-float layout) — see permute.yaml's real-kernel-limit case.
    const DataType dtype = input_tensor.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {
        return false;
    }

    // Reject the fused-WH-permute delegation this port does not implement (ops/permute/permute.py's
    // _fused_wh_ok, transcribed per permute.yaml's "left-out-for-now" case): dims[-1]==rank-2 with
    // enough outer batch and tile-aligned H/W routes the whole call to TransposeCodegen's fused-WH
    // kernels instead of this op's row-invariant/blocked-generic kernels.
    if (dims[rank - 1] == rank - 2) {
        constexpr uint32_t kFusedMinNc = 6;  // _PERMUTE_FUSED_MIN_NC
        constexpr uint32_t kTileH = 32;
        constexpr uint32_t kTileW = 32;
        uint32_t nc = 1;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            nc *= shape[i];
        }
        const uint32_t h = shape[rank - 2];
        const uint32_t w = shape[rank - 1];
        if (nc >= kFusedMinNc && h % kTileH == 0 && w % kTileW == 0) {
            return false;
        }
    }

    return true;
}

bool is_demoted(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims) {
    // Perf-only demotion, consulted by the "auto" selector alone: the case stays correct and
    // supported, so a forced implementation="codegen" call still runs it.
    //
    // A permutation that moves the last axis selects the blocked path (tilize -> transpose_tile ->
    // pack_untilize). That trip through the compute engine costs about what it saves, so the whole
    // path measures at parity: across the swept surface every blocked config lands between 0.95x
    // and 1.03x native device time, inside the run-to-run spread, while the row-invariant path
    // (last axis fixed, no compute, batched stick reads) wins on every config at 0.69x-0.88x.
    // Routing the blocked path to native keeps the win and gives up nothing measurable. Measured on
    // Blackhole; the mechanism is not arch-specific, so the predicate stays unconditional.
    const uint32_t rank = input_tensor.logical_shape().rank();
    if (rank < 2 || rank != dims.size()) {
        return false;
    }
    return dims[rank - 1] != rank - 1;
}

ImplementationSelector parse_implementation(const std::string& value) {
    if (value == "auto") {
        return ImplementationSelector::Auto;
    }
    if (value == "native") {
        return ImplementationSelector::Native;
    }
    if (value == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("permute: invalid implementation '{}' (expected 'auto', 'native', or 'codegen')", value);
}

}  // namespace ttnn::operations::data_movement
