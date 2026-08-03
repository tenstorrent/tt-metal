// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

#include <tt_stl/assert.hpp>

#include "permute_codegen_device_operation.hpp"

namespace ttnn::operations::data_movement {

bool supported_by_codegen(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    if (rank != dims.size() || rank < 2 || rank > PermuteCodegenDeviceOperation::kMaxDims) {
        return false;
    }
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    if (input_tensor.memory_config().is_sharded()) {
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
    // Perf-only demotions, consulted by the "auto" selector alone: these cases stay correct and
    // supported, so a forced implementation="codegen" call still runs them. Both conditions below
    // are mechanisms measured on device, not an enumerated regression list -- every case that
    // matches loses to native for the stated structural reason, and cases outside them either win
    // or sit inside the measurement window.
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    if (rank != dims.size()) {
        return false;
    }
    const DataType dtype = input_tensor.dtype();
    const uint32_t elem_size = input_tensor.element_size();
    const bool w_changing = dims[rank - 1] != rank - 1;
    const uint32_t x = shape[dims[rank - 1]];  // width of the moved axis == blocked path's X
    constexpr uint32_t kXBlockSize = 32;

    // The blocked path (tilize -> transpose_tile -> pack_untilize) carries float32/uint32 through
    // its CBs reinterpreted as int32 so the datum move stays bit-exact; native compiles the same
    // compute natively for float32 and accepts the TF32 rounding, which is ~120-160 ns/dispatch
    // cheaper. That fixed compute cost only decides the ratio once every 32x32 block is full of
    // real data; below X == kXBlockSize the blocks are mostly padding and the kernel is
    // write-latency-bound, which hides it.
    if ((dtype == DataType::FLOAT32 || dtype == DataType::UINT32) && w_changing && x >= kXBlockSize) {
        return true;
    }

    // The same blocked path in its narrow-write regime: with X small, x_blocks == 1 and the writer
    // emits 32 NOC writes per block carrying only X*elem_size bytes each, scattered a full output
    // plane apart because dims[0] == rank-1 hoists W to the outermost output axis. Per-write issue
    // cost, not bytes, sets the time, and the codegen blocked reader/writer pays ~40 cycles more
    // per block there than native's. X == 1 is excluded: both legs are equally write-starved and
    // tie. Two-byte dtypes halve the write count per byte and win, so this is 4-byte only. rank is
    // fenced empirically -- the rank-5 analogues of this shape all measure at or above parity.
    if (rank == 4 && elem_size == 4 && dims[0] == rank - 1 && x >= 2 && x < kXBlockSize) {
        return true;
    }

    // The degenerate end of the same regime, dtype-independent: X == 1 leaves one real datum per
    // 32x32 block, so the whole dispatch is per-write issue cost, and W landing at output position
    // rank-2 makes those writes contiguous-strided rather than plane-scattered -- exactly the shape
    // native's writer handles with less per-row overhead. W anywhere else, or X >= 2, measures on
    // the winning side.
    if (w_changing && dims[rank - 2] == rank - 1 && x == 1) {
        return true;
    }

    return false;
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
