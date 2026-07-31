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
    // Ungeneralized demotions (perf-only, "auto" gate): phase-7 measurement found no predicate
    // relating device/attrs to these regressions, so each branch is an exact
    // (shape, dims, dtype) regression example from permute.yaml's perf-demoted ledger. None of
    // these match the "left-out-for-now" fused-WH scope=out condition (all have NC < 6 or
    // dims[-1] != rank-2), so they are genuine in-scope perf demotions, not mis-scoped cases.
    struct DemotedCase {
        std::initializer_list<uint32_t> shape;
        std::initializer_list<uint32_t> dims;
        DataType dtype;
    };
    static const DemotedCase kDemoted[] = {
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::INT32},
        {{1, 2, 3, 64, 96}, {2, 3, 1, 4, 0}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {2, 3, 1, 4, 0}, DataType::INT32},
        {{1, 4, 96, 128}, {3, 2, 0, 1}, DataType::INT32},
        {{2, 3, 4, 32, 64}, {4, 0, 2, 3, 1}, DataType::INT32},
        {{2, 3, 64, 96}, {3, 2, 0, 1}, DataType::FLOAT32},
        {{2, 3, 64, 96}, {3, 2, 1, 0}, DataType::FLOAT32},
        {{2, 96, 128}, {0, 2, 1}, DataType::FLOAT32},
        {{2, 96, 128}, {2, 0, 1}, DataType::FLOAT32},
        {{3, 64, 96}, {0, 2, 1}, DataType::FLOAT32},
        {{3, 64, 96}, {2, 0, 1}, DataType::FLOAT32},
        {{64, 96}, {1, 0}, DataType::FLOAT32},
        {{96, 64}, {1, 0}, DataType::FLOAT32},
    };

    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    if (rank != dims.size()) {
        return false;
    }
    const DataType dtype = input_tensor.dtype();
    for (const auto& demoted : kDemoted) {
        if (demoted.shape.size() != rank || demoted.dtype != dtype) {
            continue;
        }
        bool matches = true;
        uint32_t i = 0;
        for (uint32_t dim : demoted.shape) {
            if (shape[i] != dim) {
                matches = false;
                break;
            }
            ++i;
        }
        if (matches) {
            i = 0;
            for (uint32_t d : demoted.dims) {
                if (dims[i] != d) {
                    matches = false;
                    break;
                }
                ++i;
            }
        }
        if (matches) {
            return true;
        }
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
