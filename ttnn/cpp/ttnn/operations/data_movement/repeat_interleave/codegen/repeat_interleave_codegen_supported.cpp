// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"

#include <tt-metalium/assert.hpp>

namespace ttnn::operations::data_movement {

namespace {

uint32_t normalize_dim(int32_t dim, uint32_t ndim) {
    return static_cast<uint32_t>(dim >= 0 ? dim : dim + static_cast<int32_t>(ndim));
}

}  // namespace

bool supported_by_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    // Sharded I/O composition (native S2I unshard, then I2S/reshard restore) is not ported;
    // reject a sharded input and a sharded requested output up front (repeat_interleave.yaml's
    // hand-authored sharded-input case -- the auto-generated ledger has no memory_config axis).
    if (input.memory_config().is_sharded()) {
        return false;
    }
    if (output_mem_config.has_value() && output_mem_config->is_sharded()) {
        return false;
    }

    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    // codegen_repeat_interleave.py invalidate_vector: "repeat_interleave codegen requires >=2D".
    if (ndim < 2) {
        return false;
    }
    // operation_attributes_t.rep_dim is stored left-padded to rank 4 (repeat_interleave.cpp's
    // codegen dispatch helper); a wider rank has no representable padded value.
    if (ndim > 4) {
        return false;
    }
    // invalidate_vector: "repeats must be >=1".
    if (repeats < 1) {
        return false;
    }
    // RepeatInterleaveCodegen.repeat_interleave takes an allocation-only path for a zero-volume
    // input and never launches a zero-work kernel; this program factory has no such path.
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }

    const uint32_t nd = normalize_dim(dim, ndim);
    if (nd >= ndim) {
        return false;
    }

    // coverage.dtypes in repeat_interleave.yaml (the ledger's swept/certified dtype set).
    const DataType dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {
        return false;
    }

    if (input.layout() == Layout::TILE) {
        // Sub-tile (last two) dims subdivide a 32x32 tile: tile-page replication != torch's
        // element-level interleave along H/W, so those dims are deferred (invalidate_vector:
        // "sub-tile (last two) dims deferred for TILE path").
        return nd < ndim - 2;
    }

    if (input.layout() == Layout::ROW_MAJOR) {
        // invalidate_vector: "RM_LAYOUT requires inner (last) dim >= 2".
        if (shape[ndim - 1] < 2) {
            return false;
        }
        // Within-stick (last W) dim IS a real, working kernel
        // (reader_repeat_interleave_lastdim_rm.cpp) but codegen_repeat_interleave.py's
        // invalidate_vector still defers it ("within-stick (last W) dim deferred for RM path").
        // This predicate mirrors that stale ledger, not device capability, per
        // repeat_interleave.yaml's dim == ndim - 1 out-of-scope case; the program factory does
        // not wire that kernel because supported_by_codegen() never lets it be reached.
        return nd != ndim - 1;
    }

    return false;
}

bool is_demoted(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    // Perf-demoted ledger: [1, 32, 64]|dim=1&repeats=2|float32|row_major.
    const auto& shape = input.logical_shape();
    if (shape.rank() != 3 || shape[0] != 1 || shape[1] != 32 || shape[2] != 64) {
        return false;
    }
    if (repeats != 2 || normalize_dim(dim, shape.rank()) != 1) {
        return false;
    }
    return input.dtype() == DataType::FLOAT32 && input.layout() == Layout::ROW_MAJOR;
}

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
    TT_THROW(
        "repeat_interleave: implementation must be one of \"auto\", \"native\", \"codegen\"; got \"{}\"",
        implementation);
}

}  // namespace ttnn::operations::data_movement
