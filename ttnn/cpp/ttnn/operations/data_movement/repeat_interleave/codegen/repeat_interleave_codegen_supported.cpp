// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"

namespace ttnn::operations::data_movement::repeat_interleave {

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::kNative;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::kCodegen;
    }
    return ImplementationSelector::kAuto;
}

// Transcribed from common/sweeps/codegen_repeat_interleave.py's invalidate_vector, which is the
// ledger's source of truth for this op (codegen_repeat_interleave.py has no upstream_sweep_module).
// `dim` is the raw (possibly negative) user-facing axis, normalized here exactly like the sweep's
// `nd = dim if dim >= 0 else dim + ndim`.
//
// NOTE: invalidate_vector rejects ROW_MAJOR nd == ndim - 1 (the within-stick/W dim), but that gate
// is stale: RepeatInterleaveCodegen (ops/repeat_interleave/repeat_interleave.py) actually implements
// this case via build_repeat_interleave_lastdim_rm. Transcribed verbatim anyway, per the manifest's
// explicit scope: out classification for that case and the instruction that supported_by_codegen()
// must agree with the manifest's cases.
bool supported_by_codegen(const Tensor& input_tensor, uint32_t repeats, int32_t dim) {
    const auto logical_shape = input_tensor.logical_shape();
    const int32_t ndim = static_cast<int32_t>(logical_shape.rank());
    if (ndim < 2) {
        return false;
    }
    if (repeats < 1) {
        return false;
    }
    // Not covered by invalidate_vector (the sweep carries no memory_config axis) but required by
    // the manifest's hand-authored sharded-input case: the port has no S2I unshard step, so a
    // sharded input can never reach the codegen path.
    if (input_tensor.is_sharded()) {
        return false;
    }

    const int32_t nd = dim >= 0 ? dim : dim + ndim;
    if (nd < 0 || nd >= ndim) {
        return false;
    }

    const tt::tt_metal::Layout layout = input_tensor.layout();
    if (layout == tt::tt_metal::Layout::TILE) {
        // Sub-tile (last two) dims: no tile-granular sequencer exists.
        return nd < ndim - 2;
    }

    if (layout == tt::tt_metal::Layout::ROW_MAJOR) {
        if (input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B) {
            return false;
        }
        if (logical_shape[-1] < 2) {
            return false;
        }
        return nd != ndim - 1;
    }

    return false;
}

// No perf-demoted cases have been identified for repeat_interleave yet. Still emitted (per the
// porting guide) so the auto branch's `supported_by_codegen(attrs) && !is_demoted(attrs)` wiring is
// identical regardless of whether anything is demoted.
bool is_demoted(const Tensor& /*input_tensor*/, uint32_t /*repeats*/, int32_t /*dim*/) { return false; }

}  // namespace ttnn::operations::data_movement::repeat_interleave
