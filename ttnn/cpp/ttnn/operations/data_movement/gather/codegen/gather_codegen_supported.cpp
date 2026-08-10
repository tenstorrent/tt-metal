// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_supported.hpp"

#include <cstdint>

#include <tt_stl/assert.hpp>

#include "gather_codegen_program_factory.hpp"

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

bool supported_execution_controls(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Every codegen kernel sizes its output CB and its per-tile transfer from the output tensor's
    // own aligned TILE page, which is only well-defined for a non-sharded interleaved buffer; no
    // manifest case or sweep vector exercises a sharded output. Both the requested memory config
    // and a caller-supplied destination decide that placement, so both are checked here.
    //
    // sub_core_grids is deliberately absent: unlike the codegen builders of other ported ops, all
    // three gather factories thread it into their work split (split_gather_work), exactly as
    // ops/gather/spec.py threads ctx.attr("sub_core_grids") into split_cores, so the caller's core
    // reservation is honoured rather than widened.
    const auto& output_mem_config = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();
    if (output_mem_config.is_sharded()) {
        return false;
    }
    return !(optional_output_tensor.has_value() && optional_output_tensor.value().memory_config().is_sharded());
}

bool supported_by_codegen(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor) {
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

    // ttnn::gather() runs this gate ahead of its own dim range check, so an out-of-range dim must
    // not be indexed here; let it fall through to native's error.
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    if (dim < -rank || dim >= rank) {
        return false;
    }
    const int32_t gather_axis = dim < 0 ? dim + rank : dim;

    // A zero-extent index has no tile-rows to split, so the work split has nothing to divide;
    // ops/gather/gather.py answers this case from host metadata before ever building a program,
    // and ttnn::gather()'s own early exit only covers a rank-0 shape.
    if (input_tensor.logical_shape().volume() == 0 || input_index_tensor.logical_shape().volume() == 0) {
        return false;
    }

    // codegen_gather.py::invalidate_vector: an index element must be able to name any position
    // along the gathered axis. The kernels read the index at its own byte width, so a uint16 index
    // on a longer axis cannot express the upper positions at all. Positions are 0-based, so the
    // limit is the axis LENGTH, not the max value: 65536 addresses 0..65535 and fits.
    constexpr uint32_t kUint16MaxAxisLength = 65536;
    if (input_index_tensor.dtype() == DataType::UINT16 &&
        input_tensor.logical_shape()[gather_axis] > kUint16MaxAxisLength) {
        return false;
    }

    // Device-resource feasibility. Both dimension-scaled CB plans (row-buffered/tiled: Wt_input and
    // max(4, Wt_index) tile pages; streaming: chunk_tiles input pages) scale down to the streaming
    // floor, so a call is feasible exactly when that floor fits per-core L1. Only answerable with a
    // real device behind the tensors; the prim's validation step raises native's structural error
    // for anything else.
    const bool on_device = input_tensor.storage_type() == StorageType::DEVICE &&
                           input_index_tensor.storage_type() == StorageType::DEVICE &&
                           input_tensor.buffer() != nullptr && input_index_tensor.buffer() != nullptr;
    return !(on_device && !ttnn::prim::gather_min_plan_fits_l1(input_tensor, input_index_tensor));
}

bool is_demoted(const Tensor& /*input_tensor*/, int8_t /*dim*/, const Tensor& /*input_index_tensor*/) {
    // The perf gate demotes NOTHING: every measured in-scope configuration beats the native prim on
    // device, so `auto` runs codegen wherever supported_by_codegen() admits it. Re-derived from the
    // ledger each round rather than accreted -- the earlier exact-match table listed the perf-grid
    // shape tuples, all of which the ported measurements record as codegen wins over native, and the
    // five phase-2b seeds it also carried were reversed once measured on the forced-codegen leg.
    //
    // The signature stays so the auto branch's `supported && !demoted` wiring is identical whether or
    // not a demotion is ever found; a future demotion belongs here as a condition over the normalized
    // attributes, not as a shape-tuple carve-out.
    //
    // One in-scope configuration -- input [1,1,32,64], dim=-2, index [1,1,16,64], bfloat16, TILE --
    // beats the native prim but sits below parity with the same descriptor replayed through
    // generic_op, and was accepted there rather than demoted. It post-transforms to Ht=2, Wt_input=1,
    // Wt_index=1, i.e. the smallest work the row-buffered factory can be given (two tile-rows, two
    // cores), where the measured difference is fixed per-launch cost and not anything the descriptor
    // chooses. Accepting it is deliberate: demoting a case that beats native would route it to the
    // slower path under `auto`, and it must keep returning false here so forced
    // implementation="codegen" continues to measure it.
    return false;
}

}  // namespace ttnn::operations::data_movement::gather
