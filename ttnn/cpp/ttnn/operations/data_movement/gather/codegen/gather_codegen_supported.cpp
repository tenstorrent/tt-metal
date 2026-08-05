// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_supported.hpp"

#include <array>
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
    // three gather factories thread it into their work split (gather_core_grid), exactly as
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
    // on a longer axis cannot express the upper positions at all.
    constexpr uint32_t kUint16Max = 65535;
    if (input_index_tensor.dtype() == DataType::UINT16 && input_tensor.logical_shape()[gather_axis] > kUint16Max) {
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
    if (on_device && !ttnn::prim::gather_min_plan_fits_l1(input_tensor, input_index_tensor)) {
        return false;
    }
    return true;
}

bool is_demoted(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor) {
    if (input_tensor.dtype() != DataType::BFLOAT16 || input_tensor.layout() != Layout::TILE) {
        return false;
    }

    // UNGENERALIZED perf demotions: no predicate over the normalized attributes was found that
    // separates these measured case_ids from the in-scope cases that stay on codegen, so each is an
    // exact-match branch on the ORIGINAL (pre pre_gather_transform_tensor) shape/dim, matching the
    // case_id's "shape|dim=X&index=shape|dtype|layout" encoding. This is the floor when no mechanism
    // is identified, not the preferred form. None of them matches the ROW_MAJOR scope:out condition,
    // so they stay demoted here rather than being rejected by supported_by_codegen().
    struct DemotedCase {
        std::array<uint32_t, 4> input_shape;
        uint8_t input_rank;
        int8_t dim;
        std::array<uint32_t, 4> index_shape;
        uint8_t index_rank;
    };
    static constexpr std::array<DemotedCase, 13> kUngeneralizedDemotions = {{
        {{1, 1, 32, 64}, 4, -1, {1, 1, 32, 32}, 4},
        {{1, 1, 64, 64}, 4, -1, {1, 1, 64, 32}, 4},
        {{1, 1, 128, 128}, 4, -1, {1, 1, 128, 64}, 4},
        {{1, 1, 256, 256}, 4, -1, {1, 1, 256, 128}, 4},
        {{1, 1, 64, 128}, 4, -2, {1, 1, 32, 128}, 4},
        {{1, 1, 128, 128}, 4, -2, {1, 1, 64, 128}, 4},
        {{1, 1, 32, 15360}, 4, -1, {1, 1, 32, 7680}, 4},
        {{0, 1, 32, 64}, 3, -1, {0, 1, 32, 32}, 3},
        {{0, 1, 64, 128}, 3, -1, {0, 1, 64, 64}, 3},
        {{0, 0, 32, 64}, 2, -1, {0, 0, 32, 32}, 2},
        {{0, 0, 64, 128}, 2, -1, {0, 0, 64, 64}, 2},
        {{0, 0, 128, 256}, 2, -1, {0, 0, 128, 128}, 2},
        {{0, 0, 1, 151936}, 2, -1, {0, 0, 1, 151936}, 2},
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
    // The table stores dim as the sweep spells it (negative, from the end). A caller naming the
    // same axis positively is the same measured case, so compare in that one form.
    const auto rank = static_cast<int32_t>(input_shape.rank());
    if (dim < -rank || dim >= rank) {
        return false;
    }
    const int32_t dim_from_end = dim < 0 ? dim : dim - rank;
    for (const auto& c : kUngeneralizedDemotions) {
        if (c.dim == dim_from_end && shape_matches(input_shape, c.input_shape, c.input_rank) &&
            shape_matches(index_shape, c.index_shape, c.index_rank)) {
            return true;
        }
    }
    return false;
}

}  // namespace ttnn::operations::data_movement::gather
