// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::repeat_interleave_codegen {

namespace {

uint32_t normalize_dim(int32_t dim, uint32_t ndim) {
    return static_cast<uint32_t>(dim >= 0 ? dim : dim + static_cast<int32_t>(ndim));
}

uint32_t page_alignment(const MemoryConfig& memory_config) {
    using tt::tt_metal::BufferType;
    return memory_config.buffer_type() == BufferType::L1 ? tt::tt_metal::hal::get_l1_alignment()
                                                         : tt::tt_metal::hal::get_dram_alignment();
}

}  // namespace

RmCbBudget rm_cb_budget(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    const auto& shape = input.logical_shape();
    const uint32_t stick_size = shape[shape.rank() - 1] * input.element_size();
    const MemoryConfig& out_config = output_mem_config.value_or(input.memory_config());
    // Reader and writer each page a whole stick through the same slot, so the slot has to satisfy
    // the stricter of the two sides' page alignments.
    const uint32_t slot_stride = std::max(
        tt::round_up(stick_size, page_alignment(input.memory_config())),
        tt::round_up(stick_size, page_alignment(out_config)));

    // get_max_l1_space() measures down from the lowest occupied compute L1 address when there is
    // one, so the budget excludes L1 already held by resident buffers instead of assuming everything
    // above the allocator base is free.
    const uint32_t l1_capacity = get_max_l1_space(input);
    return {slot_stride, slot_stride == 0 ? 0 : l1_capacity / slot_stride};
}

bool supported_by_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    // The gate below asks only about layout, dtype and memory config, and a host tensor answers all
    // three -- Tensor::is_sharded() is false off-device and memory_config() reads the spec, not a
    // buffer. Reject up front so a host tensor takes native's own validation rather than tripping
    // over Tensor::device() in rm_cb_budget() or a null buffer in the program factory.
    if (input.storage_type() != StorageType::DEVICE || input.buffer() == nullptr) {
        return false;
    }

    // Sharded I/O composition (unshard to interleaved, run, reshard) is not implemented here, so
    // reject a sharded input and a sharded requested output.
    if (input.memory_config().is_sharded()) {
        return false;
    }
    if (output_mem_config.has_value() && output_mem_config->is_sharded()) {
        return false;
    }

    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    // A rank-1 tensor's only dim is its within-stick / sub-tile dim, which neither layout's kernels
    // replicate.
    if (ndim < 2) {
        return false;
    }
    // operation_attributes_t.rep_dim is stored left-padded to rank 4 (repeat_interleave.cpp's
    // codegen dispatch helper); a wider rank has no representable padded value.
    if (ndim > 4) {
        return false;
    }
    // repeats == 1 is a no-op the native path answers without dispatching anything; repeats == 0
    // makes the output empty and the kernels have no zero-work path.
    if (repeats < 2) {
        return false;
    }
    // No zero-work path here either, so a zero-volume input stays on native.
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }

    const uint32_t nd = normalize_dim(dim, ndim);
    if (nd >= ndim) {
        return false;
    }

    // The dtypes the port is swept and certified against.
    const DataType dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {
        return false;
    }

    if (input.layout() == Layout::TILE) {
        // Both the host-side page map and the program factory's CB page size are built from the
        // 32x32 constants, so an off-default tile shape gives the kernels a page count and a page
        // size the buffer does not have. A transposed tile keeps both but swizzles the datums
        // within the page, and compute_output_specs() derives the output tile from the layout
        // alone, so the copied pages would come back labelled untransposed. Declining is not a
        // correctness guarantee for either case, only a refusal to claim support this factory
        // does not have.
        const auto tile = input.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
            tile.get_transpose_within_face() || tile.get_transpose_of_faces()) {
            return false;
        }
        // The last two dims subdivide a 32x32 tile, and the reader replicates whole tile pages:
        // page replication is not torch's element-level interleave along H or W.
        return nd < ndim - 2;
    }

    if (input.layout() == Layout::ROW_MAJOR) {
        // The RM reader copies whole sticks, so a stick has to hold at least two elements for the
        // interleave to be observable within it.
        if (shape[ndim - 1] < 2) {
            return false;
        }
        // The last dim lives within a stick rather than across sticks, so replicating it needs an
        // addressing scheme the RM reader here does not implement. Widening the scope means writing
        // that reader, not relaxing this clause.
        if (nd == ndim - 1) {
            return false;
        }
        // Nothing above bounds the stick width, but an RM CB slot is a whole stick in L1: reject a
        // stick too wide for the smallest viable CB so the case falls back to native instead of
        // TT_THROWing out of circular-buffer allocation at program-compile time.
        return rm_cb_budget(input, output_mem_config).max_slots >= kRmCbMinSlots;
    }

    return false;
}

bool is_demoted(
    const Tensor& /*input*/,
    uint32_t /*repeats*/,
    int32_t /*dim*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    // No shape is perf-demoted: every configuration supported_by_codegen() admits beats the native
    // composite on device time on both wormhole_b0 and blackhole. The gate stays in the routing
    // expression as the one place a genuine device regression belongs, expressed as a condition over
    // tensor attributes.
    return false;
}

}  // namespace ttnn::operations::data_movement::repeat_interleave_codegen
