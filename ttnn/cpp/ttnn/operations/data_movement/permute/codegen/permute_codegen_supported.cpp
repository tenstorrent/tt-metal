// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "permute_codegen_device_operation.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

namespace {

uint32_t page_alignment(const MemoryConfig& memory_config) {
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                                         : tt::tt_metal::hal::get_l1_alignment();
}

}  // namespace

bool is_row_invariant(ttsl::Span<const uint32_t> dims) {
    return !dims.empty() && dims[dims.size() - 1] == dims.size() - 1;
}

RmCbBudget rm_cb_budget(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t stick_bytes = shape[shape.rank() - 1] * input_tensor.element_size();
    const MemoryConfig& out_config = output_mem_config.value_or(input_tensor.memory_config());
    // The permutation leaves the last dim in place on this path, so both sides share one stick
    // width; only the per-buffer-type alignment differs, and one slot serves reader and writer.
    const uint32_t slot_stride = std::max(
        tt::round_up(stick_bytes, page_alignment(input_tensor.memory_config())),
        tt::round_up(stick_bytes, page_alignment(out_config)));

    auto* device = input_tensor.device();
    const uint32_t l1_capacity =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return {slot_stride, slot_stride == 0 ? 0 : l1_capacity / slot_stride};
}

bool supported_by_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config) {
    // Everything below asks about layout, dtype, shape and memory config, all of which a host
    // tensor answers too -- Tensor::is_sharded() is false off-device and memory_config() reads the
    // spec, not a buffer. Reject up front so a host tensor takes native's own validation instead of
    // tripping over Tensor::device() in rm_cb_budget() or a null buffer in the program factory.
    if (input_tensor.storage_type() != StorageType::DEVICE || input_tensor.buffer() == nullptr) {
        return false;
    }

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
    // sharded, so the routed entry has somewhere to fall back to.
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

    // The dtypes the row-major port is swept and certified against. bfloat8_b + ROW_MAJOR is also
    // independently invalid: the shared-exponent block-float layout has no row-major
    // representation.
    const DataType dtype = input_tensor.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {
        return false;
    }

    // The fused width-height permutation -- last axis moving to the second-to-last position, with
    // enough outer batch and tile-aligned H/W to make the fused transpose kernels worthwhile --
    // belongs to the transpose op's fused path, not to the row-invariant/blocked-generic kernels
    // here.
    if (dims[rank - 1] == rank - 2) {
        constexpr uint32_t kFusedMinNc = 6;
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

    // Nothing above bounds the stick width, but a row-invariant CB slot is a whole stick in L1:
    // reject a stick too wide for kRmCbSlots of them so the case falls back to native instead of
    // TT_THROWing out of circular-buffer allocation at program-compile time. Only the row-invariant
    // factory pages whole sticks; the blocked-generic one pages fixed 32-element chunks, so its
    // footprint does not scale with any tensor dimension.
    if (is_row_invariant(dims) && rm_cb_budget(input_tensor, output_mem_config).max_slots < kRmCbSlots) {
        return false;
    }

    return true;
}

bool is_demoted(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims) {
    // Perf-only demotion, consulted by the routed entry alone: the case stays correct and
    // supported, so a forced permute_force_codegen call still runs it.
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
    return !is_row_invariant(dims);
}

}  // namespace ttnn::operations::data_movement::permute_codegen
