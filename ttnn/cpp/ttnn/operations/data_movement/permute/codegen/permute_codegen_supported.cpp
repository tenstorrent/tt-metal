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

bool is_permutation(ttsl::Span<const uint32_t> dims) {
    // The seen-set is a bitmask, so the rank has to fit its width. kMaxDims is far below 64, and a
    // rank that large is out of scope on its own.
    if (dims.empty() || dims.size() > 64) {
        return false;
    }
    uint64_t seen = 0;
    for (const uint32_t dim : dims) {
        if (dim >= dims.size()) {
            return false;
        }
        const uint64_t bit = uint64_t{1} << dim;
        if ((seen & bit) != 0) {
            return false;
        }
        seen |= bit;
    }
    return true;
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
    const uint32_t cb_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    // Static circular buffers grow up from the unreserved base and are rejected at program-creation
    // time if the region reaches the lowest live L1 buffer, not the architectural end of L1
    // (ProgramImpl::validate_circular_buffer_region). Budgeting against the full range would admit a
    // stick that fits an empty L1 and then TT_THROW out of allocation instead of falling back --
    // reachable here because this path holds kRmCbSlots whole sticks where native's row-major reader
    // holds two, so an L1-resident tensor leaves it four times as little room to spare.
    const auto lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(*lowest_occupied) : device->l1_size_per_core();
    const uint32_t l1_capacity = ceiling > cb_base ? ceiling - cb_base : 0;
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
    // A repeated axis survives ttnn::permute's per-axis normalization, and it is not benign here:
    // dims == [1, 1, 2] is row-invariant, so it selects the row-invariant factory, whose output
    // extents come from the permuted shape while its row count comes from the input -- for an input
    // whose leading dim exceeds its second the kernels then write more rows than the output holds.
    // Declining it hands the call to native, which keeps whatever behaviour it has for malformed
    // dims rather than the port inventing a new one.
    if (!is_permutation(dims)) {
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
    if (is_row_invariant(dims)) {
        // rm_cb_budget() reads the L1 that is occupied *now*, and the output buffer is allocated
        // after this gate returns -- an L1 output would lower the very frontier the budget measured,
        // so a stick admitted here could still collide with it. Estimating the output's per-bank
        // footprint would put an unverifiable number in the gate; declining the placement costs only
        // a fall back to native, and interleaved DRAM on both sides is the surface this path is
        // certified over.
        const MemoryConfig& out_config = output_mem_config.value_or(input_tensor.memory_config());
        if (out_config.buffer_type() != tt::tt_metal::BufferType::DRAM) {
            return false;
        }
        if (rm_cb_budget(input_tensor, output_mem_config).max_slots < kRmCbSlots) {
            return false;
        }
    }

    return true;
}

bool is_demoted(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims) {
    // Perf-only demotion, consulted by the routed entry alone: the case stays correct and
    // supported, so a forced permute_force_codegen call still runs it.
    //
    // A permutation that moves the last axis selects the blocked path (tilize -> transpose_tile ->
    // pack_untilize). That trip through the compute engine costs about what it saves, and the result
    // is not a uniform parity: over the 66 blocked configs of the swept surface, forced codegen
    // reaches 1.24x native device time on Blackhole while winning on 13 of them (best 1.12x), and on
    // Wormhole it is never worse than 1.03x and wins on 47 (best 1.26x). The spread straddles parity
    // on both, and differs enough between them that one measured range would not describe both.
    // The predicate stays unconditional and demotes the class as a whole: it trades the measured
    // wins away to avoid the measured losses, and native is what shipped for these configs, so a
    // demoted call is never a regression. The row-invariant path (last axis fixed, no compute,
    // batched stick reads) wins on every config and is what this port routes.
    const uint32_t rank = input_tensor.logical_shape().rank();
    if (rank < 2 || rank != dims.size()) {
        return false;
    }
    return !is_row_invariant(dims);
}

}  // namespace ttnn::operations::data_movement::permute_codegen
