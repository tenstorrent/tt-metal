// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"

#include <algorithm>
#include <tuple>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device) {
    return device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

namespace {

// Two-input width-dim demotion: the staged-copy volume past which codegen loses to native.
//
// Calibration: measured on Wormhole only. The predicate around it is arch-adaptive -- the grid
// comes from the live device, so max_sticks_per_core differs between a WH 8x8 and a BH 13x10 --
// but this byte threshold does not, and Blackhole has never been swept. It demotes the
// unaligned-width two-input class as a whole, including the cases inside it where codegen still
// wins, in exchange for never losing badly on the ones where the byte-copy loop dominates.
//
// Mechanism: reader_concat_rm_width_interleaved.cpp
// gates a batched direct-write fast path on both input rows filling their
// physical page pitch exactly and on input1's payload landing at an offset its
// own transport can address (kernel :60-68). Whichever side fails its condition
// stages instead: every stick of it goes through a scratch CB and is copied into
// the assembly page two bytes at a time by a volatile uint16_t RISC loop
// (kernel :102-145) -- no batching, no read/write overlap. Native's own fallback for the same unaligned-last-dim
// condition (concat.cpp's build_non_aligned_last_dim_concat) dispatches 4
// programs (transpose -> concat -> transpose). So this is fallback-vs-fallback:
// codegen wins on dispatch count while the staged-copy volume is small, and
// loses once the byte-copy loop dominates native's extra dispatches. The
// crossover is a volume threshold, not the alignment condition
// itself -- alignment only selects which regime (staged vs. fast-path) applies.
constexpr uint64_t kStagedCopyDemotionBytes = 2400;

bool rm_width_2in_unaligned_staged_copy_volume(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    if (input_tensors.size() != 2) {
        return false;
    }
    const Tensor& in0 = input_tensors[0];
    const Tensor& in1 = input_tensors[1];
    const uint32_t ndim = in0.logical_shape().rank();
    if (dim != ndim - 1) {
        return false;
    }

    // Mirror of the reader's compile-time path selection (kernel :60-68) over the same buffer
    // values create_descriptor_rm_width hands it. Input1's payload lands at input0's stick size
    // inside the assembly page, and IN1_NOC_ALIGNMENT is input1's own buffer alignment, so input1
    // writes directly only when its stick fills its page *and* input0's stick is a multiple of
    // that alignment. Two inputs may sit in different memory configurations -- only N > 2 is held
    // to one shared config -- so input0 filling its own page does not answer the second term.
    const auto* src0 = in0.buffer();
    const auto* src1 = in1.buffer();
    const uint32_t in0_stick = static_cast<uint32_t>(src0->page_size());
    const uint32_t in1_stick = static_cast<uint32_t>(src1->page_size());
    const uint32_t in1_noc_alignment = static_cast<uint32_t>(src1->alignment());
    const bool in0_direct = in0_stick == static_cast<uint32_t>(src0->aligned_page_size());
    const bool in1_direct =
        in1_stick == static_cast<uint32_t>(src1->aligned_page_size()) && in0_stick % in1_noc_alignment == 0;
    if (in0_direct && in1_direct) {
        // The reader takes the batched direct-write fast path, which wins at
        // every measured size in this regime.
        return false;
    }

    // Only the sides that miss the direct path pay the scratch-staged byte copy.
    const uint32_t staged_row_bytes = (in0_direct ? 0 : in0_stick) + (in1_direct ? 0 : in1_stick);

    tt::tt_metal::IDevice* device = in0.device();
    uint32_t total_out_sticks = 1;
    for (uint32_t i = 0; i + 1 < ndim; ++i) {
        total_out_sticks *= in0.logical_shape()[i];
    }
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto split = tt::tt_metal::split_work_to_cores(grid_size, total_out_sticks, /*row_wise=*/false);
    const uint32_t max_sticks_per_core = std::get<4>(split);

    const uint64_t staged_volume = static_cast<uint64_t>(max_sticks_per_core) * staged_row_bytes;
    return staged_volume >= kStagedCopyDemotionBytes;
}

bool dtype_in_scope(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::INT32 ||
           dtype == tt::tt_metal::DataType::UINT32;
}

// Mirrors reader_concat_rm_width_nway.cpp's runtime direct-write predicate: a
// stick whose physical page pitch equals its logical size has no trailing pad
// bytes to trim, and a destination offset that is a multiple of the shared
// transport alignment is a legal NOC endpoint. Every input sits in the same
// memory configuration (enforced above for N>2), so one buffer's alignment
// answers for all. Only when every input satisfies both does the reader take
// the batched-read fast path instead of the per-input scratch-staged byte copy.
bool width_nway_all_direct(const std::vector<Tensor>& input_tensors) {
    const uint32_t alignment = static_cast<uint32_t>(input_tensors[0].buffer()->alignment());
    uint32_t offset = 0;
    for (const auto& t : input_tensors) {
        auto* buf = t.buffer();
        const uint32_t stick_size = static_cast<uint32_t>(buf->page_size());
        const uint32_t page_size = static_cast<uint32_t>(buf->aligned_page_size());
        if (stick_size != page_size || offset % alignment != 0) {
            return false;
        }
        offset += stick_size;
    }
    return true;
}

}  // namespace

bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensors.size() < 2 || input_tensors.size() > ttnn::prim::kConcatMaxNwayInputs) {
        return false;
    }
    for (const auto& t : input_tensors) {
        // A host tensor answers every check below this loop -- layout(), dtype() and
        // memory_config() read the spec, not a buffer -- and then the CB-fit helpers those checks
        // end in dereference device() and buffer(). Decline up front so such a call reaches
        // native's own validation instead of a null deref inside a routing predicate.
        if (t.storage_type() != StorageType::DEVICE || t.buffer() == nullptr) {
            return false;
        }
        // No builder has a zero-work path: an empty input contributes a zero-length block that the
        // readers' block-cycling cursor cannot advance past, and a zero-width output turns the
        // stick count into a division by zero. Native answers these shapes.
        if (t.logical_shape().volume() == 0) {
            return false;
        }
        // Every stick computation here and in the kernels is in logical extents, but an
        // interleaved row-major buffer's page is padded_shape[-1] * element_size. When the two
        // differ the width builders would assemble padded sticks at logical offsets and write
        // silently wrong data, so decline rather than reinterpret.
        if (t.padded_shape() != t.logical_shape()) {
            return false;
        }
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }
    const Tensor& first = input_tensors[0];
    if (first.memory_config().is_sharded()) {
        return false;
    }
    if (first.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return false;
    }
    if (!dtype_in_scope(first.dtype())) {
        return false;
    }
    const auto dtype = first.dtype();
    const uint32_t ndim = first.logical_shape().rank();
    if (dim >= ndim) {
        return false;
    }
    for (const auto& t : input_tensors) {
        if (t.layout() != ttnn::ROW_MAJOR_LAYOUT || t.dtype() != dtype || t.memory_config().is_sharded() ||
            t.logical_shape().rank() != ndim) {
            return false;
        }
    }

    if (input_tensors.size() > 2) {
        // The N-way readers share one TensorAccessorArgs ABI across every
        // input, so every input must sit in the exact same memory configuration.
        const auto& first_mem = first.memory_config();
        for (const auto& t : input_tensors) {
            if (t.memory_config() != first_mem) {
                return false;
            }
        }
    }

    return ttnn::prim::plan_concat_cbs(input_tensors, dim, output_mem_config, usable_l1_bytes(first.device()))
        .has_value();
}

bool fits_live_l1(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    const Tensor& first = input_tensors[0];
    const uint64_t live = get_max_l1_space(first);
    // get_max_l1_space() is read before this call's own output is allocated, and the CBs are
    // sized from that output's page -- so the output's per-core footprint has to come off the
    // budget here or the plan is measured against L1 it will not have.
    const auto out_spec = ttnn::prim::concat_output_spec(input_tensors, dim, output_mem_config);
    const uint32_t pending_output = get_pending_l1_output_reservation(
        first,
        out_spec.padded_shape(),
        output_mem_config,
        first.dtype(),
        first.layout(),
        /*require_constructible=*/true);
    if (pending_output >= live) {
        return false;
    }
    return ttnn::prim::plan_concat_cbs(input_tensors, dim, output_mem_config, live - pending_output).has_value();
}

bool is_demoted(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    if (rm_width_2in_unaligned_staged_copy_volume(input_tensors, dim)) {
        return true;
    }
    // reader_concat_rm_width_nway.cpp now carries the same aligned direct-write
    // fast path as the two-input width reader (every read batched into the
    // reserved output page, one barrier, no copy) whenever every input's stick
    // fills its page exactly and lands at an offset that is a multiple of the
    // shared transport alignment. Only the remaining per-input scratch-staged
    // byte copy pays the measured regression, so demote width-dim N-way
    // concat only when that fallback would actually fire.
    const uint32_t ndim = input_tensors[0].logical_shape().rank();
    const bool is_width = (dim == ndim - 1);
    if (!is_width || input_tensors.size() <= 2) {
        return false;
    }
    return !width_nway_all_direct(input_tensors);
}

bool supported_execution_controls(unsigned int groups, const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    // sub_core_grids restricts native's default ConcatProgramFactory to a
    // caller-chosen core subset; no ConcatCodegen builder honours it (every
    // builder places work over the full compute_with_storage_grid_size()).
    // groups > 1 has no defined effect on native's own interleaved path
    // either (only the sharded factories consume it, and this port's scope
    // already excludes sharded), but ConcatCodegenParams carries no such
    // field, so route it to native defensively rather than silently drop it.
    return groups == 1 && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::concat_codegen
