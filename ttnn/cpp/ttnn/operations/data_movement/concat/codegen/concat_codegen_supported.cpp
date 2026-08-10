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

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

namespace {

// concat.yaml's rm-width-2in-unaligned-staged-copy-volume predicate.
//
// Mechanism (concat.yaml demotion block): reader_concat_rm_width_interleaved.cpp
// gates a batched direct-write fast path on both input rows filling their
// physical page pitch exactly (kernel :60-68). When that fails, every stick is
// staged through a scratch CB and copied into the assembly page two bytes at a
// time by a volatile uint16_t RISC loop (kernel :102-145) -- no batching, no
// read/write overlap. Native's own fallback for the same unaligned-last-dim
// condition (concat.cpp's build_non_aligned_last_dim_concat) dispatches 4
// programs (transpose -> concat -> transpose). So this is fallback-vs-fallback:
// codegen wins on dispatch count while the staged-copy volume is small, and
// loses once the byte-copy loop dominates native's extra dispatches. The
// crossover is a volume threshold (S >= 2400 B), not the alignment condition
// itself -- alignment only selects which regime (staged vs. fast-path) applies.
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

    auto row_bytes = [](const Tensor& t) {
        return static_cast<uint32_t>(t.logical_shape()[-1]) * static_cast<uint32_t>(t.element_size());
    };
    auto row_bytes_aligned = [](const Tensor& t, uint32_t bytes) {
        return bytes % static_cast<uint32_t>(t.buffer()->alignment()) == 0;
    };

    const uint32_t in0_row_bytes = row_bytes(in0);
    const uint32_t in1_row_bytes = row_bytes(in1);
    const bool in0_aligned = row_bytes_aligned(in0, in0_row_bytes);
    const bool in1_aligned = row_bytes_aligned(in1, in1_row_bytes);
    if (in0_aligned && in1_aligned) {
        // Every input's row bytes is an alignment multiple: the reader takes
        // the batched direct-write fast path, which wins at every measured
        // size in this regime.
        return false;
    }

    // in0 unaligned implies IN0_STICK_SIZE % IN1_NOC_ALIGNMENT != 0 (kernel's
    // in1_direct condition), so input1 cannot be direct either: both inputs
    // stage and the staged row equals the full output row. Otherwise only
    // input1 stages.
    const uint32_t staged_row_bytes = in0_aligned ? in1_row_bytes : (in0_row_bytes + in1_row_bytes);

    tt::tt_metal::IDevice* device = in0.device();
    uint32_t total_out_sticks = 1;
    for (uint32_t i = 0; i + 1 < ndim; ++i) {
        total_out_sticks *= in0.logical_shape()[i];
    }
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto split = tt::tt_metal::split_work_to_cores(grid_size, total_out_sticks, /*row_wise=*/false);
    const uint32_t max_sticks_per_core = std::get<4>(split);

    const uint64_t staged_volume = static_cast<uint64_t>(max_sticks_per_core) * staged_row_bytes;
    return staged_volume >= 2400;
}

bool dtype_in_scope(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::INT32 ||
           dtype == tt::tt_metal::DataType::UINT32;
}

// Projected per-core CB fit for the non-width RM builders (build_concat_rm /
// build_concat_rm_nonwidth_nway): one CB sized to the largest of every
// input's and the projected output's aligned RM page. Mirrors
// concat_codegen_program_factory.cpp's create_descriptor_rm{,_nonwidth_nway}
// so the gate and the factory cannot drift.
bool nonwidth_cb_fits(const std::vector<Tensor>& input_tensors, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::IDevice* device = input_tensors[0].device();
    const uint32_t stick_size = input_tensors[0].logical_shape()[-1] * input_tensors[0].element_size();
    const uint32_t out_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    uint32_t cb_page = tt::align(stick_size, out_alignment);
    for (const auto& t : input_tensors) {
        cb_page = std::max(cb_page, static_cast<uint32_t>(t.buffer()->aligned_page_size()));
    }
    return ttnn::prim::plan_concat_cb(cb_page, ttnn::prim::kConcatNonWidthBatch, ttnn::prim::concat_l1_budget(device))
        .has_value();
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

// Projected per-core CB fit for the width RM builders (build_concat_rm_width /
// build_concat_rm_width_nway): the write-batched output CB plus a fixed-size
// scratch CB. Mirrors concat_codegen_program_factory.cpp's
// create_descriptor_rm_width{,_nway}.
bool width_cb_fits(const std::vector<Tensor>& input_tensors, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::IDevice* device = input_tensors[0].device();
    uint32_t out_width = 0;
    uint32_t scratch_page = 0;
    for (const auto& t : input_tensors) {
        out_width += t.logical_shape()[-1];
        scratch_page = std::max(scratch_page, static_cast<uint32_t>(t.buffer()->aligned_page_size()));
    }
    if (input_tensors.size() == 2) {
        // build_concat_rm_width's scratch CB carries an extra L1-granularity
        // margin the 2-tensor kernel needs; the N-way kernel does not.
        scratch_page += device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    }
    const uint32_t out_stick = out_width * input_tensors[0].element_size();
    const uint32_t out_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    const uint32_t out_page = tt::align(out_stick, out_alignment);

    const uint64_t l1_budget = ttnn::prim::concat_l1_budget(device);
    if (scratch_page > l1_budget) {
        return false;
    }
    return ttnn::prim::plan_concat_cb(out_page, ttnn::prim::kConcatWidthWriteBatch, l1_budget - scratch_page)
        .has_value();
}

}  // namespace

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
    TT_THROW("Unknown concat implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensors.size() < 2 || input_tensors.size() > ttnn::prim::kConcatMaxNwayInputs) {
        return false;
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
        // input (spec.py's build_concat_rm_{width,nonwidth}_nway), so every
        // input must sit in the exact same memory configuration.
        const auto& first_mem = first.memory_config();
        for (const auto& t : input_tensors) {
            if (t.memory_config() != first_mem) {
                return false;
            }
        }
    }

    const bool is_width = (dim == ndim - 1);
    return is_width ? width_cb_fits(input_tensors, output_mem_config)
                    : nonwidth_cb_fits(input_tensors, output_mem_config);
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
    // byte copy pays the measured regression (phase-7 seeds: {32,32}/{32,64}/
    // {32,96} dim=-1; {1,32,32}/{1,32,64}/{1,32,32} dim=2; {1,1,32,32}x3
    // dim=-1 -- all of them 64 B-aligned Blackhole sticks that, prior to the
    // kernel fix, fell through to the byte-copy path unconditionally). Demote
    // width-dim N-way concat only when that fallback would actually fire.
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
