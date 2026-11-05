// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Output CB depth is two batches, so reader and writer can each own one.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch) { return 2 * std::max(read_batch, write_batch); }

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;

// Batching the CB plan starts from; plan_batches halves it until the plan fits the L1 it is given. An empirical default
// from Blackhole runs, not derived from the memory system.
constexpr uint32_t kBatch = 4;

bool dtype_ok(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::INT32:
        case DataType::UINT32: return true;
        default: return false;
    }
}

ttnn::Shape concat_output_shape(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    ttnn::Shape shape = input_tensors.front().logical_shape();
    shape[dim] = 0;
    for (const Tensor& input : input_tensors) {
        shape[dim] += input.logical_shape()[dim];
    }
    return shape;
}

bool interleaved_in_dram_or_l1(const tt::tt_metal::MemoryConfig& config) {
    return config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
           (config.buffer_type() == BufferType::DRAM || config.buffer_type() == BufferType::L1);
}

// The factory and the kernels carry stick counts and stick byte sizes as 32-bit values, and cb_pages rounds the output
// stick up to an alignment in 32 bits, so a geometry past UINT32_MAX would wrap into a smaller, legal-looking plan.
bool fits_32_bit_geometry(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    const Tensor& first = input_tensors.front();
    const ttnn::Shape& first_shape = first.logical_shape();
    const uint32_t rank = first_shape.rank();
    uint64_t concat_extent = 0;
    for (const Tensor& input : input_tensors) {
        concat_extent += input.logical_shape()[dim];
    }
    if (concat_extent > UINT32_MAX) {
        return false;
    }
    uint64_t sticks = 1;
    for (uint32_t d = 0; d + 1 < rank; ++d) {
        sticks *= d == dim ? concat_extent : first_shape[d];
        if (sticks > UINT32_MAX) {
            return false;
        }
    }
    const uint64_t out_width = dim + 1 == rank ? concat_extent : first_shape[kLastAxis];
    const uint64_t alignment = std::max(tt::tt_metal::hal::get_l1_alignment(), tt::tt_metal::hal::get_dram_alignment());
    const uint64_t out_stick_bytes = out_width * first.element_size();
    return (out_stick_bytes + alignment - 1) / alignment * alignment <= UINT32_MAX;
}

bool inputs_supported(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    if (input_tensors.size() < 2 || input_tensors.size() > kConcatMaxNwayInputs) {
        return false;
    }
    const Tensor& first = input_tensors.front();
    // buffer() throws on an unallocated tensor instead of returning null, so allocation is tested first.
    if (first.storage_type() != ttnn::StorageType::DEVICE || !first.is_allocated()) {
        return false;
    }
    // Shared by routing and the forced and primitive entries, so none of them runs the kernels on an unsupported
    // architecture.
    if (!codegen_supported_on_arch(first)) {
        return false;
    }
    // The live-L1 planning below reads the local allocator, which a mesh with no local device does not have; the
    // launcher short-circuits such a mesh, so only the gate has to keep it on the native path.
    if (first.device()->is_remote_only()) {
        return false;
    }
    const uint32_t rank = first.logical_shape().rank();
    if (rank == 0 || attrs.dim >= rank) {
        return false;
    }
    for (const Tensor& input : input_tensors) {
        if (input.storage_type() != ttnn::StorageType::DEVICE || !input.is_allocated() ||
            input.device() != first.device()) {
            return false;
        }
        if (input.layout() != ttnn::ROW_MAJOR_LAYOUT || input.dtype() != first.dtype() || !dtype_ok(input.dtype())) {
            return false;
        }
        if (!interleaved_in_dram_or_l1(input.memory_config())) {
            return false;
        }
        // The kernels derive sticks from the logical shape, and an empty input leaves a core no work.
        if (input.padded_shape() != input.logical_shape() || input.logical_shape().volume() == 0 ||
            input.logical_shape().rank() != rank) {
            return false;
        }
        for (uint32_t d = 0; d < rank; ++d) {
            if (d != attrs.dim && input.logical_shape()[d] != first.logical_shape()[d]) {
                return false;
            }
        }
    }
    return fits_32_bit_geometry(input_tensors, attrs.dim);
}

uint64_t l1_output_reservation(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    return get_pending_l1_output_reservation(
        first, concat_output_shape(input_tensors, attrs.dim), attrs.output_mem_config, first.dtype(), first.layout());
}

uint64_t saturating_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

}  // namespace

bool concats_width(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    return attrs.dim + 1 == input_tensors.front().logical_shape().rank();
}

bool has_two_inputs(const std::vector<Tensor>& input_tensors) { return input_tensors.size() == 2; }

CbPages cb_pages(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    const auto out_shape = concat_output_shape(input_tensors, attrs.dim);
    const uint32_t out_stick = out_shape[kLastAxis] * first.element_size();

    // A slot offset inside the CB is a NoC destination for the reads, so it must be legal for every source's transport;
    // the writer's page pitch in the output buffer follows the output's transport alignment.
    uint32_t cb_alignment = tt::tt_metal::hal::get_l1_alignment();
    if (attrs.output_mem_config.buffer_type() == BufferType::DRAM) {
        cb_alignment = std::max(cb_alignment, tt::tt_metal::hal::get_dram_alignment());
    }
    uint32_t max_input_page = 0;
    for (const Tensor& input : input_tensors) {
        if (input.memory_config().buffer_type() == BufferType::DRAM) {
            cb_alignment = std::max(cb_alignment, tt::tt_metal::hal::get_dram_alignment());
        }
        max_input_page = std::max(max_input_page, static_cast<uint32_t>(input.buffer()->aligned_page_size()));
    }

    if (concats_width(input_tensors, attrs)) {
        // The two-input reader stages through a scratch page with one L1 alignment of slack.
        const uint32_t scratch_slack = has_two_inputs(input_tensors) ? tt::tt_metal::hal::get_l1_alignment() : 0;
        return CbPages{
            .out_page = tt::round_up(out_stick, cb_alignment), .scratch_page = max_input_page + scratch_slack};
    }
    return CbPages{.out_page = tt::round_up(std::max(out_stick, max_input_page), cb_alignment), .scratch_page = 0};
}

namespace {

uint64_t footprint_bytes(const CbPages& pages, const BatchPlan& batches) {
    const uint64_t out_bytes =
        static_cast<uint64_t>(cb_depth(batches.read_batch, batches.write_batch)) * pages.out_page;
    if (pages.scratch_page == 0) {
        return out_bytes;
    }
    // Every CB starts on the allocator's alignment, so the scratch CB sits behind the padding of the output CB.
    const uint64_t cb_start_alignment =
        std::max(tt::tt_metal::hal::get_l1_alignment(), tt::tt_metal::hal::get_dram_alignment());
    return tt::round_up(out_bytes, cb_start_alignment) + pages.scratch_page;
}

}  // namespace

uint64_t cb_footprint_bytes(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, const BatchPlan& batches) {
    return footprint_bytes(cb_pages(input_tensors, attrs), batches);
}

// The plan runs inside the per-call program hash, so the page geometry is derived once, not per halving step.
std::optional<BatchPlan> plan_batches(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, uint64_t budget) {
    const CbPages pages = cb_pages(input_tensors, attrs);
    BatchPlan batches{kBatch, kBatch};
    while (footprint_bytes(pages, batches) > budget) {
        if (batches.read_batch == 1 && batches.write_batch == 1) {
            return std::nullopt;
        }
        batches.read_batch = std::max(1u, batches.read_batch / 2);
        batches.write_batch = std::max(1u, batches.write_batch / 2);
    }
    return batches;
}

std::optional<BatchPlan> plan_batches_live(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, bool output_allocated) {
    // An allocated output has already lowered the live frontier; charging it again would shrink the plan.
    const uint64_t live = get_max_l1_space(input_tensors.front());
    const uint64_t budget = output_allocated ? live : saturating_sub(live, l1_output_reservation(input_tensors, attrs));
    return plan_batches(input_tensors, attrs, budget);
}

bool fits_live_l1(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    return plan_batches_live(input_tensors, attrs, /*output_allocated=*/false).has_value();
}

tt::tt_metal::TensorSpec codegen_output_spec(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const Tensor& first = input_tensors.front();
    return tt::tt_metal::TensorSpec(
        concat_output_shape(input_tensors, attrs.dim),
        tt::tt_metal::TensorLayout(
            first.dtype(),
            tt::tt_metal::PageConfig(first.layout(), first.tensor_spec().tile()),
            attrs.output_mem_config));
}

bool supported_by_codegen(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    if (!inputs_supported(input_tensors, attrs) || !interleaved_in_dram_or_l1(attrs.output_mem_config)) {
        return false;
    }
    // Static window: the output allocation between routing and validate would flip a live answer.
    const uint64_t budget =
        saturating_sub(get_static_l1_space(input_tensors.front()), l1_output_reservation(input_tensors, attrs));
    return plan_batches(input_tensors, attrs, budget).has_value();
}

namespace {

// The N-way width reader dispatches per input at runtime and builds a TensorAccessor per page, a fixed cost the
// reader pays once per call (estimated at roughly 80-85 ns) that only a longer stream of sticks per core hides. On
// Blackhole, N-way width concats of up to this many output sticks measured about 7-10% slower than native. The cutoff
// is an empirical Blackhole figure; no other architecture was measured.
constexpr uint32_t kMaxSticksForWidthNwayFixedCost = 32;
constexpr size_t kMinNwayInputs = 3;

// Wormhole N300 CI measured width concats (two-input and N-way) up to about 1.3x slower in device time than native,
// while its non-width concats ran 0.40-0.78x of native. Measured once, on one architecture, so the whole width class
// goes native there. Blackhole is gated only by the N-way cutoff below.
bool demoted_width_on_wormhole(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    const auto* device = input_tensors.front().device();
    return device != nullptr && device->arch() == tt::ARCH::WORMHOLE_B0 && concats_width(input_tensors, attrs);
}

}  // namespace

bool is_demoted(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs) {
    if (demoted_width_on_wormhole(input_tensors, attrs)) {
        return true;
    }
    if (input_tensors.size() < kMinNwayInputs || input_tensors.front().layout() != ttnn::ROW_MAJOR_LAYOUT ||
        !concats_width(input_tensors, attrs)) {
        return false;
    }
    const ttnn::Shape out_shape = concat_output_shape(input_tensors, attrs.dim);
    if (out_shape.volume() == 0 || out_shape[kLastAxis] == 0) {
        return false;
    }
    if (out_shape.volume() / out_shape[kLastAxis] > kMaxSticksForWidthNwayFixedCost) {
        return false;
    }
    // The fixed cost is per page and independent of dtype and of input widths.
    return true;
}

bool supported_execution_controls(
    unsigned int groups, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return groups == 1 && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::concat_codegen
