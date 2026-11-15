// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_context_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::step_context {

namespace {

using namespace tt::tt_metal;

constexpr uint32_t kCbPkt = tt::CBIndex::c_0;
constexpr uint32_t kCbScratch = tt::CBIndex::c_1;
constexpr uint32_t kCbRows = tt::CBIndex::c_2;
constexpr uint32_t kCbTiles = tt::CBIndex::c_3;
constexpr uint32_t kSlotBytes = 64;
constexpr uint32_t kBf16TileBytes = tt::constants::TILE_HW * 2;

constexpr char kKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/step_context/device/kernels/dataflow/step_context.cpp";

void add_cb(Program& program, const CoreRangeSet& cores, uint32_t index, uint32_t bytes) {
    CreateCircularBuffer(
        program, cores, CircularBufferConfig(bytes, {{index, tt::DataFormat::UInt32}}).set_page_size(index, bytes));
}

}  // namespace

StepContextProgramFactory::cached_program_t StepContextProgramFactory::create(
    const StepContextParams& attrs, const StepContextInputs& args, StepContextTensorReturn& outputs) {
    Program program{};
    const auto& pkt = args.packet;
    auto* device = pkt.device();
    const auto grid = device->compute_with_storage_grid_size();

    const uint32_t n_int = attrs.int_shapes.size();
    const uint32_t n_rope = attrs.rope_tables.size();

    // Integer outputs in order, kIntElemsPerCore elements per core at most (an output stays on one core).
    std::vector<uint32_t> int_first;
    uint32_t elems = 0;
    for (uint32_t i = 0; i < n_int; ++i) {
        const uint32_t n = attrs.int_terms[i].size() / kTermWords;
        if (int_first.empty() || elems + n > kIntElemsPerCore) {
            int_first.push_back(i);
            elems = 0;
        }
        elems += n;
    }
    const uint32_t n_int_cores = int_first.size();
    std::vector<uint32_t> rope_output;
    for (uint32_t j = 0; j < n_rope; ++j) {
        for (uint32_t r = 0; r < rope_rows_of(attrs, j); ++r) {
            rope_output.push_back(j);
        }
    }
    TT_FATAL(n_int_cores + rope_output.size() <= grid.x * grid.y, "decode_step_context: not enough cores");
    auto core_at = [&](uint32_t i) { return CoreCoord{i % grid.x, i / grid.x}; };

    std::vector<CoreCoord> int_cores;
    std::set<CoreRange> int_ranges;
    for (uint32_t c = 0; c < n_int_cores; ++c) {
        int_cores.push_back(core_at(c));
        int_ranges.insert(CoreRange(int_cores.back()));
    }
    std::vector<CoreCoord> rope_cores;
    std::set<CoreRange> rope_ranges;
    for (uint32_t i = 0; i < rope_output.size(); ++i) {
        rope_cores.push_back(core_at(n_int_cores + i));
        rope_ranges.insert(CoreRange(rope_cores.back()));
    }
    std::set<CoreRange> all_ranges = int_ranges;
    all_ranges.insert(rope_ranges.begin(), rope_ranges.end());
    const CoreRangeSet all_cores(all_ranges);

    const uint32_t pkt_bytes = pkt.logical_shape()[3] * sizeof(int32_t);
    uint32_t max_rd = tt::constants::TILE_WIDTH;
    for (const auto& t : args.rope_tables) {
        max_rd = std::max(max_rd, static_cast<uint32_t>(t.logical_shape()[3] / 2));
    }
    const uint32_t max_tiles = max_rd / tt::constants::TILE_WIDTH;
    uint32_t max_core_outputs = 1;
    for (uint32_t c = 0; c < n_int_cores; ++c) {
        const uint32_t next = c + 1 < n_int_cores ? int_first[c + 1] : n_int;
        max_core_outputs = std::max(max_core_outputs, next - int_first[c]);
    }
    add_cb(program, all_cores, kCbPkt, tt::round_up(pkt_bytes, kSlotBytes));
    add_cb(program, all_cores, kCbScratch, max_core_outputs * kSlotBytes);
    add_cb(program, all_cores, kCbRows, 2 * tt::round_up(2 * max_rd * sizeof(int32_t), kSlotBytes));
    // Three outputs of (tile height x Rd) bf16 each: at most three 32 x 32 tiles per 32 columns.
    add_cb(program, all_cores, kCbTiles, 3 * max_tiles * kBf16TileBytes);
    const uint32_t sem = n_rope ? CreateSemaphore(program, CoreRangeSet(rope_ranges), 0) : 0;

    const Buffer* int_out = outputs[0].buffer();
    const Buffer* table_buf = n_rope ? args.rope_tables[0].buffer() : pkt.buffer();
    const Buffer* rope_out = n_rope ? outputs[n_int].buffer() : int_out;
    auto ct_args = [&](uint32_t role) {
        std::vector<uint32_t> ct = {role, kCbPkt, kCbScratch, kCbRows, kCbTiles, pkt_bytes, sem};
        TensorAccessorArgs(pkt.buffer()).append_to(ct);
        TensorAccessorArgs(int_out).append_to(ct);
        TensorAccessorArgs(table_buf).append_to(ct);
        TensorAccessorArgs(rope_out).append_to(ct);
        return ct;
    };
    const KernelHandle int_kernel_id =
        CreateKernel(program, kKernelPath, CoreRangeSet(int_ranges), ReaderDataMovementConfig(ct_args(0)));
    KernelHandle rope_kernel_id = 0;
    if (n_rope) {
        rope_kernel_id =
            CreateKernel(program, kKernelPath, CoreRangeSet(rope_ranges), ReaderDataMovementConfig(ct_args(1)));
    }

    std::vector<std::vector<uint32_t>> int_addr_slots(n_int_cores);
    for (uint32_t c = 0; c < n_int_cores; ++c) {
        const uint32_t next = c + 1 < n_int_cores ? int_first[c + 1] : n_int;
        std::vector<uint32_t> int_args = {pkt.buffer()->address(), next - int_first[c]};
        for (uint32_t i = int_first[c]; i < next; ++i) {
            const auto& terms = attrs.int_terms[i];
            int_addr_slots[c].push_back(int_args.size());
            int_args.push_back(outputs[i].buffer()->address());
            int_args.push_back(terms.size() / kTermWords);
            for (int32_t w : terms) {
                int_args.push_back(static_cast<uint32_t>(w));
            }
        }
        TT_FATAL(
            int_args.size() <= tt::tt_metal::max_runtime_args,
            "decode_step_context: {} integer-term args exceed the {} runtime args of a kernel",
            int_args.size(),
            tt::tt_metal::max_runtime_args);
        SetRuntimeArgs(program, int_kernel_id, int_cores[c], int_args);
    }

    // Row r of RoPE output j runs on its own core; rows 1.. send theirs to row 0's core, which writes the tiles.
    uint32_t gather = 0;
    for (uint32_t i = 0; i < rope_cores.size(); ++i) {
        const uint32_t j = rope_output[i];
        if (i == 0 || rope_output[i - 1] != j) {
            gather = i;
        }
        const uint32_t r = i - gather;
        const auto& table = args.rope_tables[attrs.rope_tables[j]];
        const uint32_t rows = table.logical_shape()[2];
        const uint32_t rd = table.logical_shape()[3] / 2;
        const CoreCoord hub = device->worker_core_from_logical_core(rope_cores[gather]);
        SetRuntimeArgs(
            program,
            rope_kernel_id,
            rope_cores[i],
            {pkt.buffer()->address(),
             table.buffer()->address(),
             rows - kRopeLoRows,
             rd,
             attrs.rope_fields[j] + r,
             static_cast<uint32_t>(attrs.rope_offsets[j]),
             outputs[n_int + 3 * j].buffer()->address(),
             outputs[n_int + 3 * j + 1].buffer()->address(),
             outputs[n_int + 3 * j + 2].buffer()->address(),
             r,
             rope_rows_of(attrs, j),
             rope_tile_height_of(attrs, j),
             static_cast<uint32_t>(hub.x),
             static_cast<uint32_t>(hub.y)});
    }

    return cached_program_t{
        std::move(program),
        {.int_kernel_id = int_kernel_id,
         .rope_kernel_id = rope_kernel_id,
         .int_cores = int_cores,
         .int_first = int_first,
         .int_addr_slots = int_addr_slots,
         .rope_cores = rope_cores,
         .rope_output = rope_output}};
}

void StepContextProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const StepContextParams& attrs,
    const StepContextInputs& args,
    StepContextTensorReturn& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const uint32_t pkt_addr = args.packet.buffer()->address();
    const uint32_t n_int = attrs.int_shapes.size();

    for (uint32_t c = 0; c < shared.int_cores.size(); ++c) {
        auto& int_args = GetRuntimeArgs(program, shared.int_kernel_id, shared.int_cores[c]);
        int_args[0] = pkt_addr;
        for (uint32_t s = 0; s < shared.int_addr_slots[c].size(); ++s) {
            int_args[shared.int_addr_slots[c][s]] = outputs[shared.int_first[c] + s].buffer()->address();
        }
    }
    for (uint32_t i = 0; i < shared.rope_cores.size(); ++i) {
        const uint32_t j = shared.rope_output[i];
        auto& a = GetRuntimeArgs(program, shared.rope_kernel_id, shared.rope_cores[i]);
        a[0] = pkt_addr;
        a[1] = args.rope_tables[attrs.rope_tables[j]].buffer()->address();
        a[6] = outputs[n_int + 3 * j].buffer()->address();
        a[7] = outputs[n_int + 3 * j + 1].buffer()->address();
        a[8] = outputs[n_int + 3 * j + 2].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::step_context
