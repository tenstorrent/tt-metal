// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_context_program_factory.hpp"

#include <algorithm>
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
    TT_FATAL(1 + n_rope <= grid.x * grid.y, "decode_step_context: not enough cores");
    auto core_at = [&](uint32_t i) { return CoreCoord{i % grid.x, i / grid.x}; };

    const CoreCoord int_core = core_at(0);
    std::vector<CoreCoord> rope_cores;
    for (uint32_t j = 0; j < n_rope; ++j) {
        rope_cores.push_back(core_at(1 + j));
    }
    std::set<CoreRange> all_ranges{CoreRange(int_core)};
    std::set<CoreRange> rope_ranges;
    for (const auto& c : rope_cores) {
        all_ranges.insert(CoreRange(c));
        rope_ranges.insert(CoreRange(c));
    }
    const CoreRangeSet all_cores(all_ranges);

    const uint32_t pkt_bytes = pkt.logical_shape()[3] * sizeof(int32_t);
    uint32_t max_rd = tt::constants::TILE_WIDTH;
    for (const auto& t : args.rope_tables) {
        max_rd = std::max(max_rd, static_cast<uint32_t>(t.logical_shape()[3] / 2));
    }
    const uint32_t max_tiles = max_rd / tt::constants::TILE_WIDTH;
    add_cb(program, all_cores, kCbPkt, tt::round_up(pkt_bytes, kSlotBytes));
    add_cb(program, all_cores, kCbScratch, std::max(n_int, 1u) * kSlotBytes);
    add_cb(program, all_cores, kCbRows, 2 * tt::round_up(2 * max_rd * sizeof(int32_t), kSlotBytes));
    add_cb(program, all_cores, kCbTiles, 3 * max_tiles * kBf16TileBytes);

    const Buffer* int_out = outputs[0].buffer();
    const Buffer* table_buf = n_rope ? args.rope_tables[0].buffer() : pkt.buffer();
    const Buffer* rope_out = n_rope ? outputs[n_int].buffer() : int_out;
    auto ct_args = [&](uint32_t role) {
        std::vector<uint32_t> ct = {role, kCbPkt, kCbScratch, kCbRows, kCbTiles, pkt_bytes};
        TensorAccessorArgs(pkt.buffer()).append_to(ct);
        TensorAccessorArgs(int_out).append_to(ct);
        TensorAccessorArgs(table_buf).append_to(ct);
        TensorAccessorArgs(rope_out).append_to(ct);
        return ct;
    };
    const KernelHandle int_kernel_id =
        CreateKernel(program, kKernelPath, CoreRangeSet(CoreRange(int_core)), ReaderDataMovementConfig(ct_args(0)));
    KernelHandle rope_kernel_id = 0;
    if (n_rope) {
        rope_kernel_id =
            CreateKernel(program, kKernelPath, CoreRangeSet(rope_ranges), ReaderDataMovementConfig(ct_args(1)));
    }

    std::vector<uint32_t> int_args = {pkt.buffer()->address(), n_int};
    std::vector<uint32_t> int_addr_slots;
    for (uint32_t i = 0; i < n_int; ++i) {
        const auto& terms = attrs.int_terms[i];
        int_addr_slots.push_back(int_args.size());
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
    SetRuntimeArgs(program, int_kernel_id, int_core, int_args);

    for (uint32_t j = 0; j < n_rope; ++j) {
        const auto& table = args.rope_tables[attrs.rope_tables[j]];
        const uint32_t rows = table.logical_shape()[2];
        const uint32_t rd = table.logical_shape()[3] / 2;
        SetRuntimeArgs(
            program,
            rope_kernel_id,
            rope_cores[j],
            {pkt.buffer()->address(),
             table.buffer()->address(),
             rows - kRopeLoRows,
             rd,
             attrs.rope_fields[j],
             static_cast<uint32_t>(attrs.rope_offsets[j]),
             outputs[n_int + 3 * j].buffer()->address(),
             outputs[n_int + 3 * j + 1].buffer()->address(),
             outputs[n_int + 3 * j + 2].buffer()->address()});
    }

    return cached_program_t{
        std::move(program),
        {.int_kernel_id = int_kernel_id,
         .rope_kernel_id = rope_kernel_id,
         .int_core = int_core,
         .rope_cores = rope_cores,
         .int_addr_slots = int_addr_slots}};
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

    auto& int_args = GetRuntimeArgs(program, shared.int_kernel_id, shared.int_core);
    int_args[0] = pkt_addr;
    for (uint32_t i = 0; i < n_int; ++i) {
        int_args[shared.int_addr_slots[i]] = outputs[i].buffer()->address();
    }
    for (uint32_t j = 0; j < shared.rope_cores.size(); ++j) {
        auto& a = GetRuntimeArgs(program, shared.rope_kernel_id, shared.rope_cores[j]);
        a[0] = pkt_addr;
        a[1] = args.rope_tables[attrs.rope_tables[j]].buffer()->address();
        a[6] = outputs[n_int + 3 * j].buffer()->address();
        a[7] = outputs[n_int + 3 * j + 1].buffer()->address();
        a[8] = outputs[n_int + 3 * j + 2].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::step_context
