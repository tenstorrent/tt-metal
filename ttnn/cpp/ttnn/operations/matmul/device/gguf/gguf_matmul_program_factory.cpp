// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t kMaxPartials = 7;    // a finishing core adds its fp32 partial and up to 7 others in DEST's 8 tiles
constexpr uint32_t kInputChunk = 16;    // input tiles per multicast
constexpr uint32_t kGroupsPerRead = 4;  // one-page groups: records per unit of reads
constexpr uint32_t kPageUnits = 4;      // units of pages c_0 holds
constexpr uint32_t kRecordSlots = 3;    // multi-page groups: records staged in L1, two of them in flight from DRAM
constexpr uint32_t kMaxSegments = 64;   // keeps each kernel's runtime arguments within the limit

constexpr auto kKernels = "ttnn/cpp/ttnn/operations/matmul/device/kernels/gguf/";

// Compute order of a core's work on one N tile; must match gguf_matmul_compute.cpp and gguf_matmul_writer.cpp.
enum Role : uint32_t {
    kWhole = 0,   // all K groups: bf16 output tile
    kTail = 1,    // later K groups: fp32 partial to the finishing core
    kHead = 2,    // first K groups: fp32 partial kept on this core
    kFinish = 3,  // no K work: the kept partial plus the others, bf16 output tile
};

struct Segment {
    uint32_t n;       // N tile
    uint32_t g0, g1;  // K groups [g0, g1)
    Role role;
    uint32_t partner = 0;  // kTail: the plan core finishing the N tile
    uint32_t slot = 0;     // kTail: its partial's slot there
    uint32_t parts = 0;    // kFinish: partials from other cores
};

// The n_tiles x groups (N tile, K group) items in N-tile order, split evenly over at most `cores` cores: per core its
// segments in compute order. Compute order: the head of its last N tile (from K group 0, so it runs while the input
// streams in), the whole N tiles, the tail of its first N tile, the finish of its last. A finish waits only for
// tails, which wait for nothing, so no chain of waits forms across cores. Fewer cores until no N tile spans more than
// kMaxPartials + 1 of them.
std::vector<std::vector<Segment>> plan_segments(uint32_t n_tiles, uint32_t groups, uint32_t cores) {
    const uint32_t items = n_tiles * groups;
    for (uint32_t c = std::min(cores, items);; c--) {
        std::vector<std::vector<Segment>> plan(c);
        std::map<uint32_t, uint32_t> owner, parts;
        for (uint32_t i = 0; i < c; i++) {
            const uint32_t a = uint64_t(i) * items / c, b = uint64_t(i + 1) * items / c;
            std::vector<Segment> head, whole, tail;
            for (uint32_t n = a / groups; n <= (b - 1) / groups; n++) {
                const uint32_t g0 = std::max(a, n * groups) - n * groups;
                const uint32_t g1 = std::min(b, (n + 1) * groups) - n * groups;
                if (g0 == 0 && g1 == groups) {
                    whole.push_back({n, 0, groups, kWhole});
                } else if (g0 == 0) {
                    head = {{n, 0, g1, kHead}, {n, 0, 0, kFinish}};
                    owner[n] = i;
                } else {
                    tail = {{n, g0, g1, kTail, owner.at(n), parts[n]++}};
                }
            }
            if (!head.empty()) {
                plan[i].push_back(head[0]);
            }
            plan[i].insert(plan[i].end(), whole.begin(), whole.end());
            plan[i].insert(plan[i].end(), tail.begin(), tail.end());
            if (!head.empty()) {
                plan[i].push_back(head[1]);
            }
        }
        uint32_t most = 0;
        for (const auto& [n, count] : parts) {
            most = std::max(most, count);
        }
        if (most <= kMaxPartials) {
            for (auto& segments : plan) {
                for (Segment& s : segments) {
                    if (s.role == kFinish) {
                        s.parts = parts[s.n];
                    }
                }
            }
            return plan;
        }
    }
}

CoreRangeSet core_ranges(const std::vector<CoreCoord>& cores) {
    CoreRangeSet ranges;
    for (const CoreCoord& c : cores) {
        ranges = ranges.merge(CoreRangeSet(CoreRange(c)));
    }
    return ranges;
}

void add_cb(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex index,
    tt::DataFormat format,
    uint32_t page_bytes,
    uint32_t pages,
    std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
    CircularBufferConfig config =
        CircularBufferConfig(page_bytes * pages, {{index, format}}).set_page_size(index, page_bytes);
    if (tile) {
        config.set_tile_dims(index, *tile);
    }
    CreateCircularBuffer(program, cores, config);
}

}  // namespace

GgufMatmulProgramFactory::cached_program_t GgufMatmulProgramFactory::create(
    const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const Tensor& input = inputs.input;
    const Tensor& weight = inputs.weight;
    auto* device = input.device();
    const GgufTraits& f = gguf_traits(weight.dtype());
    const uint32_t k_tiles = weight.logical_shape()[0] / 32, n_tiles = weight.logical_shape()[1] / 32;
    const uint32_t groups = k_tiles / f.group_tiles, pages_per_group = f.group_tiles / f.page_tiles;
    const uint32_t rows = input.logical_shape().volume() / input.logical_shape()[-1];
    const uint32_t in0_rows = rows > 16 ? 32 : rows > 8 ? 16 : 8;

    // N tiles in bank order when each N tile's records sit in one DRAM bank (page g * n_tiles + n of an interleaved
    // buffer is in bank page % banks), so that a run of a core's items is in one bank.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t banks = device->allocator()->get_num_banks(BufferType::DRAM);
    const bool bank_local = n_tiles % banks == 0;
    std::vector<uint32_t> order(n_tiles);
    for (uint32_t j = 0; j < n_tiles; j++) {
        order[j] = bank_local ? j % (n_tiles / banks) * banks + j / (n_tiles / banks) : j;
    }
    std::vector<std::vector<Segment>> plan = plan_segments(n_tiles, groups, grid.x * grid.y);
    for (auto& segments : plan) {
        TT_FATAL(segments.size() <= kMaxSegments, "GGUF matmul: {} segments on a core", segments.size());
        for (Segment& s : segments) {
            s.n = order[s.n];
        }
    }

    // Plan core i runs on cores[i]: row-major, or with bank-local records the free core nearest the banks of its items
    // (Manhattan distance from each bank's adjacent worker, weighted by the core's groups there), cheapest pairs first.
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < plan.size(); i++) {
        cores.emplace_back(i % grid.x, i / grid.x);
    }
    if (bank_local) {
        const auto adjacent =
            device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0, distributed::MeshCoordinate(0, 0));
        std::vector<CoreCoord> all;
        for (uint32_t i = 0; i < grid.x * grid.y; i++) {
            all.emplace_back(i % grid.x, i / grid.x);
        }
        std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> pairs;  // cost, plan core, core
        for (uint32_t i = 0; i < plan.size(); i++) {
            for (uint32_t c = 0; c < all.size(); c++) {
                uint64_t cost = 0;
                for (const Segment& s : plan[i]) {
                    const CoreCoord& a = adjacent.at(s.n % banks);
                    cost += uint64_t(s.g1 - s.g0) *
                            (std::abs(int(all[c].x) - int(a.x)) + std::abs(int(all[c].y) - int(a.y)));
                }
                pairs.emplace_back(cost, i, c);
            }
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<bool> placed(plan.size(), false), taken(all.size(), false);
        for (const auto& [cost, i, c] : pairs) {
            if (!placed[i] && !taken[c]) {
                cores[i] = all[c];
                placed[i] = taken[c] = true;
            }
        }
    }
    std::vector<CoreCoord> virt;
    for (const CoreCoord& c : cores) {
        virt.push_back(device->worker_core_from_logical_core(c));
    }
    uint32_t most_parts = 1;
    for (const auto& segments : plan) {
        for (const Segment& s : segments) {
            most_parts = std::max(most_parts, s.parts);
        }
    }

    Program program = CreateProgram();
    const CoreRangeSet full(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    const CoreRangeSet compute = core_ranges(cores);

    // c_1, the input, sits at the same address on every core of the grid: the collector multicasts it there.
    add_cb(program, full, tt::CBIndex::c_1, tt::DataFormat::Float16_b, 2048, k_tiles);
    const uint32_t l1_page = f.l1_page_bytes();
    const tt::tt_metal::Tile page_tile(
        l1_page == 1024   ? std::array<uint32_t, 2>{16, 16}
        : l1_page == 2048 ? std::array<uint32_t, 2>{16, 32}
                          : std::array<uint32_t, 2>{32, 32});
    const uint32_t unit = pages_per_group > 1 ? pages_per_group : kGroupsPerRead;
    add_cb(program, compute, tt::CBIndex::c_0, tt::DataFormat::UInt32, l1_page, kPageUnits * unit, page_tile);
    if (pages_per_group > 1) {
        add_cb(program, compute, tt::CBIndex::c_2, tt::DataFormat::UInt32, kRecordSlots * f.page_bytes(), 1);
    }
    add_cb(program, compute, tt::CBIndex::c_16, tt::DataFormat::Float16_b, 2048, 2);
    add_cb(program, compute, tt::CBIndex::c_17, tt::DataFormat::Float32, 4096, 1);
    add_cb(program, compute, tt::CBIndex::c_18, tt::DataFormat::UInt32, 4096, most_parts, tt::tt_metal::Tile());
    add_cb(program, compute, tt::CBIndex::c_20, tt::DataFormat::UInt32, 4096, 1, tt::tt_metal::Tile());
    const uint32_t input_sem = CreateSemaphore(program, full, 0);
    const uint32_t partial_sem = CreateSemaphore(program, compute, 0);

    // The collector: the core nearest the middle of the grid whose writer runs on NoC 0 (odd column), if any. The
    // multicast rectangle is the whole grid, corners in the writer's NoC order.
    size_t collector = 0;
    uint32_t best = UINT32_MAX;
    for (size_t i = 0; i < cores.size(); i++) {
        const uint32_t d = std::abs(int(cores[i].x) - int(grid.x / 2)) + std::abs(int(cores[i].y) - int(grid.y / 2));
        if (cores[i].x % 2 == 1 && d < best) {
            collector = i;
            best = d;
        }
    }
    const CoreCoord v0 = device->worker_core_from_logical_core(CoreCoord(0, 0));
    const CoreCoord v1 = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, grid.y - 1));
    const std::array<std::array<uint32_t, 4>, 2> rect = {
        {{uint32_t(v1.x), uint32_t(v1.y), uint32_t(v0.x), uint32_t(v0.y)},
         {uint32_t(v0.x), uint32_t(v0.y), uint32_t(v1.x), uint32_t(v1.y)}}};

    std::vector<uint32_t> reader_ct = {f.format_id, kPageUnits * unit, kGroupsPerRead, kRecordSlots};
    TensorAccessorArgs(*weight.buffer()).append_to(reader_ct);
    const uint32_t chunk = std::min(kInputChunk, k_tiles);
    std::vector<uint32_t> writer_ct = {k_tiles, chunk, input_sem, partial_sem};
    TensorAccessorArgs(*input.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    // The weight reader of a core uses one NoC and its writer the other, alternating by column so that both NoCs carry
    // DRAM reads.
    shared_variables_t shared;
    shared.cores = cores;
    for (uint32_t parity = 0; parity < 2; parity++) {
        std::vector<CoreCoord> sub;
        for (const CoreCoord& c : cores) {
            if (c.x % 2 == parity) {
                sub.push_back(c);
            }
        }
        if (sub.empty()) {
            continue;
        }
        const CoreRangeSet ranges = core_ranges(sub);
        shared.reader[parity] = CreateKernel(
            program,
            std::string(kKernels) + "gguf_matmul_reader.cpp",
            ranges,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = parity ? NOC::NOC_1 : NOC::NOC_0,
                .compile_args = reader_ct});
        shared.writer[parity] = CreateKernel(
            program,
            std::string(kKernels) + "gguf_matmul_writer.cpp",
            ranges,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = parity ? NOC::NOC_0 : NOC::NOC_1,
                .compile_args = writer_ct});
    }
    const KernelHandle compute_kernel = CreateKernel(
        program,
        std::string(kKernels) + "gguf_matmul_compute.cpp",
        compute,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .compile_args = {f.format_id, k_tiles, chunk, in0_rows}});

    for (size_t i = 0; i < cores.size(); i++) {
        const uint32_t parity = cores[i].x % 2;
        std::vector<uint32_t> reader_args = {uint32_t(weight.buffer()->address()), n_tiles, 0};
        std::vector<uint32_t> writer_args = {
            i == collector, uint32_t(input.buffer()->address()), uint32_t(output.buffer()->address())};
        writer_args.insert(writer_args.end(), rect[parity].begin(), rect[parity].end());
        writer_args.insert(writer_args.end(), {uint32_t(grid.x * grid.y), uint32_t(plan[i].size())});
        std::vector<uint32_t> compute_args = {uint32_t(plan[i].size())};
        for (const Segment& s : plan[i]) {
            if (s.g1 > s.g0) {
                reader_args.insert(reader_args.end(), {s.n, s.g0, s.g1});
                reader_args[2]++;
            }
            if (s.role == kTail) {
                writer_args.insert(
                    writer_args.end(), {s.slot, s.role, uint32_t(virt[s.partner].x), uint32_t(virt[s.partner].y)});
            } else {
                writer_args.insert(writer_args.end(), {s.n, s.role, s.parts, 0});
            }
            compute_args.insert(
                compute_args.end(),
                {s.role == kFinish ? s.parts : s.g0 * f.group_tiles, s.g1 * f.group_tiles, uint32_t(s.role)});
        }
        SetRuntimeArgs(program, shared.reader[parity], cores[i], reader_args);
        SetRuntimeArgs(program, shared.writer[parity], cores[i], writer_args);
        SetRuntimeArgs(program, compute_kernel, cores[i], compute_args);
    }
    return {std::move(program), std::move(shared)};
}

void GgufMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const shared_variables_t& shared = cached_program.shared_variables;
    for (const CoreCoord& core : shared.cores) {
        const uint32_t parity = core.x % 2;
        GetRuntimeArgs(cached_program.program, shared.reader[parity], core)[0] = inputs.weight.buffer()->address();
        auto& writer_args = GetRuntimeArgs(cached_program.program, shared.writer[parity], core);
        writer_args[1] = inputs.input.buffer()->address();
        writer_args[2] = output.buffer()->address();
    }
}

}  // namespace ttnn::prim
