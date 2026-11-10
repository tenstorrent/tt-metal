// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace gguf_program;

namespace {

constexpr uint32_t kMaxPartials = 7;    // a finishing core adds its fp32 partial and up to 7 others in DEST's 8 tiles
constexpr uint32_t kInputChunk = 16;    // input tiles per multicast
constexpr uint32_t kMaxSegments = 64;   // keeps each kernel's runtime arguments within the limit
constexpr uint32_t kScaleTiles = 3;     // TQ2_0: scale tiles in c_3
constexpr uint32_t kTq2Unit = 2;        // TQ2_0: records per unit of reads (measured, decode)
constexpr uint32_t kTq2Units = 8;       // TQ2_0: units c_0 holds
constexpr uint32_t kTq2InFlight = 2;    // TQ2_0: units in flight

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

// K groups [g0, g1) on grid rows [row0, row0 + rows): only these rows receive this share of the input. Plan cores
// [core0, core1).
struct KBand {
    uint32_t g0, g1, row0, rows;
    uint32_t core0 = 0, core1 = 0;
};

// Each band's (N tile, K group) items in N-tile order, split evenly over at most its cores. The piece of an N tile
// that starts at K group 0 holds the tile: whole when no other core has a piece of it, else a head whose finish adds
// the other pieces' fp32 partials (tails). Per core, segments in compute order: heads (from the band's first K tile,
// so they run while the input streams in), wholes, tails by first K group (in a later band, a core's tails that start
// at the band's first K tile run before the one that starts inside it), finishes. A finish waits only for tails,
// which wait for nothing, so no chain of waits forms across cores. Fewer cores until no N tile spans more than
// kMaxPartials + 1 cores.
std::vector<std::vector<Segment>> plan_segments(uint32_t n_tiles, std::vector<KBand>& bands, uint32_t grid_x) {
    uint32_t most_cores = 0;
    for (const KBand& b : bands) {
        most_cores = std::max(most_cores, std::min(b.rows * grid_x, n_tiles * (b.g1 - b.g0)));
    }
    for (uint32_t c = most_cores;; c--) {
        std::vector<std::vector<Segment>> pieces;
        std::vector<uint32_t> spans(n_tiles, 0);
        for (KBand& band : bands) {
            const uint32_t groups = band.g1 - band.g0, items = n_tiles * groups;
            const uint32_t cores = std::min({c, band.rows * grid_x, items});
            band.core0 = pieces.size();
            for (uint32_t i = 0; i < cores; i++) {
                const uint32_t a = uint64_t(i) * items / cores, b = uint64_t(i + 1) * items / cores;
                std::vector<Segment>& core = pieces.emplace_back();
                for (uint32_t n = a / groups; n <= (b - 1) / groups; n++) {
                    const uint32_t g0 = std::max(a, n * groups) - n * groups;
                    const uint32_t g1 = std::min(b, (n + 1) * groups) - n * groups;
                    core.push_back({n, band.g0 + g0, band.g0 + g1, kWhole});
                    spans[n]++;
                }
            }
            band.core1 = pieces.size();
        }
        if (*std::max_element(spans.begin(), spans.end()) > kMaxPartials + 1) {
            continue;
        }
        std::vector<uint32_t> owner(n_tiles), first_slot(n_tiles), parts(n_tiles, 0);
        for (uint32_t i = 0; i < pieces.size(); i++) {
            for (const Segment& s : pieces[i]) {
                if (s.g0 == 0) {
                    owner[s.n] = i;
                }
            }
        }
        std::vector<std::vector<Segment>> plan(pieces.size());
        for (uint32_t i = 0; i < pieces.size(); i++) {
            std::vector<Segment> heads, wholes, tails, finishes;
            uint32_t slots = 0;  // c_18 holds each finish's partials together, in finish order
            for (const Segment& s : pieces[i]) {
                if (s.g0 > 0) {
                    tails.push_back({s.n, s.g0, s.g1, kTail, owner[s.n]});
                } else if (spans[s.n] == 1) {
                    wholes.push_back(s);
                } else {
                    heads.push_back({s.n, s.g0, s.g1, kHead});
                    finishes.push_back({s.n, 0, 0, kFinish, 0, 0, spans[s.n] - 1});
                    first_slot[s.n] = slots;
                    slots += spans[s.n] - 1;
                }
            }
            std::stable_sort(
                tails.begin(), tails.end(), [](const Segment& a, const Segment& b) { return a.g0 < b.g0; });
            for (const auto* group : {&heads, &wholes, &tails, &finishes}) {
                plan[i].insert(plan[i].end(), group->begin(), group->end());
            }
        }
        for (auto& segments : plan) {
            for (Segment& s : segments) {
                if (s.role == kTail) {
                    s.slot = first_slot[s.n] + parts[s.n]++;
                }
            }
        }
        return plan;
    }
}

CoreRangeSet core_ranges(const std::vector<CoreCoord>& cores) {
    CoreRangeSet ranges;
    for (const CoreCoord& c : cores) {
        ranges = ranges.merge(CoreRangeSet(CoreRange(c)));
    }
    return ranges;
}

}  // namespace

GgufMatmulProgramFactory::cached_program_t GgufMatmulProgramFactory::create(
    const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const Tensor& input = inputs.input;
    const Tensor& weight = inputs.weight;
    auto* device = input.device();
    const GgufTraits& f = gguf_traits(weight.dtype());
    const uint32_t k_tiles = weight.logical_shape()[0] / 32, n_tiles = weight.logical_shape()[1] / 32;
    const uint32_t groups = k_tiles / f.group_tiles;
    const uint32_t rows = input.logical_shape().volume() / input.logical_shape()[-1];
    const uint32_t in0_rows = rows > 16 ? 32 : rows > 8 ? 16 : 8;
    // TQ2_0 and expanded TQ1_0 (TQ2_0's pages): block-float weight tiles, no decoding (gguf_matmul_tq2_*.cpp). With one
    // input row the scales apply once per window of 16 K groups (one DEST row each), otherwise once per K group.
    const bool tq2 = weight.dtype() == DataType::GGUF_TQ2_0 || weight.dtype() == DataType::GGUF_TQ1_0_EXPANDED;
    const bool one_row = tq2 && rows == 1;
    const uint32_t window = one_row ? 16 : 1;

    // K bands: when the input of all K tiles in c_1 and the other circular buffers (below) exceed L1, the grid's rows
    // split into bands, each taking a share of K (whole groups; one-row TQ2_0: whole windows) and holding only that.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_x = grid.x, grid_y = grid.y;
    const uint32_t l1_free = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t scale_tile_bytes = one_row ? 2048 : 4096;
    const bool lookup_counts = !tq2 && f.code_offset > 0 && !is_gguf_expanded(weight.dtype());
    // The weight's circular buffers, c_16 and c_17.
    const uint32_t fixed_l1 = (tq2 ? kTq2Units * kTq2Unit * f.page_bytes() + kScaleTiles * scale_tile_bytes
                                   : weight_reader_l1_bytes(f) + (lookup_counts ? 64 : 0)) +
                              2 * 2048 + 4096;
    const uint32_t units = (groups + window - 1) / window;  // shares are whole units of `window` K groups
    std::vector<KBand> bands;
    std::vector<std::vector<Segment>> plan;
    uint32_t band_tiles = 0, most_parts = 1, most_heads = 1;  // c_1, c_18, c_20 pages
    for (uint32_t count = 1;; count++) {
        TT_FATAL(
            count <= std::min(grid_y, kMaxPartials + 1),
            "GGUF matmul: the input of {} K tiles does not fit L1",
            k_tiles);
        bands.clear();
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t g0 = std::min(groups, units * i / count * window);
            const uint32_t g1 = i + 1 == count ? groups : std::min(groups, units * (i + 1) / count * window);
            bands.push_back({g0, g1, grid_y * i / count, grid_y * (i + 1) / count - grid_y * i / count});
        }
        plan = plan_segments(n_tiles, bands, grid_x);
        band_tiles = 0;
        most_parts = most_heads = 1;
        for (const KBand& band : bands) {
            band_tiles = std::max(band_tiles, (band.g1 - band.g0) * f.group_tiles);
        }
        for (const auto& segments : plan) {
            uint32_t parts = 0, heads = 0;
            for (const Segment& s : segments) {
                parts += s.parts;
                heads += s.role == kHead ? 1 : 0;
            }
            most_parts = std::max(most_parts, parts);
            most_heads = std::max(most_heads, heads);
        }
        if (band_tiles * 2048 + fixed_l1 + 4096 * (most_parts + most_heads) <= l1_free) {
            break;
        }
    }

    // N tiles in bank order when each N tile's records sit in one DRAM bank (page g * n_tiles + n of an interleaved
    // buffer is in bank page % banks), so that a run of a core's items is in one bank.
    const uint32_t banks = device->allocator()->get_num_banks(BufferType::DRAM);
    const bool bank_local = n_tiles % banks == 0;
    std::vector<uint32_t> order(n_tiles);
    for (uint32_t j = 0; j < n_tiles; j++) {
        order[j] = bank_local ? j % (n_tiles / banks) * banks + j / (n_tiles / banks) : j;
    }
    for (auto& segments : plan) {
        TT_FATAL(segments.size() <= kMaxSegments, "GGUF matmul: {} segments on a core", segments.size());
        for (Segment& s : segments) {
            s.n = order[s.n];
        }
    }

    // A band's plan core i runs on a core of its rows: row-major, or with bank-local records the free core nearest the
    // banks of its items (Manhattan distance from each bank's adjacent worker, weighted by the core's groups there),
    // cheapest pairs first.
    std::vector<CoreCoord> cores(plan.size());
    std::vector<uint32_t> band_of(plan.size());
    std::unordered_map<uint32_t, CoreCoord> adjacent;
    if (bank_local) {
        adjacent =
            device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0, distributed::MeshCoordinate(0, 0));
    }
    for (uint32_t b = 0; b < bands.size(); b++) {
        const KBand& band = bands[b];
        std::vector<CoreCoord> all;
        for (uint32_t i = 0; i < grid_x * band.rows; i++) {
            all.emplace_back(i % grid_x, band.row0 + i / grid_x);
        }
        for (uint32_t i = band.core0; i < band.core1; i++) {
            cores[i] = all[i - band.core0];
            band_of[i] = b;
        }
        if (!bank_local) {
            continue;
        }
        std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> pairs;  // cost, plan core, core
        for (uint32_t i = band.core0; i < band.core1; i++) {
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

    Program program = CreateProgram();
    const CoreRangeSet full(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    const CoreRangeSet compute = core_ranges(cores);

    // c_1, the band's share of the input, sits at the same address on every core of the grid: the band's collector
    // multicasts it there. One-row TQ2_0 reads it as [16, 32] tiles at 32-byte offsets.
    if (one_row) {
        add_cb(
            program,
            full,
            tt::CBIndex::c_1,
            tt::DataFormat::Float16_b,
            32,
            64 * band_tiles,
            tt::tt_metal::Tile({16, 32}));
    } else {
        add_cb(program, full, tt::CBIndex::c_1, tt::DataFormat::Float16_b, 2048, band_tiles);
    }
    // TQ2_0: whole records in 64-byte pages (the matmul indexes the weight tiles at 256-byte steps), and the scale
    // tiles, fp32: [16, 32] with one input row, else [32, 32].
    if (tq2) {
        add_cb(
            program, compute, tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, 64, kTq2Units * kTq2Unit * f.page_bytes() / 64);
        add_cb(
            program,
            compute,
            tt::CBIndex::c_3,
            tt::DataFormat::Float32,
            scale_tile_bytes,
            kScaleTiles,
            tt::tt_metal::Tile(one_row ? std::array<uint32_t, 2>{16, 32} : std::array<uint32_t, 2>{32, 32}));
    } else {
        add_weight_reader_cbs(program, compute, f);
    }
    add_cb(program, compute, tt::CBIndex::c_16, tt::DataFormat::Float16_b, 2048, 2);
    add_cb(program, compute, tt::CBIndex::c_17, tt::DataFormat::Float32, 4096, 1);
    add_cb(program, compute, tt::CBIndex::c_18, tt::DataFormat::UInt32, 4096, most_parts, tt::tt_metal::Tile());
    add_cb(program, compute, tt::CBIndex::c_20, tt::DataFormat::UInt32, 4096, most_heads, tt::tt_metal::Tile());
    const uint32_t input_sem = CreateSemaphore(program, full, 0);
    const uint32_t partial_sem = CreateSemaphore(program, compute, 0);
    // Codebook formats: grid-lookup requests from the reader to the writer of its core, the answers, the page address.
    const uint32_t request_sem = CreateSemaphore(program, compute, 0);
    const uint32_t answer_sem = CreateSemaphore(program, compute, 0);
    const uint32_t address_sem = CreateSemaphore(program, compute, 0);

    // Per band, the collector: its plan core nearest the middle of the band whose writer runs on NoC 0 (odd column), if
    // any. The multicast rectangle is the band's rows, corners in the writer's NoC order.
    std::vector<size_t> collector;
    std::vector<std::array<std::array<uint32_t, 4>, 2>> rect;
    for (const KBand& band : bands) {
        size_t c = band.core0;
        uint32_t best = UINT32_MAX;
        for (size_t i = band.core0; i < band.core1; i++) {
            const uint32_t d = std::abs(int(cores[i].x) - int(grid_x / 2)) +
                               std::abs(int(cores[i].y) - int(band.row0 + band.rows / 2));
            if (cores[i].x % 2 == 1 && d < best) {
                c = i;
                best = d;
            }
        }
        collector.push_back(c);
        const CoreCoord v0 = device->worker_core_from_logical_core(CoreCoord(0, band.row0));
        const CoreCoord v1 = device->worker_core_from_logical_core(CoreCoord(grid_x - 1, band.row0 + band.rows - 1));
        rect.push_back(
            {{{uint32_t(v1.x), uint32_t(v1.y), uint32_t(v0.x), uint32_t(v0.y)},
              {uint32_t(v0.x), uint32_t(v0.y), uint32_t(v1.x), uint32_t(v1.y)}}});
    }

    const std::map<std::string, std::string> defines = lookup_defines(weight);
    if (lookup_counts) {
        // The page counts of the compute kernel's lookups (gguf_matmul_compute.cpp).
        add_cb(program, compute, tt::CBIndex::c_3, tt::DataFormat::UInt32, 64, 1, tt::tt_metal::Tile());
    }

    std::vector<uint32_t> reader_ct =
        tq2 ? std::vector<uint32_t>{kTq2Units * kTq2Unit, kTq2Unit, kTq2InFlight, window, kScaleTiles, scale_tile_bytes}
            : weight_reader_args(weight, request_sem, answer_sem, address_sem);
    if (tq2) {
        TensorAccessorArgs(*weight.buffer()).append_to(reader_ct);
    }
    const uint32_t chunk = std::min(kInputChunk, band_tiles);
    std::vector<uint32_t> writer_ct = {
        band_tiles,
        chunk,
        input_sem,
        partial_sem,
        f.format_id,
        request_sem,
        answer_sem,
        address_sem,
        one_row ? 1u : 0u,
        bands.size() > 1 ? 1u : 0u};
    TensorAccessorArgs(*input.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

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
        add_movers(
            program,
            core_ranges(sub),
            parity,
            tq2 ? "gguf_matmul_tq2_reader.cpp" : "gguf_matmul_reader.cpp",
            "gguf_matmul_writer.cpp",
            reader_ct,
            writer_ct,
            defines,
            defines,
            shared);
    }
    // One compute kernel per band: the band's first K tile and K tiles are compile-time.
    std::vector<KernelHandle> compute_kernels;
    for (const KBand& band : bands) {
        const std::vector<CoreCoord> band_cores(cores.begin() + band.core0, cores.begin() + band.core1);
        const uint32_t k0 = band.g0 * f.group_tiles, band_k = (band.g1 - band.g0) * f.group_tiles;
        compute_kernels.push_back(
            band_cores.empty()
                ? KernelHandle{}
                : CreateKernel(
                      program,
                      std::string(kKernels) + (tq2 ? "gguf_matmul_tq2_compute.cpp" : "gguf_matmul_compute.cpp"),
                      core_ranges(band_cores),
                      ComputeConfig{
                          // TQ2_0: HiFi3 for exact block-float products. A decoded MXFP4 weight has a 2-bit
                          // significand, which LoFi's SrcA bits hold: HiFi2's second phase would add 0, so LoFi gives
                          // the same output, half the MVMULs.
                          .math_fidelity = tq2                                      ? MathFidelity::HiFi3
                                           : weight.dtype() == DataType::GGUF_MXFP4 ? MathFidelity::LoFi
                                                                                    : MathFidelity::HiFi2,
                          .fp32_dest_acc_en = true,
                          .dst_full_sync_en = true,
                          .compile_args = tq2 ? std::vector<uint32_t>{band_k, chunk, one_row ? 1u : 0u, window, k0}
                                              : std::vector<uint32_t>{f.format_id, band_k, chunk, in0_rows, k0},
                          .defines = defines}));
    }

    for (size_t i = 0; i < cores.size(); i++) {
        const uint32_t parity = cores[i].x % 2;
        const KBand& band = bands[band_of[i]];
        const uint32_t k0 = band.g0 * f.group_tiles, band_k = (band.g1 - band.g0) * f.group_tiles;
        std::vector<uint32_t> reader_args = {uint32_t(weight.buffer()->address()), n_tiles, 0};
        std::vector<uint32_t> writer_args = {
            i == collector[band_of[i]], uint32_t(input.buffer()->address()), uint32_t(output.buffer()->address())};
        writer_args.insert(writer_args.end(), rect[band_of[i]][parity].begin(), rect[band_of[i]][parity].end());
        writer_args.insert(writer_args.end(), {grid_x * band.rows, uint32_t(plan[i].size())});
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
        if (bands.size() > 1) {
            writer_args.insert(writer_args.end(), {k0, band_k});
        }
        SetRuntimeArgs(program, shared.reader[parity], cores[i], reader_args);
        SetRuntimeArgs(program, shared.writer[parity], cores[i], writer_args);
        SetRuntimeArgs(program, compute_kernels[band_of[i]], cores[i], compute_args);
    }
    return {std::move(program), std::move(shared)};
}

void GgufMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    override_addresses(cached_program, inputs, output, 1);
}

}  // namespace ttnn::prim
